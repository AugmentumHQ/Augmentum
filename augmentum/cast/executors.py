"""Local render executors.

Dispatches by ``job.kind`` to the right backend:

  - RENDER_HTML  → headless Chrome (HTMLRenderer) → PNG → output store
  - RENDER_VRM   → offscreen WebGL → still / stream (future)
  - RENDER_VIDEO → ffmpeg + hardware encoder (future)
  - RENDER_WEBRTC → aiortc peer connection setup (future)

Each kind that lacks a real backend (or whose backend isn't available
on this node — e.g. no Chrome binary) gracefully falls back to a stub
RenderResult. The orchestration loop keeps working; the result's
``metadata["stub"]`` flag tells the caller "the contract was honored
but no real render happened on this node."

Never raises — every failure path returns RenderResult(ok=False).
The dispatcher contract is "always return a RenderResult"; raising
would force every call site to wrap in try/except.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.cast.render import (
    RENDER_HTML,
    RenderJob,
    RenderResult,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.cast.html_renderer import HTMLRenderer
    from augmentum.cast.output_store import RenderOutputStore

log = get_logger(__name__)


async def execute_local_render(
    job: RenderJob,
    *,
    node_id: str = "",
    user_id: str = "",
    html_renderer: HTMLRenderer | None = None,
    output_store: RenderOutputStore | None = None,
) -> RenderResult:
    """Run a render job on this node.

    Returns RenderResult with ``ok=True`` + ``output_url`` on success.
    ``output_url`` is a relative path (e.g.
    ``/api/cast/render-output/ro_xxx``) — callers/consumers turn it
    absolute by joining with the rendering node's HTTPS edge.

    Failed renders return ``ok=False`` with a code; the orchestration
    loop continues unmodified.
    """
    if job.kind == RENDER_HTML and html_renderer is not None and output_store is not None:
        return await _render_html_branch(
            job, node_id=node_id, user_id=user_id,
            html_renderer=html_renderer, output_store=output_store,
        )

    # No backend wired for this kind (or this node) — stub result so
    # the orchestration loop is exercised end-to-end while the real
    # backend lands.
    log.info(
        "cast_render_local_stub",
        kind=job.kind,
        target_device_id=job.target_device_id,
        node_id=node_id,
    )
    return RenderResult(
        ok=True,
        location="local",
        node_id=node_id,
        output_url=f"stub://{job.kind}/{job.target_device_id or 'no-target'}",
        metadata={
            "stub": True,
            "kind": job.kind,
            "payload_keys": sorted(job.payload.keys()),
        },
    )


async def _render_html_branch(
    job: RenderJob,
    *,
    node_id: str,
    user_id: str,
    html_renderer: HTMLRenderer,
    output_store: RenderOutputStore,
) -> RenderResult:
    """Real HTML → PNG path. Payload keys understood:

      - ``html``: required, the document body to render
      - ``viewport_w`` / ``viewport_h``: optional, default 1920x1080
      - ``wait_for_load_s``: optional, default 5s
      - ``single_use``: optional, default False — set True for one-shot
        casts where the receiver only fetches once
    """
    html = str(job.payload.get("html") or "")
    if not html:
        return RenderResult(
            ok=False, location="local", node_id=node_id,
            code="payload_missing_html",
            message="RENDER_HTML payload must include an 'html' string",
        )

    try:
        png_bytes = await html_renderer.render_html_to_image(
            html,
            viewport_w=int(job.payload.get("viewport_w") or 1920),
            viewport_h=int(job.payload.get("viewport_h") or 1080),
            wait_for_load_s=float(job.payload.get("wait_for_load_s") or 5.0),
        )
    except Exception as exc:
        log.warning(
            "cast_render_html_failed",
            kind=job.kind, target_device_id=job.target_device_id,
            error=str(exc)[:200],
        )
        return RenderResult(
            ok=False, location="local", node_id=node_id,
            code="render_failed", message=str(exc)[:240],
        )

    stored = output_store.store(
        body=png_bytes,
        content_type="image/png",
        user_id=user_id,
        single_use=bool(job.payload.get("single_use")),
        metadata={
            "kind": job.kind,
            "target_device_id": job.target_device_id,
        },
    )

    return RenderResult(
        ok=True,
        location="local",
        node_id=node_id,
        output_url=f"/api/cast/render-output/{stored.token}",
        metadata={
            "kind": job.kind,
            "content_type": "image/png",
            "bytes": len(png_bytes),
            "single_use": stored.single_use,
        },
    )
