"""Render dispatcher — orchestration glue between routing and execution.

Single public entry point: ``dispatch_render(job, *, user_id, ...)``.
Decides where the job runs via the router director's
``maybe_route_render``, then either runs the local executor or ships
the job to a peer over fabric. Returns a RenderResult either way.

Solo deployments (no fabric, no director) skip the routing decision
and run local directly — this preserves the local-first invariant
for users without paired peers and keeps the surface usable when
fabric is disabled.

Never raises for routing or execution failures. Callers check
``result.ok`` + ``result.code`` instead. The exception path is
reserved for truly unexpected programming errors (e.g. bad arg
types) — those still raise so they get caught in tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.cast.executors import execute_local_render
from augmentum.cast.render import RenderJob, RenderResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

    from augmentum.cast.html_renderer import HTMLRenderer
    from augmentum.cast.output_store import RenderOutputStore
    from augmentum.fabric.director import RoutingDirector

log = get_logger(__name__)


async def dispatch_render(
    job: RenderJob,
    *,
    user_id: str,
    director: RoutingDirector | None = None,
    http_client: httpx.AsyncClient | None = None,
    html_renderer: HTMLRenderer | None = None,
    output_store: RenderOutputStore | None = None,
) -> RenderResult:
    """Route + execute a render job.

    Args:
      job: what to render, for which target.
      user_id: owner of the target device — passed through to peer
        calls for signed-envelope auth on the receiving side. Required
        even for local-only paths so the contract stays uniform.
      director: routing brain. ``None`` is valid — single-machine
        installs without fabric pass None and we run local directly.
      http_client: needed only when the route lands on a peer. ``None``
        is fine for solo installs.
      html_renderer / output_store: backends the local executor uses
        for real (non-stub) RENDER_HTML. ``None`` is fine — executor
        falls back to a stub and the dispatch still completes cleanly.

    Returns:
      RenderResult with ``ok=True`` and an ``output_url`` on success.
      Failed dispatches return ``ok=False`` with a code identifying
      what went wrong (no_capable_node / peer_unreachable / etc).
    """
    # No director = no fabric. Always run local; matches the single-
    # machine grace contract: solo installs work without any fabric
    # plumbing in scope.
    if director is None:
        return await execute_local_render(
            job,
            user_id=user_id,
            html_renderer=html_renderer,
            output_store=output_store,
        )

    route = await director.maybe_route_render(job=job)

    if route is None:
        log.warning(
            "cast_render_dispatch_no_capable_node",
            kind=job.kind, target_device_id=job.target_device_id,
        )
        return RenderResult(
            ok=False,
            code="no_capable_node",
            message=f"no node can render kind={job.kind!r}",
        )

    if route.location == "local":
        return await execute_local_render(
            job,
            node_id=route.node_id,
            user_id=user_id,
            html_renderer=html_renderer,
            output_store=output_store,
        )

    # Peer route. Look up transport details and delegate.
    if http_client is None:
        log.warning(
            "cast_render_dispatch_missing_http_client",
            node_id=route.node_id,
        )
        return RenderResult(
            ok=False,
            location="peer",
            node_id=route.node_id,
            code="missing_http_client",
            message="peer route requires an http_client",
        )

    peer_state = director._coordinator.peer_state(route.node_id)  # noqa: SLF001
    if peer_state is None or peer_state.paired is None:
        return RenderResult(
            ok=False,
            location="peer",
            node_id=route.node_id,
            code="peer_state_missing",
            message=f"peer {route.node_id!r} not registered or unpaired",
        )

    # Lazy import so single-machine installs never load the fabric
    # render-client module.
    from augmentum.fabric.render_client import render_via_peer
    return await render_via_peer(
        http_client=http_client,
        identity=director._coordinator._identity,  # noqa: SLF001
        user_id=user_id,
        peer_node_id=route.node_id,
        peer_addr=peer_state.paired.addr,
        job=job,
    )
