"""``addon_install`` job handler — staged, progress-reporting add-on build.

Adopts the same honest-progress contract as the Service Install Standard
(``service_install.py``), because the user-visible promise is identical even
though the work underneath is a build rather than a pull:

    preparing → building <dependency> → building <add-on> → anchoring → ready

"Ready" means USABLE: the job completes only after the image exists AND is
anchored, because an unanchored image is exactly the state that produced the
2026-07-25 disappearance. Reporting "ready" before anchoring would be
reporting a condition we know decays.

Builds are long (up to ~25 minutes for the emulator add-on, which compiles
Dolphin from source), which is precisely why this runs through the job queue:
an in-flight install survives a server restart, and the Discover card polls
``/api/jobs/{id}`` for real stages instead of showing a spinner for half an
hour.

Payload:
    {"addon_id": "game-stream-emulator"}

Idempotency: an add-on whose image already exists is re-anchored and skipped
rather than rebuilt, so a restart re-entry finishes the install instead of
starting the 25-minute build over.
"""

from __future__ import annotations

import contextlib
from typing import Any

from augmentum.jobs.context import JobContext
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def make_addon_install_handler(app):
    """Build the staged add-on-install handler bound to ``app.state``."""

    async def handler(ctx: JobContext) -> dict[str, Any]:
        # Lazy imports keep the jobs package free of a hard aiodocker dep.
        from augmentum.addons.builder import AddonBuildError, build_addon
        from augmentum.addons.catalog import addon_by_id, resolve_build_order
        from augmentum.addons.registry import ensure_anchor

        payload = ctx.payload or {}
        addon_id = str(payload.get("addon_id") or "").strip()
        spec = addon_by_id(addon_id)
        if spec is None:
            raise ValueError(f"addon_install: unknown add-on {addon_id!r}")

        await ctx.update_progress(0.01, stage="preparing")
        await ctx.check_cancel()

        # Dependencies first (the shared streaming base is the FROM parent of
        # every leaf image). The user asked for one add-on and should never
        # have to know the base exists.
        order = resolve_build_order(addon_id)

        docker = getattr(app.state, "docker_client", None)
        owned = False
        if docker is None:
            import aiodocker

            docker = aiodocker.Docker()
            owned = True

        built: list[str] = []
        skipped: list[str] = []
        try:
            # Each spec gets its own slice of the bar, weighted by measured
            # build time so the emulator's 25-minute compile doesn't look
            # like it stalled while the 8-minute base flew by.
            weights = [max(s.build_minutes, 1) for s in order]
            total_weight = sum(weights)
            consumed = 0

            for spec_i, weight in zip(order, weights, strict=True):
                await ctx.check_cancel()
                base = consumed / total_weight
                span = weight / total_weight

                already = ""
                with contextlib.suppress(Exception):
                    info = await docker.images.inspect(spec_i.image)
                    already = str(info.get("Id") or "")

                if already:
                    # Present from a host build, a prior install, or a
                    # restart re-entry. Re-anchor and move on -- rebuilding
                    # would cost 25 minutes to produce the same layers.
                    skipped.append(spec_i.id)
                    log.info("addon_build_skipped_present", addon=spec_i.id)
                else:
                    label = spec_i.title if spec_i.user_facing else "streaming runtime"

                    async def _progress(f: float, stage: str, _b=base, _s=span,
                                        _label=label) -> None:
                        await ctx.update_progress(
                            min(_b + _s * f, 0.97), stage=f"building {_label} — {stage}",
                        )

                    await ctx.update_progress(
                        min(base + span * 0.02, 0.97), stage=f"building {label}",
                    )
                    try:
                        await build_addon(spec_i, docker=docker, on_progress=_progress)
                    except AddonBuildError as exc:
                        # Build failures are almost never transient (a compile
                        # error, a 403 from the proxy ACL, no disk). Retrying a
                        # 25-minute build to fail identically wastes the user's
                        # evening, so fail loudly with the daemon's own text.
                        log.warning(
                            "addon_build_failed", addon=spec_i.id, error=str(exc),
                        )
                        raise
                    built.append(spec_i.id)

                consumed += weight

            # Anchor everything we touched, dependencies included. This is the
            # install record AND the prune protection -- see addons/registry.
            await ctx.update_progress(0.98, stage="anchoring")
            anchored: list[str] = []
            for spec_i in order:
                if await ensure_anchor(spec_i, docker=docker):
                    anchored.append(spec_i.id)
        finally:
            if owned:
                with contextlib.suppress(Exception):
                    await docker.close()

        await ctx.update_progress(1.0, stage="ready")
        log.info(
            "addon_installed",
            addon=addon_id, built=built, reused=skipped, anchored=anchored,
        )
        return {
            "addon_id": addon_id,
            "capability": spec.capability,
            "built": built,
            "reused": skipped,
            "anchored": anchored,
            "ready": True,
        }

    return handler
