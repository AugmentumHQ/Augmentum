"""TitleService -- the orchestrator.

Glues together TitleStore + SourceRegistry + RuntimeRegistry and is the
single object route handlers consume. Encapsulates:

* Manifest reads (``list``, ``get``, filters)
* Imports (route delegates to the chosen Source)
* Launches (resolves the runtime, records a ``title_runs`` row, returns
  the LaunchHandle + run_id)
* Stops (delegates to runtime, ends the run)

Errors mapped to ``TitleServiceError`` so route handlers can translate
them to 4xx/5xx without leaking implementation details.
"""

from __future__ import annotations

import time

from augmentum.titles.manifest import TitleManifest
from augmentum.titles.runtimes import LaunchHandle, RuntimeRegistry
from augmentum.titles.sources import DiscoveryItem, SourceImportError, SourceRegistry
from augmentum.titles.store import TitleStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class TitleServiceError(Exception):
    """Base for errors the route layer should surface as 400/404/500."""


class TitleNotFound(TitleServiceError):
    pass


class TitleNotPlayable(TitleServiceError):
    """No registered runtime supports this title."""


class TitleService:
    def __init__(
        self,
        *,
        store: TitleStore,
        sources: SourceRegistry,
        runtimes: RuntimeRegistry,
    ) -> None:
        self._store = store
        self._sources = sources
        self._runtimes = runtimes

    # ── Reads ──────────────────────────────────────────────────────

    async def list_titles(
        self,
        *,
        user_id: str,
        kind: str | None = None,
        pinned_only: bool = False,
        limit: int = 200,
    ) -> list[TitleManifest]:
        return await self._store.list_for_user(
            user_id=user_id,
            kind=kind,
            pinned_only=pinned_only,
            limit=limit,
        )

    async def get_title(
        self, title_id: str, *, user_id: str = "",
    ) -> TitleManifest:
        manifest = await self._store.get(title_id, user_id=user_id)
        if manifest is None:
            raise TitleNotFound(f"title {title_id!r} not found")
        return manifest

    # ── Discovery ─────────────────────────────────────────────────

    async def discover_titles(
        self,
        *,
        source_id: str,
        query: dict | None = None,
        user_id: str = "",
        decorate_installed: bool = True,
    ) -> list[DiscoveryItem]:
        """Browse a source's catalog. Returns DiscoveryItems.

        When ``decorate_installed`` is True (the default), each item's
        ``installed`` flag is filled in by checking the user's library
        for a matching (source_id, source_remote_id) pair. Skip the
        decoration when the caller doesn't care -- saves one query.
        """
        source = self._sources.get(source_id)
        if source is None:
            raise TitleServiceError(f"unknown source: {source_id!r}")
        items = await source.discover(query or {}, user_id=user_id)
        if not decorate_installed or not user_id or not items:
            return items

        installed_keys = await self._installed_keys(user_id=user_id)
        decorated: list[DiscoveryItem] = []
        for item in items:
            key = (item.source_id, item.source_remote_id)
            decorated.append(
                # frozen dataclass -- can't mutate, build a copy
                DiscoveryItem(
                    **{**item.__dict__, "installed": key in installed_keys},
                )
                if item.installed is None
                else item
            )
        return decorated

    async def _installed_keys(self, *, user_id: str) -> set[tuple[str, str]]:
        """Set of (source_id, source_remote_id) pairs the user owns.

        Cheap: one query, scans the artifacts metadata. Acceptable at
        library sizes the order of hundreds; promote to an indexed
        lookup if it ever needs to scale.
        """
        titles = await self._store.list_for_user(user_id=user_id, limit=500)
        return {
            (t.source_id, t.source_remote_id)
            for t in titles
            if t.source_remote_id
        }

    # ── Imports ────────────────────────────────────────────────────

    async def import_title(
        self,
        *,
        user_id: str,
        source_id: str,
        manifest_data: dict,
    ) -> tuple[TitleManifest, bool]:
        """Materialise a manifest from a Source into the user's library.

        Returns ``(manifest, created)``. ``created`` is True for newly
        inserted artifacts, False when the source detected a duplicate
        (e.g. internal-rom matching by sha256) and returned the
        pre-existing artifact id. Sources that don't implement
        de-dup (most of them) return a bare string -- we treat that
        as ``created=True``.
        """
        source = self._sources.get(source_id)
        if source is None:
            raise TitleServiceError(f"unknown source: {source_id!r}")
        try:
            result = await source.import_for_user(
                manifest_data, user_id=user_id,
            )
        except SourceImportError as exc:
            raise TitleServiceError(str(exc)) from exc

        # Sources that pre-date the de-dup contract return ``str``;
        # the rest (currently just InternalRomSource) return a
        # ``(artifact_id, created)`` tuple. Normalise both.
        if isinstance(result, tuple):
            artifact_id, created = result
        else:
            artifact_id, created = result, True

        manifest = await self._store.get(artifact_id, user_id=user_id)
        if manifest is None:
            # Source claimed success but the row didn't materialise as a
            # title -- defensive guard against partial inserts.
            raise TitleServiceError(
                "import succeeded but title row could not be projected",
            )
        return manifest, created

    # ── Pin / metadata ────────────────────────────────────────────

    async def set_pinned(
        self, title_id: str, *, user_id: str, pinned: bool,
    ) -> bool:
        return await self._store.set_pinned(
            title_id, user_id=user_id, pinned=pinned,
        )

    async def update_metadata(
        self, title_id: str, *, user_id: str, patch: dict,
    ) -> bool:
        return await self._store.update_metadata(
            title_id, user_id=user_id, patch=patch,
        )

    async def delete_title(
        self, title_id: str, *, user_id: str,
    ) -> TitleManifest:
        """Remove a title from the user's library.

        Returns the deleted manifest so the caller can reclaim any blobs
        it owned (ROM bytes, save slots). Raises ``TitleNotFound`` when
        there's no such title for this user.
        """
        manifest = await self._store.delete(title_id, user_id=user_id)
        if manifest is None:
            raise TitleNotFound(f"title {title_id!r} not found")
        return manifest

    # ── Launch / stop ─────────────────────────────────────────────

    async def launch(
        self,
        title_id: str,
        *,
        user_id: str,
        ctx: dict | None = None,
        prefer_runtime: str | None = None,
    ) -> dict:
        """Pick a runtime, start it, record a run, return the handle.

        ``ctx`` carries runtime-specific options (resolution, bitrate,
        encoder, world_id, ...). ``prefer_runtime`` overrides the
        manifest's preferred runtime if provided AND the runtime
        supports the title.

        Returns:
            {
              "run_id": str,
              "handle": LaunchHandle.__dict__-like dict,
              "manifest": manifest.to_dict(),
            }
        """
        manifest = await self.get_title(title_id, user_id=user_id)
        ctx = dict(ctx or {})
        ctx.setdefault("user_id", user_id)

        runtime = None
        if prefer_runtime:
            candidate = self._runtimes.get(prefer_runtime)
            if candidate is not None and await candidate.supports(manifest):
                runtime = candidate
        if runtime is None:
            runtime = await self._runtimes.resolve_for(manifest)
        if runtime is None:
            raise TitleNotPlayable(
                f"no runtime supports title {title_id!r} (kind={manifest.kind})"
            )

        # Time the launch so the run row records click->ready latency.
        # For the iframe runtime the latency is essentially zero (we
        # return the URL immediately); the actual paint time is browser
        # work the client measures separately. For streamed runtimes
        # this includes container start.
        t0 = time.perf_counter()
        try:
            handle = await runtime.launch(manifest, ctx)
        except Exception as exc:
            # BIOS-missing and core-unavailable are user-actionable
            # conditions (install the BIOS / wait for the streaming
            # runtime). Surface as TitleNotPlayable so the route maps to
            # 409 with the original message intact. Other launch
            # exceptions stay as opaque 400 errors.
            from augmentum.titles.runtimes import (
                BiosMissingError,
                CoreUnavailableError,
            )
            log.warning(
                "title_launch_failed",
                title_id=title_id,
                runtime_id=runtime.id,
                error=str(exc),
            )
            if isinstance(exc, (BiosMissingError, CoreUnavailableError)):
                raise TitleNotPlayable(str(exc)) from exc
            raise TitleServiceError(
                f"runtime {runtime.id!r} failed to launch: {exc}"
            ) from exc
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        run_id = await self._store.create_run(
            user_id=user_id,
            artifact_id=manifest.id,
            runtime_id=runtime.id,
            source_id=manifest.source_id,
            launch_latency_ms=elapsed_ms,
        )
        await self._store.touch_last_played(manifest.id, user_id=user_id)

        return {
            "run_id": run_id,
            "handle": _handle_to_dict(handle),
            "manifest": manifest.to_dict(),
        }

    async def end_run(
        self,
        run_id: str,
        *,
        user_id: str,
        runtime_id: str = "",
        session_id: str = "",
        exit_reason: str = "clean",
        avg_fps: float | None = None,
        avg_rtt_ms: float | None = None,
        avg_bitrate_kbps: int | None = None,
        crashes: int = 0,
        metadata: dict | None = None,
    ) -> bool:
        # Tear down the runtime side first (best-effort), then close
        # the run row. Order matters: an exception in stop() should
        # still leave the row closed.
        if runtime_id and session_id:
            runtime = self._runtimes.get(runtime_id)
            if runtime is not None:
                try:
                    await runtime.stop(session_id, user_id=user_id)
                except Exception as exc:
                    log.warning(
                        "title_runtime_stop_failed",
                        run_id=run_id,
                        runtime_id=runtime_id,
                        session_id=session_id,
                        error=str(exc),
                    )
        return await self._store.end_run(
            run_id,
            user_id=user_id,
            exit_reason=exit_reason,
            avg_fps=avg_fps,
            avg_rtt_ms=avg_rtt_ms,
            avg_bitrate_kbps=avg_bitrate_kbps,
            crashes=crashes,
            metadata=metadata,
        )

    # ── Telemetry / history ───────────────────────────────────────

    async def list_runs(
        self,
        *,
        user_id: str,
        title_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        return await self._store.list_runs(
            user_id=user_id, artifact_id=title_id, limit=limit,
        )

    # ── Registry passthroughs (route layer reads) ─────────────────

    def list_runtimes(self) -> list[dict]:
        return [
            {
                "id": rt.id,
                "label": rt.label,
                "capabilities": dict(rt.capabilities),
            }
            for rt in self._runtimes.list()
        ]

    def list_sources(self) -> list[dict]:
        return [
            {"id": src.id, "label": src.label}
            for src in self._sources.list()
        ]


def _handle_to_dict(handle: LaunchHandle) -> dict:
    return {
        "runtime_id": handle.runtime_id,
        "kind": handle.kind,
        "target": handle.target,
        "session_id": handle.session_id,
        "metadata": dict(handle.metadata),
    }
