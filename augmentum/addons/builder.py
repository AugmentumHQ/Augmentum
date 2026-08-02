"""Build an add-on image from the recipe that ships in this repo.

Two things make this safe enough to justify turning ``BUILD=1`` on in the
docker-socket-proxy ACL (see the annotated comment in compose.yaml):

1. **The context is narrow.** Only ``./services/game-stream`` is mounted
   into the augmentum container, read-only, and only that directory is
   tarred. Every ``COPY`` in the four game-stream Dockerfiles reads from
   ``services/game-stream/scripts/``, so nothing else is needed and the repo
   root is never sent to the daemon.
2. **Nothing is taken from request input.** The Dockerfile path, the image
   tag and every build arg come from :mod:`augmentum.addons.catalog`. The
   install route supplies an add-on *id* and nothing else.

Progress is derived from Docker's own ``Step N/M`` lines rather than
invented. When the daemon stops emitting step counts (BuildKit output, a
cached layer burst), the fraction holds instead of jumping — a bar that
stalls honestly beats one that lies smoothly.
"""

from __future__ import annotations

import io
import re
import tarfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from augmentum.addons.catalog import BUILD_CONTEXT_ROOT, AddonSpec
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_STEP_RE = re.compile(r"^Step (\d+)/(\d+)")


class AddonBuildError(RuntimeError):
    """Raised with the daemon's own error text, which is the actionable part."""


def build_context_tar(spec: AddonSpec) -> io.BytesIO:
    """Tar ``services/<context_subdir>`` with repo-relative arcnames.

    The arcname prefix mirrors the repo layout so a Dockerfile's ``COPY
    services/game-stream/scripts/x`` resolves identically whether the build
    ran here or via ``docker compose build`` on the host. One recipe, two
    build paths, no divergence.
    """
    src = Path(BUILD_CONTEXT_ROOT) / spec.context_subdir
    if not src.is_dir():
        raise AddonBuildError(
            f"add-on build context {src} is not mounted into this container. "
            f"Add './services/{spec.context_subdir}:{BUILD_CONTEXT_ROOT}/"
            f"{spec.context_subdir}:ro' to the augmentum service in "
            f"compose.yaml and restart."
        )

    buf = io.BytesIO()
    prefix = f"services/{spec.context_subdir}"
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(src), arcname=prefix, recursive=True)
    buf.seek(0)
    return buf


async def build_addon(
    spec: AddonSpec,
    *,
    docker: Any,
    on_progress: Callable[[float, str], Any] | None = None,
) -> str:
    """Build ``spec``'s image. Returns the resulting image id.

    ``on_progress(fraction, stage)`` is awaited (if awaitable) for each
    meaningful step so the caller can forward it to the job queue.
    """
    context = build_context_tar(spec)
    dockerfile = f"services/{spec.dockerfile}"
    tag = spec.image

    log.info(
        "addon_build_start",
        addon=spec.id, image=tag, dockerfile=dockerfile,
        build_args=spec.build_args,
    )

    async def _emit(fraction: float, stage: str) -> None:
        if on_progress is None:
            return
        result = on_progress(fraction, stage)
        if hasattr(result, "__await__"):
            await result

    last_line = ""
    try:
        stream: AsyncIterator[dict[str, Any]] = docker.images.build(
            fileobj=context,
            encoding="gzip",
            path_dockerfile=dockerfile,
            tag=tag,
            buildargs=dict(spec.build_args),
            # rm: drop intermediate containers. pull=False keeps the build
            # reproducible against the base image already on the host —
            # `ubuntu:24.04` moving under us mid-catalog would defeat the
            # point of pinning the build args.
            rm=True,
            pull=False,
            stream=True,
        )
        async for chunk in stream:
            if not isinstance(chunk, dict):
                continue
            if "error" in chunk:
                detail = str(chunk.get("error") or "").strip()
                raise AddonBuildError(detail or "docker build failed")
            line = str(chunk.get("stream") or "").strip()
            if not line:
                continue
            last_line = line
            match = _STEP_RE.match(line)
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                # Reserve the head and tail of the bar for prep and
                # anchoring, which happen outside this function.
                fraction = 0.10 + 0.80 * (current / max(total, 1))
                await _emit(min(fraction, 0.90), f"building ({current}/{total})")
    except AddonBuildError:
        raise
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
        if status == 403:
            raise AddonBuildError(
                "the docker-socket-proxy refused POST /build (HTTP 403). "
                "Add-on installs need BUILD=1 in the docker-proxy environment "
                "block of compose.yaml, then `docker compose up -d "
                "docker-proxy`. To keep BUILD disabled, build add-ons on the "
                "host instead with `start.sh build` / `start.bat build`."
            ) from exc
        raise AddonBuildError(
            f"docker build failed ({type(exc).__name__}): {exc}. "
            f"Last output: {last_line[:200]}"
        ) from exc

    info = await docker.images.inspect(tag)
    image_id = str(info.get("Id") or "")
    log.info("addon_build_complete", addon=spec.id, image=tag, image_id=image_id[:19])
    return image_id
