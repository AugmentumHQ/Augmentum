"""Model lifecycle management across backends."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from urllib.parse import quote

import httpx

from augmentum.config import settings
from augmentum.models.base import ModelInfo
from augmentum.models.provider_registry import ProviderRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelStatus:
    """Detailed status of a model."""

    name: str
    available: bool
    backend: str
    size: int = 0
    quantization: str = ""
    parameter_count: str = ""
    loaded: bool = False
    vram_usage: int = 0


@dataclass
class RunningModel:
    """A currently loaded/running model."""

    name: str
    backend: str
    size_vram: int = 0
    size_ram: int = 0
    expires_at: str = ""
    details: dict = field(default_factory=dict)


class ModelManager:
    """Manages model lifecycle across backends."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry
        # Per-backend list_models failure state for log coalescing: maps
        # backend name -> last error-type key. Warn ONCE when a backend
        # starts failing (or its error changes), debug on repeats, and log a
        # recovery when it lists again. Without this, an intermittently-off
        # backend (e.g. a custom engine under test) warned every probe cycle
        # — 433 list_models_failed in 24h for one offline backend.
        self._list_failure_state: dict[str, str] = {}
        self._hf_file_size_cache: dict[tuple[str, str], tuple[float, int]] = {}
        self._hf_size_cache_ttl = 3600.0
        self._hf_missing_size_ttl = 60.0
        self._hf_probe_semaphore = asyncio.Semaphore(6)

    def _hf_headers(self, *, ranged: bool = False) -> dict[str, str]:
        headers = {
            "User-Agent": "Augmentum/1.0",
            "Accept": "application/octet-stream",
        }
        if settings.huggingface_token:
            headers["Authorization"] = f"Bearer {settings.huggingface_token}"
        if ranged:
            headers["Range"] = "bytes=0-0"
        return headers

    @staticmethod
    def _hf_extract_size(headers: httpx.Headers) -> int:
        for key in ("x-linked-size", "x-file-size"):
            value = headers.get(key)
            if value:
                with contextlib.suppress(TypeError, ValueError):
                    size = int(value)
                    if size > 0:
                        return size

        content_range = headers.get("content-range")
        if content_range and "/" in content_range:
            with contextlib.suppress(TypeError, ValueError):
                size = int(content_range.rsplit("/", 1)[-1])
                if size > 0:
                    return size

        content_length = headers.get("content-length")
        if content_length:
            with contextlib.suppress(TypeError, ValueError):
                size = int(content_length)
                if size > 0:
                    return size

        return 0

    @staticmethod
    def _hf_resolve_url(repo_id: str, filename: str) -> str:
        repo = quote(repo_id, safe="/")
        path = quote(filename, safe="/")
        return f"https://huggingface.co/{repo}/resolve/main/{path}"

    async def resolve_hf_file_size(self, repo_id: str, filename: str) -> int:
        cache_key = (repo_id, filename)
        cached = self._hf_file_size_cache.get(cache_key)
        now = time.monotonic()
        if cached:
            cached_ttl = self._hf_size_cache_ttl if cached[1] > 0 else self._hf_missing_size_ttl
            if (now - cached[0]) < cached_ttl:
                return cached[1]

        url = self._hf_resolve_url(repo_id, filename)
        timeout = httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=10.0)

        async with self._hf_probe_semaphore:
            for method, ranged in (("HEAD", False), ("GET", True)):
                try:
                    async with httpx.AsyncClient(
                        follow_redirects=True,
                        timeout=timeout,
                    ) as client:
                        resp = await client.request(method, url, headers=self._hf_headers(ranged=ranged))
                        resp.raise_for_status()
                        size = self._hf_extract_size(resp.headers)
                        if size > 0:
                            self._hf_file_size_cache[cache_key] = (now, size)
                            return size
                except Exception:
                    log.debug(
                        "hf_file_size_probe_failed",
                        repo=repo_id,
                        file=filename,
                        method=method,
                        exc_info=True,
                    )

        self._hf_file_size_cache[cache_key] = (now, 0)
        return 0

    async def fill_missing_hf_file_sizes(
        self,
        repo_id: str,
        files: list[dict],
        *,
        limit: int | None = None,
    ) -> list[dict]:
        if not files:
            return files

        enriched = [dict(file) for file in files]
        missing_indexes = [idx for idx, file in enumerate(enriched) if (file.get("size") or 0) <= 0]
        if limit is not None:
            missing_indexes = missing_indexes[:limit]
        if not missing_indexes:
            return enriched

        sizes = await asyncio.gather(
            *(
                self.resolve_hf_file_size(
                    repo_id,
                    (enriched[idx].get("filename") or enriched[idx].get("name") or ""),
                )
                for idx in missing_indexes
                if (enriched[idx].get("filename") or enriched[idx].get("name"))
            ),
            return_exceptions=True,
        )
        resolved_indexes = [
            idx for idx in missing_indexes if (enriched[idx].get("filename") or enriched[idx].get("name"))
        ]
        for idx, resolved in zip(resolved_indexes, sizes, strict=False):
            if isinstance(resolved, Exception):
                log.debug(
                    "hf_file_size_enrichment_failed",
                    repo=repo_id,
                    file=(enriched[idx].get("filename") or enriched[idx].get("name") or ""),
                    error=str(resolved),
                )
                continue
            if resolved and resolved > 0:
                enriched[idx]["size"] = resolved

        return enriched

    async def list_all_models(self) -> list[ModelInfo]:
        """List models from all registered backends, merged.

        Probes are run IN PARALLEL with a per-backend deadline so one slow
        cloud provider (deepseek/openrouter on a degraded route) can't
        block the Model Manager UI for the sum of its connect timeouts.
        Failures are logged but don't gate the response — matches the
        same pattern in ``ollama_routes.ollama_tags`` and
        ``provider_registry.refresh_model_map``.
        """
        import asyncio

        _PROBE_DEADLINE_S = 6.0
        # Routing-only backends (e.g. the secondary engine slot) share the
        # primary's GGUF catalog and must not be enumerated here — listing
        # them would both duplicate every model and inflate ``n_backends``,
        # tagging otherwise-unique models with a `` (backend)`` suffix.
        _is_excluded = getattr(self._registry, "is_listing_excluded", None)

        def _excluded(k: str) -> bool:
            # ``is True`` so a MagicMock registry (returns a truthy Mock)
            # doesn't accidentally exclude every backend in tests; the real
            # registry returns a plain bool.
            return callable(_is_excluded) and _is_excluded(k) is True

        listable = {
            k: b
            for k, b in self._registry.backends.items()
            if not _excluded(k)
        }
        n_backends = len(listable)

        async def _probe(name: str, backend) -> tuple[str, list[ModelInfo] | None]:
            try:
                models = await asyncio.wait_for(
                    backend.list_models(), timeout=_PROBE_DEADLINE_S,
                )
                # Recovery: a backend that was failing now lists again.
                if self._list_failure_state.pop(name, None) is not None:
                    log.info("list_models_recovered", backend=name)
                return name, list(models)
            except Exception as exc:
                # Coalesce per (backend, error-type): warn the first time a
                # backend fails (or when its failure MODE changes), debug on
                # repeats. An intermittently-offline backend under test
                # shouldn't spam a warning every probe cycle.
                is_timeout = isinstance(exc, asyncio.TimeoutError)
                err_key = "timeout" if is_timeout else type(exc).__name__
                event = "list_models_timeout" if is_timeout else "list_models_failed"
                if self._list_failure_state.get(name) != err_key:
                    self._list_failure_state[name] = err_key
                    if is_timeout:
                        log.warning(event, backend=name, timeout_s=_PROBE_DEADLINE_S)
                    else:
                        log.warning(event, backend=name, exc_info=True)
                else:
                    log.debug(event, backend=name, repeat=True)
                return name, None

        probe_results = await asyncio.gather(
            *(_probe(n, b) for n, b in listable.items()),
            return_exceptions=False,
        )
        all_models: list[ModelInfo] = []
        for name, models in probe_results:
            if not models:
                continue
            if n_backends > 1:
                # Display-only disambiguation — build COPIES, never mutate in
                # place. Backends cache and return SHARED ModelInfo instances
                # (OpenAICompatibleBackend._list_models_cache), so mutating
                # ``m.name`` here permanently poisons that cache: the
                # provider_registry probe then reads the polluted
                # "name (provider)" and maps the model under an unrequestable
                # key, so the cloud model silently becomes unroutable (and the
                # suffix compounds — "name (p) (p)" — on every call). dataclasses
                # .replace keeps the cached objects pristine.
                models = [replace(m, name=f"{m.name} ({name})") for m in models]
            all_models.extend(models)
        return all_models

    async def get_model_status(self, model_name: str) -> ModelStatus:
        """Get status of a specific model."""
        for backend_name, backend in self._registry.backends.items():
            try:
                details = await backend.show_model(model_name)
                return ModelStatus(
                    name=model_name,
                    available=True,
                    backend=backend_name,
                    quantization=details.quantization_level or "",
                    parameter_count=details.parameter_size or "",
                )
            except Exception:
                log.debug("model_status_probe_failed", model=model_name, backend=backend_name, exc_info=True)
                continue
        return ModelStatus(name=model_name, available=False, backend="unknown")

    async def get_running_models(self) -> list[RunningModel]:
        """Get currently loaded models from all backends that support it.

        Runs on the resource-ledger collect path, so every backend probe is
        wrapped in a hard deadline: a wedged Ollama / llama.cpp must never
        stall the snapshot. Mirrors ``list_models``' per-backend
        ``asyncio.wait_for`` guard (a timeout there used to be missing here,
        leaving an unbounded ``await`` on the hot path).
        """
        import asyncio

        _RUNNING_DEADLINE_S = 4.0
        results: list[RunningModel] = []

        # Ollama: /api/ps
        ollama = self._registry.get_backend("ollama")
        if ollama is not None:
            try:
                resp = await asyncio.wait_for(
                    ollama._client.get(f"{ollama._base_url}/api/ps"),
                    timeout=_RUNNING_DEADLINE_S,
                )
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("models", []):
                    results.append(RunningModel(
                        name=m.get("name", ""),
                        backend="ollama",
                        size_vram=m.get("size_vram", 0),
                        size_ram=m.get("size", 0) - m.get("size_vram", 0),
                        expires_at=m.get("expires_at", ""),
                        details=m.get("details", {}),
                    ))
            except Exception:  # incl. TimeoutError from wait_for
                log.warning("get_running_models_ollama_failed", exc_info=True)

        # llama.cpp: slots or router models
        from augmentum.models.llama_cpp import LlamaCppBackend

        llamacpp = self._registry.get_backend("llamacpp")
        if isinstance(llamacpp, LlamaCppBackend):
            try:
                async def _probe_llamacpp() -> list[RunningModel]:
                    out: list[RunningModel] = []
                    if await llamacpp.is_router_mode():
                        for m in await llamacpp.list_router_models():
                            if m.get("status") == "loaded":
                                out.append(RunningModel(
                                    name=m.get("model", m.get("id", "unknown")),
                                    backend="llamacpp",
                                    details=m,
                                ))
                    else:
                        slots = await llamacpp.get_slots()
                        seen: set[str] = set()
                        for slot in slots:
                            model_name = slot.get("model", "unknown")
                            if model_name not in seen:
                                seen.add(model_name)
                                out.append(RunningModel(
                                    name=model_name,
                                    backend="llamacpp",
                                    details=slot,
                                ))
                    return out

                results.extend(
                    await asyncio.wait_for(_probe_llamacpp(), timeout=_RUNNING_DEADLINE_S)
                )
            except Exception:  # incl. TimeoutError from wait_for
                log.warning("get_running_models_llamacpp_failed", exc_info=True)

        return results

    async def pull_model(self, name: str) -> AsyncIterator[dict]:
        """Pull/download a model, yielding progress updates."""
        import json

        backend = self._registry.get_backend("ollama")
        if backend is None:
            yield {"error": "Ollama backend not available"}
            return

        async with backend._client.stream(
            "POST",
            f"{backend._base_url}/api/pull",
            json={"name": name},
        ) as resp:
            async for line in resp.aiter_lines():
                if line.strip():
                    try:
                        yield json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        yield {"status": line}

    async def load_model(self, name: str, backend_key: str | None = None) -> bool:
        """Pre-load a model into memory.

        Supports Ollama (empty chat with keep_alive) and llama.cpp (router
        mode load_model).
        """
        from augmentum.models.llama_cpp import LlamaCppBackend

        # If a specific backend is requested, try that one
        if backend_key:
            backend = self._registry.get_backend(backend_key)
            if isinstance(backend, LlamaCppBackend):
                if await backend.is_router_mode():
                    return await backend.load_model(name)
                log.debug("llamacpp_load_skipped", reason="not router mode")
                return False
        else:
            backend_key = "ollama"

        # Default: Ollama load
        backend = self._registry.get_backend("ollama")
        if backend is None:
            # Try llamacpp as fallback
            llamacpp = self._registry.get_backend("llamacpp")
            if isinstance(llamacpp, LlamaCppBackend) and await llamacpp.is_router_mode():
                return await llamacpp.load_model(name)
            return False
        try:
            resp = await backend._client.post(
                f"{backend._base_url}/api/chat",
                json={"model": name, "messages": [], "keep_alive": "5m"},
            )
            return resp.status_code == 200
        except Exception:
            log.debug("model_load_failed", model=name, exc_info=True)
            return False

    async def unload_model(self, name: str, backend_key: str | None = None) -> bool:
        """Unload a model from memory.

        Supports Ollama (chat with keep_alive=0) and llama.cpp (router
        mode unload_model).
        """
        from augmentum.models.llama_cpp import LlamaCppBackend

        if backend_key:
            backend = self._registry.get_backend(backend_key)
            if isinstance(backend, LlamaCppBackend):
                if await backend.is_router_mode():
                    return await backend.unload_model(name)
                log.debug("llamacpp_unload_skipped", reason="not router mode")
                return False

        # Default: Ollama unload
        backend = self._registry.get_backend("ollama")
        if backend is None:
            llamacpp = self._registry.get_backend("llamacpp")
            if isinstance(llamacpp, LlamaCppBackend) and await llamacpp.is_router_mode():
                return await llamacpp.unload_model(name)
            return False
        try:
            resp = await backend._client.post(
                f"{backend._base_url}/api/chat",
                json={"model": name, "messages": [], "keep_alive": 0},
            )
            return resp.status_code == 200
        except Exception:
            log.debug("model_unload_failed", model=name, exc_info=True)
            return False

    # --- GGUF / llama.cpp model management ---

    async def list_gguf_files(self, repo_id: str) -> list[dict]:
        """List .gguf files available in a HuggingFace repo."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("huggingface_hub is not installed. Run: pip install huggingface-hub") from None

        api = HfApi()
        try:
            repo_info = await asyncio.to_thread(api.repo_info, repo_id)
        except Exception as exc:
            raise ValueError(f"Could not fetch repo info for '{repo_id}': {exc}") from exc

        siblings = repo_info.siblings or []
        results = []
        for s in siblings:
            if s.rfilename.endswith(".gguf"):
                results.append({
                    "filename": s.rfilename,
                    "size": s.size or 0,
                })
        results = sorted(results, key=lambda x: x["filename"])
        return await self.fill_missing_hf_file_sizes(repo_id, results)

    def list_local_gguf(self, model_dir: str) -> list[dict]:
        """List locally downloaded GGUF files."""
        results = []
        if not os.path.exists(model_dir):
            return results

        for entry in os.scandir(model_dir):
            if entry.is_file() and entry.name.endswith(".gguf"):
                stat = entry.stat()
                results.append({
                    "filename": entry.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })
            elif entry.is_dir():
                # Check for .gguf files in subdirectories (hf_hub_download structure)
                for sub in os.scandir(entry.path):
                    if sub.is_file() and sub.name.endswith(".gguf"):
                        stat = sub.stat()
                        results.append({
                            "filename": f"{entry.name}/{sub.name}",
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                        })

        return sorted(results, key=lambda x: x["filename"])
