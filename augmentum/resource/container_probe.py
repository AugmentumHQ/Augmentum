"""Probe managed sidecar containers (TTS / STT / sidecar LLMs) for the
resource panel.

The resource ledger tracks in-process and engine-managed models, but the
GPU-using *sidecar containers* — Higgs/Kokoro/Chatterbox TTS, Speaches/
Whisper STT, the classifier and vision sibling LLMs — are separate Docker
containers it never sees. This surfaces them as panel entries with an
owner label, device, and a ``container`` handle so the UI can pause
(docker stop → frees VRAM) and reload (docker start) them.

**RAM/CPU are measured** per container from the Docker stats API
(working-set = ``usage − inactive_file``; CPU% from the cpu/precpu delta) —
this works through WSL2 because only per-process *GPU* memory is opaque
there. **VRAM is not yet attributed here** (per-process VRAM isn't exposed
through WSL2 GPU passthrough — ``nvidia-smi --query-compute-apps`` returns
``[N/A]``); the accurate total-GPU bar already accounts for it, and the
per-sidecar VRAM self-report lands in a later slice. See
``docs/superpowers/specs/2026-06-19-resource-accounting-os-design.md``.

Non-blocking contract (§4.5 of the spec): every Docker call is wrapped in a
hard per-call timeout and the whole probe runs under a concurrency cap, so a
wedged daemon or a single hung container degrades to last-known rather than
stalling ``/status``. Results are cached for ``_CACHE_TTL_S``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# (name substring, subsystem, friendly owner label). First match wins.
_SIDECAR_PATTERNS: list[tuple[str, str, str]] = [
    ("classifier", "llm", "Classifier"),
    ("vision", "llm", "Vision sibling"),
    ("higgs", "tts", "Higgs TTS"),
    ("kokoro", "tts", "Kokoro TTS"),
    ("chatterbox", "tts", "Chatterbox TTS"),
    ("fish-tts", "tts", "Fish TTS"),
    ("qwen-tts", "tts", "Qwen TTS"),
    ("sesame", "tts", "Sesame CSM"),
    ("speaches", "stt", "Speaches STT"),
    ("whisper", "stt", "Whisper STT"),
    ("moonshine", "stt", "Moonshine STT"),
    ("docling", "ocr", "OCR (docling)"),
    ("ocr", "ocr", "OCR (docling)"),
]

# Hard infra — never a sidecar even if the name coincidentally contains a
# sidecar substring (guards e.g. an "ocr"/"vision" substring landing on real
# infra). This OVERRIDES a positive pattern match (see ``_classify``).
# NOTE: the primary-engine tokens (``ollama``/``llamacpp``/``llama-server``)
# are intentionally NOT here — they match no sidecar pattern so they're
# excluded anyway, and listing them risked dropping the classifier/vision
# siblings, which are themselves llama-servers.
_SKIP_SUBSTRINGS = (
    "augmentum-augmentum", "caddy", "searxng", "executor", "docker-proxy",
    "autoheal", "chatgpt-bridge", "codex", "coturn", "livekit", "rsshub",
    "game-stream", "ws-",
)

_CACHE_TTL_S = 6.0
# Hard per-call deadline for any single Docker API call (.show()/.stats()).
# host_probe uses 0.2/0.5s for a local agent; the Docker socket is local too,
# so a healthy call returns in single-digit ms and a wedged one fails fast
# instead of stalling the probe.
_DOCKER_CALL_TIMEOUT_S = 1.5
# Max concurrent Docker calls — fan out across sidecars so wall-clock is the
# slowest single call, not the sum, while bounding load on the daemon.
_CONCURRENCY = 8

_cache: dict = {"at": 0.0, "data": []}
# Single-flight gate. The ~2s live probe must run AT MOST once at a time: a
# ``/status`` request landing mid-sampler-pass (the cache still stale because
# the in-flight refresh hasn't published yet) must not launch its OWN duplicate
# probe on the request path. Both coalesce behind this lock; the loser re-reads
# the freshly-published cache. host_probe + ledger.collect already do this — this
# closes the gap the sampler's "read path never probes" contract assumed.
_probe_lock: asyncio.Lock | None = None
# Last-known entry per container id. Lets a single timed-out ``.show()`` /
# ``.stats()`` carry the previous value forward instead of blinking the row /
# its RAM to zero (spec §4.6: keep the last good reading, only drop a row when
# the container genuinely leaves ``containers.list``).
_last_by_cid: dict[str, dict] = {}
# Measured VRAM/RAM per container id, parsed ONCE from a sidecar llama-server's
# startup log banner (the classifier / vision siblings). Keyed by cid + the
# container's StartedAt so a restart re-parses; otherwise reused forever (the
# footprint is stable once the model is loaded — spec §4.6 rung B).
_vram_by_cid: dict[str, dict] = {}
# Only LLM-sibling containers carry a parseable VRAM banner; bound how much log
# we read so a long-running container doesn't drag in megabytes of output.
_LOG_TAIL_LINES = 6000
_LOG_WINDOW_S = 180  # the load banner is emitted within the first ~minutes


def _iso_to_unix(iso: str) -> int:
    """Best-effort parse of a Docker StartedAt ISO timestamp → unix seconds."""
    if not iso or iso.startswith("0001-01-01"):
        return 0
    try:
        from datetime import datetime
        # Docker emits RFC3339 with nanoseconds + 'Z'; trim to microseconds.
        s = iso.replace("Z", "+00:00")
        if "." in s:
            head, rest = s.split(".", 1)
            frac = rest.rstrip("+0123456789:-")
            tz = rest[len(frac):]
            s = f"{head}.{frac[:6]}{tz}" if frac else head + tz
        return int(datetime.fromisoformat(s).timestamp())
    except (ValueError, TypeError):
        return 0


async def _fetch_log_lines(c, started_iso: str) -> list[str]:
    """Fetch a sidecar's startup-window logs as lines (bounded, never raises).

    Scopes to ``since=StartedAt`` (and a window ``until``) so a long-running
    container doesn't return its entire history — the llama-server memory banner
    is printed during model load, near the start. Falls back to a plain tail if
    the aiodocker build rejects the since/until params.
    """
    start_unix = _iso_to_unix(started_iso)
    out = None
    if start_unix:
        out = await _bounded(c.log(
            stdout=True, stderr=True, tail=_LOG_TAIL_LINES,
            since=start_unix, until=start_unix + _LOG_WINDOW_S,
        ))
    if out is None:
        out = await _bounded(c.log(stdout=True, stderr=True, tail=_LOG_TAIL_LINES))
    if out is None:
        return []
    if isinstance(out, str):
        return out.splitlines()
    # aiodocker returns list[str]; flatten any stray multi-line entries.
    lines: list[str] = []
    for chunk in out:
        lines.extend(str(chunk).splitlines())
    return lines


async def _sibling_vram(c, cid: str, info: dict) -> int:
    """Measured VRAM (MiB) for an LLM-sibling container, parsed once from its
    llama-server log banner and cached by (cid, StartedAt). 0 when unavailable.
    """
    state = info.get("State") or {}
    started = str(state.get("StartedAt") or "")
    cached = _vram_by_cid.get(cid)
    if cached and cached.get("started") == started and cached.get("vram_mb"):
        return int(cached["vram_mb"])

    from augmentum.models.llama_server_manager import parse_llama_memory_from_lines
    lines = await _fetch_log_lines(c, started)
    if not lines:
        return int((cached or {}).get("vram_mb") or 0)
    vram_mib, ram_mib = parse_llama_memory_from_lines(lines)
    if vram_mib > 0 or ram_mib > 0:
        _vram_by_cid[cid] = {"vram_mb": int(vram_mib), "ram_mb": int(ram_mib), "started": started}
        return int(vram_mib)
    return int((cached or {}).get("vram_mb") or 0)


def _classify(name: str) -> tuple[str, str] | None:
    """Return (subsystem, owner_label) for a sidecar name, or None to skip.

    A positive sidecar-pattern match wins, except hard infra
    (``_SKIP_SUBSTRINGS``) overrides it — so the classifier / vision siblings
    (which run as llama-servers) are always surfaced, while a coincidental
    substring on real infra is not.
    """
    low = name.lower()
    for sub, subsystem, label in _SIDECAR_PATTERNS:
        if sub in low:
            if any(s in low for s in _SKIP_SUBSTRINGS):
                return None
            return subsystem, label
    return None


def _short(name: str) -> str:
    """Trim a long compose/ephemeral container name for display."""
    n = name.lstrip("/")
    # Drop the common ``augmentum-`` / project prefix and trailing replica index.
    for pre in ("augmentum-augmentum-", "augmentum-"):
        if n.startswith(pre):
            n = n[len(pre):]
            break
    return (n[:28] + "…") if len(n) > 29 else n


def _classify_container(name: str, labels: dict) -> tuple[str, str, str, bool] | None:
    """Classify a container into ``(kind, subsystem, label, controllable)``.

    ``kind="sidecar"`` — a managed long-lived modality sidecar (TTS/STT/LLM/OCR):
    controllable (pause/reload frees VRAM). ``kind="ephemeral"`` — something we
    launched on demand (coder workspace, game-stream session): surfaced as a
    measured info row but NOT controllable from the resource panel. ``None`` —
    not tracked.

    Ephemeral labels are checked FIRST so a game-stream container surfaces via
    its ``augmentum.game_stream`` label even though its name matches the
    ``game-stream``/``ws-`` skip entries (those exist to keep them out of the
    *sidecar* list, not to hide them entirely).
    """
    def _truthy(v) -> bool:
        return str(v or "").strip().lower() in ("true", "1", "yes")

    if _truthy(labels.get("augmentum.workspace")):
        return "ephemeral", "coder", f"Coder · {_short(name)}", False
    if _truthy(labels.get("augmentum.game_stream")):
        return "ephemeral", "game", f"Game · {_short(name)}", False

    sidecar = _classify(name)
    if sidecar is not None:
        return "sidecar", sidecar[0], sidecar[1], True
    return None


def _parse_stats(raw) -> tuple[int | None, float | None]:
    """Return (ram_working_set_mb, cpu_pct) from a Docker stats payload.

    Memory = ``usage − stats.inactive_file`` (working-set; key off
    ``inactive_file``, never ``cache`` — moby #43810/#45739). CPU% from the
    cpu/precpu delta, only when both deltas are positive (a fresh single read
    can have zeroed precpu, in which case CPU is reported as unknown rather
    than a bogus 0). ``stream=False`` may return a 1-element list or a dict
    depending on aiodocker version — guard for both.
    """
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    if not isinstance(raw, dict):
        return None, None

    ram_mb: int | None = None
    mem = raw.get("memory_stats") or {}
    usage = mem.get("usage")
    if isinstance(usage, int | float):
        inactive = ((mem.get("stats") or {}).get("inactive_file")) or 0
        try:
            ram_mb = max(0, int(usage) - int(inactive)) // (1024 * 1024)
        except (TypeError, ValueError):
            ram_mb = None

    cpu_pct: float | None = None
    try:
        cpu = raw.get("cpu_stats") or {}
        pre = raw.get("precpu_stats") or {}
        cur_total = (cpu.get("cpu_usage") or {}).get("total_usage")
        pre_total = (pre.get("cpu_usage") or {}).get("total_usage")
        cur_sys = cpu.get("system_cpu_usage")
        pre_sys = pre.get("system_cpu_usage")
        online = cpu.get("online_cpus") or len(
            (cpu.get("cpu_usage") or {}).get("percpu_usage") or []
        )
        if None not in (cur_total, pre_total, cur_sys, pre_sys) and online:
            cpu_delta = cur_total - pre_total
            sys_delta = cur_sys - pre_sys
            if cpu_delta > 0 and sys_delta > 0:
                cpu_pct = round((cpu_delta / sys_delta) * online * 100.0, 1)
    except (TypeError, ValueError, KeyError):
        cpu_pct = None

    return ram_mb, cpu_pct


def _get_docker(_app_state):
    """Return (docker_client, owned). owned=True means caller must close it.

    Always a fresh client. Reusing the container_manager's shared aiodocker
    session across the probe's per-container ``.show()`` calls proved
    unreliable (they silently failed → 0 entries) — likely a concurrent-use /
    session-state issue. A fresh client to the local docker socket is cheap
    and the probe is cached (``_CACHE_TTL_S``) so it isn't created often.
    """
    try:
        import aiodocker
        return aiodocker.Docker(), True
    except Exception:  # noqa: BLE001
        return None, False


async def _bounded(coro, *, default=None):
    """Await ``coro`` under the per-call Docker deadline; never raise."""
    try:
        return await asyncio.wait_for(coro, timeout=_DOCKER_CALL_TIMEOUT_S)
    except Exception:  # noqa: BLE001  — incl. TimeoutError; never propagate
        return default


async def probe_sidecar_containers(
    app_state, *, cache_only: bool = False,
) -> list[dict]:
    """List managed sidecar containers (running + stopped) as panel entries.

    Cached for ``_CACHE_TTL_S`` so the /status poll doesn't inspect Docker on
    every hit. Returns dicts in the same shape as the panel's ``models``
    entries, plus a ``container`` handle, ``controllable`` flag, measured
    ``ram_mb``/``cpu_pct`` (running containers), and ``confidence``/``as_of``.

    ``cache_only=True`` is the **read-path** contract (``GET /status``): never
    run the ~2s live Docker probe inline — serve the last-known list (even if
    stale) and let the background sampler own the refresh. Only a true cold
    start (nothing cached yet) falls through to a single live probe so the
    first poll isn't blank. ``cache_only=False`` (the sampler, and the manual
    ``?fresh=1`` refresh) does the live work.
    """
    global _probe_lock
    now = time.monotonic()
    if now - _cache["at"] < _CACHE_TTL_S:
        return _cache["data"]
    if cache_only and _cache["data"]:
        # Stale but present — the sampler refreshes it; the read path never
        # pays the probe cost (spec §4.5: read path must not await a live probe).
        return _cache["data"]

    # Coalesce concurrent stale-cache callers (sampler + cold /status) behind a
    # single probe; subsequent callers receive the freshly-published cache.
    if _probe_lock is None:
        _probe_lock = asyncio.Lock()
    async with _probe_lock:
        now = time.monotonic()
        if now - _cache["at"] < _CACHE_TTL_S:
            return _cache["data"]
        if cache_only and _cache["data"]:
            return _cache["data"]
        return await _probe_sidecar_containers_uncached(app_state)


async def _probe_sidecar_containers_uncached(app_state) -> list[dict]:
    """Do the actual Docker inspection + stats work, publishing to ``_cache``.

    Always takes the slow path (containers.list + per-container .show()/.stats()).
    Callers go through :func:`probe_sidecar_containers` which fronts this with the
    TTL cache, the cache_only read contract, and the single-flight lock.
    """
    now = time.monotonic()
    client, owned = _get_docker(app_state)
    if client is None:
        return _cache["data"]

    as_of = time.time()
    sem = asyncio.Semaphore(_CONCURRENCY)
    entries: list[dict] = []
    seen_cids: set[str] = set()
    try:
        containers = await _bounded(client.containers.list(all=True), default=[]) or []

        async def _inspect(c) -> tuple[str, dict | None]:
            cid = getattr(c, "id", None) or getattr(c, "_id", "") or ""
            prev = _last_by_cid.get(cid)

            async with sem:
                info = await _bounded(c.show())
            if not info:
                # ``.show()`` raced/timed out this round — keep the last-known
                # entry visible rather than dropping the row (spec §4.6:
                # staleness ≠ wrongness; only truly-gone containers disappear).
                return cid, prev
            name = (info.get("Name") or "").lstrip("/")
            if not name:
                return cid, prev
            labels = (info.get("Config") or {}).get("Labels") or {}
            classified = _classify_container(name, labels)
            if classified is None:
                # Not a tracked container — drop (infra/engine names match no
                # sidecar pattern + carry no ephemeral label).
                return cid, None
            kind, subsystem, label, controllable = classified
            state = info.get("State") or {}
            running = bool(state.get("Running"))
            # Stopped ephemeral containers are dead sessions/workspaces — hide
            # them (sidecars still show "paused" so they can be reloaded).
            if kind == "ephemeral" and not running:
                return cid, None
            host_cfg = info.get("HostConfig") or {}
            has_gpu = bool(host_cfg.get("DeviceRequests"))

            ram_mb: int | None = None
            cpu_pct: float | None = None
            measured_at = as_of
            if running:
                async with sem:
                    raw = await _bounded(c.stats(stream=False))
                if raw is not None:
                    ram_mb, cpu_pct = _parse_stats(raw)

            # VRAM for LLM siblings (classifier / vision) — measured once from
            # their llama-server log banner, then cached (spec §4.6 rung B). Other
            # sidecars' per-process VRAM is opaque on WSL2; left at 0 for now.
            vram_mb = 0
            if running and kind == "sidecar" and subsystem == "llm":
                async with sem:
                    vram_mb = await _sibling_vram(c, cid, info)

            # Stats raced/timed out (or the container just (un)paused): carry the
            # previous measured RAM/CPU forward so the value doesn't blink to 0.
            confidence = "measured"
            if ram_mb is None and prev is not None and running and prev.get("status") == "ready":
                ram_mb = prev.get("ram_mb") or None
                cpu_pct = prev.get("cpu_pct")
                measured_at = prev.get("as_of", as_of)
                confidence = "stale" if ram_mb is not None else "unknown"
            elif ram_mb is None:
                confidence = "unknown"

            return cid, {
                "name": label,
                "subsystem": subsystem,
                "backend": "container",
                "device": "gpu" if has_gpu else "cpu",
                "vram_mb": int(vram_mb or 0),
                "ram_mb": int(ram_mb or 0),
                "cpu_pct": cpu_pct,
                "quantization": "",
                "parameter_size": "",
                "family": "",
                "pipeline_type": "",
                "expires_at": "",
                "status": "ready" if running else "paused",
                "container": name,
                "controllable": controllable,
                "kind": kind,
                "confidence": confidence,
                "as_of": measured_at,
            }

        results = await asyncio.gather(
            *(_inspect(c) for c in containers), return_exceptions=True
        )
        for r in results:
            if not isinstance(r, tuple):
                continue  # an _inspect raised despite _bounded — skip defensively
            cid, entry = r
            if isinstance(entry, dict):
                entries.append(entry)
                if cid:
                    seen_cids.add(cid)
                    _last_by_cid[cid] = entry
        # Prune last-known + parsed-VRAM for containers that genuinely vanished
        # from the list, so a removed sidecar doesn't linger forever.
        for stale_cid in [k for k in _last_by_cid if k not in seen_cids]:
            _last_by_cid.pop(stale_cid, None)
        for stale_cid in [k for k in _vram_by_cid if k not in seen_cids]:
            _vram_by_cid.pop(stale_cid, None)
    except Exception:  # noqa: BLE001
        log.warning("sidecar_container_probe_failed", exc_info=True)
        # On a whole-probe failure, serve the last good cached list rather than
        # blanking the panel.
        if not entries and _cache["data"]:
            return _cache["data"]
    finally:
        if owned:
            # cleanup of a throwaway client — not a save/load path
            with contextlib.suppress(Exception):
                await client.close()

    entries.sort(key=lambda e: (e["subsystem"], e["name"]))
    _cache["at"] = now
    _cache["data"] = entries
    return entries


async def find_sidecar_container(app_state, name_substr: str) -> str | None:
    """Return the NAME of a controllable sidecar container whose name contains
    ``name_substr`` (running or stopped), or None.

    Used by the classifier-slot takeover/unload path to locate the external
    classifier container so it can be stopped (freeing its VRAM once Slot C
    serves the role) and resumed again on unload. Best-effort — returns None
    when Docker is unavailable or nothing matches; never raises.

    **Reads the sampler-maintained probe cache first.** The original version did
    a fresh sequential ``containers.list(all=True)`` + per-container ``.show()``
    sweep, which reliably TIMED OUT when called from the model-load path (the
    docker-proxy is contended while ``refresh_model_map`` probes every backend),
    so takeover-stop silently no-op'd (2026-07-16 incident). The background
    resource sampler already keeps ``probe_sidecar_containers`` warm with each
    sidecar's ``container`` name, so a cache read is both reliable and cheap.
    Only a genuinely cold cache falls through to a bounded live sweep.
    """
    name_substr = (name_substr or "").lower()
    if not name_substr:
        return None

    # Preferred path: the warm probe cache (name never blinks; no Docker call).
    try:
        cached = await probe_sidecar_containers(app_state, cache_only=True)
        for entry in cached or []:
            cname = str(entry.get("container") or "")
            if cname and name_substr in cname.lower():
                return cname
    except Exception:  # noqa: BLE001
        log.debug("find_sidecar_container_cache_read_failed", substr=name_substr)

    # Cold-cache fallback: bounded live sweep (rare — only before the sampler
    # has run once). Still best-effort; never raises.
    client, owned = _get_docker(app_state)
    if client is None:
        log.info("find_sidecar_container_not_found", substr=name_substr, source="no_docker")
        return None
    try:
        containers = await _bounded(client.containers.list(all=True), default=[]) or []
        for c in containers:
            info = await _bounded(c.show())
            if not info:
                continue
            name = (info.get("Name") or "").lstrip("/")
            if name and name_substr in name.lower() and _classify(name) is not None:
                return name
        log.info("find_sidecar_container_not_found", substr=name_substr, source="live_sweep")
    except Exception:  # noqa: BLE001
        log.warning("find_sidecar_container_failed", substr=name_substr, exc_info=True)
    finally:
        if owned:
            with contextlib.suppress(Exception):
                await client.close()
    return None
    return None


async def set_container_paused(app_state, container_name: str, paused: bool) -> tuple[bool, str]:
    """Stop (paused=True) or start (paused=False) a managed sidecar container.

    Only containers that classify as known sidecars may be controlled — guards
    against using this to stop the app itself or infra.
    """
    if not container_name or _classify(container_name) is None:
        return False, "Not a controllable sidecar container"
    client, owned = _get_docker(app_state)
    if client is None:
        return False, "Docker client unavailable"
    try:
        c = await client.containers.get(container_name)
        if paused:
            await c.stop()
        else:
            await c.start()
        # Bust the cache so the next /status reflects the new state.
        _cache["at"] = 0.0
        return True, ""
    except Exception as exc:  # noqa: BLE001
        log.warning("sidecar_container_control_failed",
                    container=container_name, paused=paused, exc_info=True)
        return False, str(exc)[:200]
    finally:
        if owned:
            # cleanup of a throwaway client — not a save/load path
            with contextlib.suppress(Exception):
                await client.close()
