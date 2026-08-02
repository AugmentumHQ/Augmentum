# Resource Ledger — Design

## What This Is

A lightweight aggregation layer that answers one question:
**"What's using my GPU and RAM right now, across all subsystems?"**

No eviction engine. No admission controller. No delta measurement.
Just a clean, unified view that collects what every backend already
reports and stores it in one place. The data we collect now is what
future systems (auto-eviction, admission control, load balancing)
will build on top of.

---

## What Each Backend Already Tells Us

| Backend | What It Reports | How |
|---------|----------------|-----|
| **Ollama** | Model name, `size_vram` (bytes), `size` (total bytes), `expires_at`, quantization, family, parameter_size | `GET /api/ps` → `models[]` |
| **llama.cpp** | Model name, slot status (loaded/idle) | `GET /slots` or router `list_models()` |
| **llama.cpp** | *(missing)* VRAM/RAM usage | Not reported — we know it's loaded, not how much it costs |
| **Diffusers** | Current model path, `is_loaded`, pipeline type | `PipelineRegistry` properties |
| **Diffusers** | VRAM allocated by PyTorch | `torch.cuda.memory_allocated()` |
| **System** | Free VRAM, free RAM, GPU name, total VRAM | `torch.cuda.mem_get_info()`, `psutil`, `nvidia-smi` |
| **Voice (local)** | Container health | Docker healthcheck or HTTP probe |
| **Voice (cloud)** | Nothing (remote) | N/A |
| **External (LM Studio etc)** | Model list | `GET /v1/models` |
| **External** | *(missing)* VRAM/RAM usage | Not reported |

**Key insight**: Ollama is the richest reporter. Diffusers gives us
PyTorch-level VRAM. Everything else is partial. The ledger aggregates
what exists and marks gaps honestly.

---

## Architecture

```
┌─────────────────────────────────────────┐
│            ResourceLedger                │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  collect() → ResourceSnapshot    │   │
│  │                                  │   │
│  │  Polls:                          │   │
│  │  • GPU state (torch/nvidia-smi)  │   │
│  │  • RAM state (psutil)            │   │
│  │  • Ollama /api/ps                │   │
│  │  • llama.cpp /slots              │   │
│  │  • PipelineRegistry state        │   │
│  │  • Runtime providers /v1/models  │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  SQLite: resource_snapshots      │   │
│  │  SQLite: resource_models         │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  GET /api/resources/status       │   │
│  │  GET /api/resources/history      │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

Three things: **collect**, **store**, **serve**.

---

## Data Model

### ResourceSnapshot — Point-in-time state of everything

```python
@dataclass
class ResourceSnapshot:
    """Complete resource state at a moment in time."""
    timestamp: datetime

    # Hardware
    gpu_name: str                    # e.g. "NVIDIA GPU-B"
    gpu_total_mb: int                # 24564
    gpu_used_mb: int                 # 14200 (from nvidia-smi or torch)
    gpu_free_mb: int                 # 10364
    ram_total_mb: int                # 65536
    ram_used_mb: int                 # 34000
    ram_free_mb: int                 # 31536

    # Loaded models across all subsystems
    models: list[TrackedModel]
```

### TrackedModel — A single loaded model

```python
@dataclass
class TrackedModel:
    """A model currently loaded somewhere."""
    name: str                        # "qwen2.5:19b-q4_K_M"
    subsystem: str                   # "llm" | "image" | "tts" | "stt"
    backend: str                     # "ollama" | "llamacpp" | "diffusers" | provider key
    device: str                      # "gpu" | "cpu" | "gpu+cpu" | "remote" | "unknown"

    # Memory — from backend reports (0 = unknown/not reported)
    vram_mb: int = 0                 # Ollama: size_vram // MB. Diffusers: torch alloc
    ram_mb: int = 0                  # Ollama: (size - size_vram) // MB

    # Metadata — whatever the backend tells us
    quantization: str = ""           # "Q4_K_M", "fp16", etc.
    parameter_size: str = ""         # "19B", "7B", etc.
    family: str = ""                 # "qwen2", "llama", etc.
    pipeline_type: str = ""          # "sd15", "sdxl", "flux" (image only)

    # Lifecycle
    expires_at: str = ""             # Ollama keep_alive expiry
    active: bool = True              # Currently serving requests?
```

### Stored model profile — What we learn over time

```python
@dataclass
class ModelProfile:
    """Accumulated knowledge about a model's resource needs.

    Built from observed TrackedModel data over time. NOT from delta
    measurement — from what backends actually report.
    """
    model_name: str                  # Normalized name
    subsystem: str
    backend: str

    # Best known resource usage (from most recent observation)
    vram_mb: int = 0                 # 0 = never observed with VRAM data
    ram_mb: int = 0
    device: str = ""                 # Most common device seen on

    # Metadata
    quantization: str = ""
    parameter_size: str = ""
    family: str = ""
    pipeline_type: str = ""

    # Tracking
    times_seen: int = 0              # How many snapshots included this model
    first_seen: datetime | None = None
    last_seen: datetime | None = None
```

**Why store profiles?** Because Ollama reports `size_vram` every time
we poll `/api/ps`. After seeing a model once, we know its footprint
forever — even when it's not currently loaded. This is the data that
future admission control will use. No delta measurement needed.

For backends that DON'T report VRAM (llama.cpp, LM Studio), the
profile stays at `vram_mb=0`. That's honest. Future systems can
add heuristic estimation as a separate concern — the ledger just
records what it observes.

---

## Implementation

### `augmentum/resource/__init__.py`

Empty init.

### `augmentum/resource/ledger.py` — The core

```python
class ResourceLedger:
    """Aggregates resource state across all Augmentum subsystems."""

    def __init__(self, db: aiosqlite.Connection | None = None):
        self._db = db
        self._model_manager: ModelManager | None = None
        self._provider_registry: ProviderRegistry | None = None
        self._pipeline_registry = None  # image PipelineRegistry
        self._hardware_profile = None   # image HardwareProfile
        self._last_snapshot: ResourceSnapshot | None = None

    # --- Wiring (called during server startup) ---

    def set_model_manager(self, mm: ModelManager) -> None: ...
    def set_provider_registry(self, pr: ProviderRegistry) -> None: ...
    def set_image_subsystem(self, pipeline_reg, hw_profile) -> None: ...

    # --- The main method ---

    async def collect(self) -> ResourceSnapshot:
        """Poll all backends and build a unified snapshot.

        This is the only method that talks to backends. Everything
        else reads from the snapshot or the DB.
        """
        models: list[TrackedModel] = []

        # 1. GPU + RAM state
        gpu_name, gpu_total, gpu_used, gpu_free = _probe_gpu()
        ram_total, ram_used, ram_free = _probe_ram()

        # 2. LLM models (Ollama, llama.cpp, runtime providers)
        if self._model_manager:
            for rm in await self._model_manager.get_running_models():
                models.append(TrackedModel(
                    name=rm.name,
                    subsystem="llm",
                    backend=rm.backend,
                    device=_infer_device(rm),
                    vram_mb=rm.size_vram // (1024 * 1024) if rm.size_vram else 0,
                    ram_mb=rm.size_ram // (1024 * 1024) if rm.size_ram else 0,
                    quantization=rm.details.get("quantization_level", ""),
                    parameter_size=rm.details.get("parameter_size", ""),
                    family=rm.details.get("family", ""),
                    expires_at=rm.expires_at,
                ))

        # 3. Image model (diffusers)
        if self._pipeline_registry and self._pipeline_registry.is_loaded:
            img_vram = _get_torch_allocated_mb()
            pipeline = self._pipeline_registry.current
            models.append(TrackedModel(
                name=self._pipeline_registry.current_model,
                subsystem="image",
                backend="diffusers",
                device="gpu" if img_vram > 100 else "cpu",
                vram_mb=img_vram,
                pipeline_type=getattr(pipeline, "pipeline_type", ""),
            ))

        # 4. Runtime providers (external OpenAI-compat like LM Studio)
        #    We know what models they serve, but not memory usage
        if self._provider_registry:
            for key, backend in self._provider_registry.backends.items():
                if key in ("ollama", "llamacpp"):
                    continue  # Already covered above
                try:
                    provider_models = await backend.list_models()
                    for m in provider_models:
                        models.append(TrackedModel(
                            name=m.name,
                            subsystem="llm",
                            backend=key,
                            device="unknown",  # Can't know for external
                            # vram_mb=0 — honest: we don't know
                        ))
                except Exception:
                    pass

        snapshot = ResourceSnapshot(
            timestamp=datetime.utcnow(),
            gpu_name=gpu_name,
            gpu_total_mb=gpu_total,
            gpu_used_mb=gpu_used,
            gpu_free_mb=gpu_free,
            ram_total_mb=ram_total,
            ram_used_mb=ram_used,
            ram_free_mb=ram_free,
            models=models,
        )

        self._last_snapshot = snapshot

        # Persist: update model profiles with new observations
        if self._db:
            await self._update_profiles(models)
            await self._store_snapshot(snapshot)

        return snapshot

    # --- Queries ---

    @property
    def last_snapshot(self) -> ResourceSnapshot | None:
        """Most recent snapshot without re-polling."""
        return self._last_snapshot

    async def get_model_profile(self, model_name: str) -> ModelProfile | None:
        """Look up stored profile for a model (from past observations)."""
        ...

    async def list_profiles(self) -> list[ModelProfile]:
        """All known model profiles."""
        ...

    async def get_history(self, hours: int = 24, limit: int = 100
                          ) -> list[ResourceSnapshot]:
        """Recent snapshots for charting/debugging."""
        ...

    # --- Profile learning ---

    async def _update_profiles(self, models: list[TrackedModel]) -> None:
        """Update stored profiles from current observations.

        For each model in the snapshot:
        - If profile exists: update vram/ram if backend reported it,
          increment times_seen, update last_seen
        - If no profile: create one
        """
        ...

    async def _store_snapshot(self, snap: ResourceSnapshot) -> None:
        """Store snapshot summary for history (compact, not every model)."""
        ...
```

### Helper functions

```python
def _probe_gpu() -> tuple[str, int, int, int]:
    """Return (gpu_name, total_mb, used_mb, free_mb).

    Tries torch.cuda first (most accurate when torch is loaded),
    falls back to nvidia-smi subprocess (works without torch).
    Returns ("", 0, 0, 0) if no GPU.
    """
    # Try 1: torch.cuda (if available — image subsystem loaded it)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            name = props.name
            total = getattr(props, "total_memory", 0) // (1024 * 1024)
            free_bytes, _ = torch.cuda.mem_get_info(0)
            free = free_bytes // (1024 * 1024)
            return name, total, total - free, free
    except Exception:
        pass

    # Try 2: nvidia-smi (works without torch)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) == 4:
                return parts[0], int(parts[1]), int(parts[2]), int(parts[3])
    except Exception:
        pass

    return "", 0, 0, 0


def _probe_ram() -> tuple[int, int, int]:
    """Return (total_mb, used_mb, free_mb)."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return (
            mem.total // (1024 * 1024),
            mem.used // (1024 * 1024),
            mem.available // (1024 * 1024),
        )
    except Exception:
        return 0, 0, 0


def _get_torch_allocated_mb() -> int:
    """Return MB of VRAM currently allocated by PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) // (1024 * 1024)
    except Exception:
        pass
    return 0


def _infer_device(rm: RunningModel) -> str:
    """Infer device from RunningModel data."""
    if rm.size_vram and rm.size_ram:
        if rm.size_vram > rm.size_ram:
            return "gpu"
        if rm.size_ram > rm.size_vram:
            return "gpu+cpu"  # Partial offload
    if rm.size_vram and not rm.size_ram:
        return "gpu"
    if rm.size_ram and not rm.size_vram:
        return "cpu"
    return "unknown"
```

---

## SQLite Storage

### Migration: `025_resource_ledger.sql`

```sql
-- Model profiles: what we've learned about each model over time
CREATE TABLE IF NOT EXISTS resource_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    subsystem TEXT NOT NULL,          -- 'llm', 'image', 'tts', 'stt'
    backend TEXT NOT NULL,
    vram_mb INTEGER NOT NULL DEFAULT 0,
    ram_mb INTEGER NOT NULL DEFAULT 0,
    device TEXT NOT NULL DEFAULT '',
    quantization TEXT NOT NULL DEFAULT '',
    parameter_size TEXT NOT NULL DEFAULT '',
    family TEXT NOT NULL DEFAULT '',
    pipeline_type TEXT NOT NULL DEFAULT '',
    times_seen INTEGER NOT NULL DEFAULT 1,
    first_seen TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(model_name, backend)
);

-- Snapshot history: periodic GPU/RAM state for trending
CREATE TABLE IF NOT EXISTS resource_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    gpu_total_mb INTEGER NOT NULL DEFAULT 0,
    gpu_used_mb INTEGER NOT NULL DEFAULT 0,
    gpu_free_mb INTEGER NOT NULL DEFAULT 0,
    ram_total_mb INTEGER NOT NULL DEFAULT 0,
    ram_used_mb INTEGER NOT NULL DEFAULT 0,
    ram_free_mb INTEGER NOT NULL DEFAULT 0,
    loaded_model_count INTEGER NOT NULL DEFAULT 0,
    loaded_models_json TEXT NOT NULL DEFAULT '[]'  -- Compact JSON of model names + vram
);

-- Keep snapshot history bounded (auto-prune old entries)
CREATE INDEX IF NOT EXISTS idx_resource_snapshots_ts
    ON resource_snapshots(timestamp);
```

**Profile update logic** (in `_update_profiles`):

```sql
-- Upsert: update if backend reported VRAM (don't overwrite good data with 0)
INSERT INTO resource_profiles (model_name, subsystem, backend, vram_mb, ram_mb,
                               device, quantization, parameter_size, family,
                               pipeline_type, times_seen, first_seen, last_seen)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, datetime('now'), datetime('now'))
ON CONFLICT(model_name, backend) DO UPDATE SET
    -- Only update vram/ram if the new value is non-zero (don't lose data)
    vram_mb = CASE WHEN excluded.vram_mb > 0 THEN excluded.vram_mb
                   ELSE resource_profiles.vram_mb END,
    ram_mb = CASE WHEN excluded.ram_mb > 0 THEN excluded.ram_mb
                  ELSE resource_profiles.ram_mb END,
    device = CASE WHEN excluded.device != '' AND excluded.device != 'unknown'
                  THEN excluded.device ELSE resource_profiles.device END,
    quantization = CASE WHEN excluded.quantization != ''
                        THEN excluded.quantization ELSE resource_profiles.quantization END,
    parameter_size = CASE WHEN excluded.parameter_size != ''
                          THEN excluded.parameter_size ELSE resource_profiles.parameter_size END,
    family = CASE WHEN excluded.family != ''
                  THEN excluded.family ELSE resource_profiles.family END,
    pipeline_type = CASE WHEN excluded.pipeline_type != ''
                         THEN excluded.pipeline_type ELSE resource_profiles.pipeline_type END,
    times_seen = resource_profiles.times_seen + 1,
    last_seen = datetime('now');
```

**Snapshot pruning** (keep last 24h at full resolution, older at 1/hour):

```python
async def _prune_snapshots(self) -> None:
    """Keep snapshot history bounded."""
    if not self._db:
        return
    # Delete snapshots older than 7 days
    await self._db.execute(
        "DELETE FROM resource_snapshots WHERE timestamp < datetime('now', '-7 days')"
    )
    await self._db.commit()
```

---

## REST API

### `augmentum/proxy/resource_routes.py`

```python
from fastapi import APIRouter, Request
from augmentum.resource.ledger import ResourceLedger

router = APIRouter(prefix="/api/resources", tags=["resources"])


@router.get("/status")
async def resource_status(request: Request):
    """Current resource state — polls all backends live."""
    ledger: ResourceLedger = request.app.state.resource_ledger
    snap = await ledger.collect()
    return {
        "gpu": {
            "name": snap.gpu_name,
            "total_mb": snap.gpu_total_mb,
            "used_mb": snap.gpu_used_mb,
            "free_mb": snap.gpu_free_mb,
        },
        "ram": {
            "total_mb": snap.ram_total_mb,
            "used_mb": snap.ram_used_mb,
            "free_mb": snap.ram_free_mb,
        },
        "models": [
            {
                "name": m.name,
                "subsystem": m.subsystem,
                "backend": m.backend,
                "device": m.device,
                "vram_mb": m.vram_mb,
                "ram_mb": m.ram_mb,
                "quantization": m.quantization,
                "parameter_size": m.parameter_size,
                "family": m.family,
                "pipeline_type": m.pipeline_type,
                "expires_at": m.expires_at,
            }
            for m in snap.models
        ],
    }


@router.get("/profiles")
async def model_profiles(request: Request):
    """All known model profiles (learned from past observations)."""
    ledger: ResourceLedger = request.app.state.resource_ledger
    profiles = await ledger.list_profiles()
    return {"profiles": [asdict(p) for p in profiles]}


@router.get("/history")
async def resource_history(request: Request, hours: int = 24, limit: int = 100):
    """GPU/RAM usage over time."""
    ledger: ResourceLedger = request.app.state.resource_ledger
    history = await ledger.get_history(hours=hours, limit=limit)
    return {"snapshots": [
        {
            "timestamp": s.timestamp.isoformat(),
            "gpu_used_mb": s.gpu_used_mb,
            "gpu_free_mb": s.gpu_free_mb,
            "ram_used_mb": s.ram_used_mb,
            "model_count": len(s.models),
        }
        for s in history
    ]}
```

Only 3 endpoints. Clean, read-only, no side effects.

---

## Server Integration

### `augmentum/proxy/server.py` — lifespan changes

```python
# During startup, after ModelManager and ProviderRegistry are created:

from augmentum.resource.ledger import ResourceLedger

ledger = ResourceLedger(db=db_conn)
ledger.set_model_manager(model_manager)
ledger.set_provider_registry(provider_registry)
app.state.resource_ledger = ledger

# After image subsystem init:
if hasattr(app.state, "image_pipeline_registry"):
    ledger.set_image_subsystem(
        app.state.image_pipeline_registry,
        app.state.image_hardware,
    )

# Initial collection (learn what's already loaded at startup):
await ledger.collect()

# Mount routes:
from augmentum.proxy.resource_routes import router as resource_router
app.include_router(resource_router)
```

**No background polling task.** The ledger collects on-demand when
the API is called. The only automatic collection is at startup (to
seed profiles for any models already loaded).

Optional: collect after image generation completes and after chat
requests complete, to keep profiles fresh. This is a one-line call:

```python
# In _generate_fn, after generation completes:
if hasattr(app.state, "resource_ledger"):
    await app.state.resource_ledger.collect()
```

---

## What The Frontend Gets

The `/api/resources/status` endpoint gives the UI everything it needs
to build a resource monitor. The response looks like:

```json
{
    "gpu": {
        "name": "NVIDIA GPU-B",
        "total_mb": 24564,
        "used_mb": 14200,
        "free_mb": 10364
    },
    "ram": {
        "total_mb": 65536,
        "used_mb": 34000,
        "free_mb": 31536
    },
    "models": [
        {
            "name": "qwen2.5:19b-q4_K_M",
            "subsystem": "llm",
            "backend": "ollama",
            "device": "gpu",
            "vram_mb": 12288,
            "ram_mb": 1024,
            "quantization": "Q4_K_M",
            "parameter_size": "19B",
            "family": "qwen2",
            "pipeline_type": "",
            "expires_at": "2026-03-11T12:30:00Z"
        },
        {
            "name": "/data/image_models/lumina-next-2b",
            "subsystem": "image",
            "backend": "diffusers",
            "device": "gpu",
            "vram_mb": 8500,
            "ram_mb": 0,
            "quantization": "",
            "parameter_size": "",
            "family": "",
            "pipeline_type": "flux",
            "expires_at": ""
        }
    ]
}
```

The frontend can render GPU/RAM bars, model cards with subsystem
icons, and the VRAM courtesy check before image generation becomes
a simple comparison against this data — no stale snapshots.

---

## Host vs Container View

When Augmentum runs in a container, `psutil` reads the container's view
of RAM/CPU. On Docker Desktop that's the WSL2/Linux VM — its RAM total
and CPU usage don't match the host OS's Task Manager / Activity Monitor.
GPU/VRAM is unaffected (it comes from `nvidia-smi`, which sees the whole
device). The host OS is otherwise opaque to code in the container.

`scripts/host_stats_agent.py` is a tiny stdlib HTTP server the operator
runs on the host (`pip install psutil; python scripts/host_stats_agent.py`).
It serves `GET /stats` with `{ram:{total_mb,used_mb,free_mb}, cpu_pct,
cpu_count, os, hostname}`. `augmentum/resource/host_probe.py` fetches it
(via `app.state.http_client`) on each `/api/resources/status` poll, with
a 5s hit-cache and a 60s backoff while the agent is down. Discovery is
zero-config on Docker Desktop — the default URL is
`http://host.docker.internal:6109/stats`. Override with
`AUGMENTUM_HOST_STATS_URL`; set `AUGMENTUM_HOST_STATS_TOKEN` to match the
agent's `--token` when binding `0.0.0.0` on plain Linux Docker.

`/api/resources/status` gains a `host` object: `{"available": false}` when
no agent answers, else `{"available": true, "source": "agent", "hostname",
"os", "cpu_pct", "cpu_count", "ram": {...}}`. The header widget renders
host-then-container rows for RAM and CPU when `host.available`, and
otherwise shows the container view plus a note pointing at the agent.

## What Future Systems Get

The `resource_profiles` table becomes the foundation for everything:

- **Admission control**: "Will lumina fit?" → check profile for
  lumina's vram_mb, compare against current gpu_free_mb. Profile
  exists because Ollama or torch reported it last time it was loaded.

- **Eviction decisions**: "What should I unload?" → query profiles
  for loaded models, sort by subsystem priority + recency.

- **Load time estimates**: Track load duration in profiles (add a
  column later) to estimate eviction cost.

- **Smart suggestions**: "You have 10GB free. These models from your
  profile history would fit: ..."

- **External provider estimation**: When LM Studio loads a model and
  we see GPU usage jump by 12GB in the next snapshot, we can correlate
  that with the model that appeared in `/v1/models`. That's implicit
  delta tracking without the fragile measurement protocol.

All of this is additive. The ledger doesn't need to change.

---

## Files to Create

```
augmentum/resource/__init__.py          — empty
augmentum/resource/ledger.py            — ResourceLedger, helpers, dataclasses
augmentum/proxy/resource_routes.py      — 3 REST endpoints
augmentum/state/migrations/025_resource_ledger.sql
tests/test_resource_ledger.py
```

## Files to Modify

```
augmentum/proxy/server.py               — init ledger, mount routes, collect on startup
augmentum/config.py                     — resource_ledger_enabled: bool = True
```

## What We're NOT Building (Yet)

- Delta measurement protocol (fragile, not needed — backends report)
- Eviction engine (future, builds on profiles)
- Admission controller (future, builds on profiles + live status)
- Background polling (on-demand is sufficient)
- Voice model tracking (containers don't report model state yet)
- Fingerprint import/export (nice-to-have, not essential)
- UI widget (separate task, consumes the API)

---

## Test Plan

```python
# test_resource_ledger.py

# 1. GPU/RAM probing
#    - Mock torch.cuda → verify gpu_name, totals
#    - Mock psutil → verify ram totals
#    - No GPU available → returns zeros gracefully

# 2. Snapshot collection
#    - Mock ModelManager.get_running_models() → verify TrackedModel mapping
#    - Ollama model with size_vram → correct vram_mb conversion (bytes to MB)
#    - llama.cpp model without VRAM → vram_mb=0, device="unknown"
#    - Image pipeline loaded → picks up torch.cuda.memory_allocated
#    - Image pipeline not loaded → no image model in snapshot
#    - External provider → device="unknown", vram_mb=0

# 3. Device inference
#    - size_vram > size_ram → "gpu"
#    - size_ram > size_vram → "gpu+cpu"
#    - only size_vram → "gpu"
#    - only size_ram → "cpu"
#    - neither → "unknown"

# 4. Profile persistence
#    - First observation creates profile with times_seen=1
#    - Second observation increments times_seen, updates last_seen
#    - Non-zero vram_mb overwrites zero (don't lose data)
#    - Zero vram_mb does NOT overwrite non-zero (preserve good data)
#    - Metadata fields update only when non-empty

# 5. Snapshot history
#    - Snapshots stored in DB
#    - Pruning removes entries older than 7 days
#    - History query respects hours + limit params

# 6. REST endpoints
#    - GET /api/resources/status returns correct shape
#    - GET /api/resources/profiles returns stored profiles
#    - GET /api/resources/history returns time series
```
