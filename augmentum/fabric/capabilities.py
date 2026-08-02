"""Capability schemas: what a peer can do, expressed as structured data.

Each capability has a typed ``kind`` discriminator + kind-specific
fields. They serialise to JSON for the wire format (carried in
heartbeat payloads) and deserialise back into typed dataclasses on
the receiver side. The schema is versioned per-kind via
``schema_version`` so individual capability types can evolve
independently without bumping the wire protocol.

Phase 2 ships three kinds (the three with cleanest existing sources):
``llm.inference``, ``image.generation``, ``knowledge.search``. TTS/STT
defer to a follow-up that handles the audio_providers store pattern
properly. Routing director (Phase 3) consumes these via the coordinator's
per-peer capability registry.

Design notes:

- All capabilities have ``kind`` + ``schema_version`` so unknown future
  kinds round-trip cleanly through older receivers (unknown kind is
  ignored, never an error).
- ``serialise`` / ``deserialise`` are class methods so callers can do
  ``Capability.deserialise(d)`` without needing to import the right
  subclass.
- Pure Python; no I/O, no global state, no side effects on import.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Stable kind discriminators. Match Phase 3's routing-rule grammar.
KIND_LLM_INFERENCE = "llm.inference"
KIND_IMAGE_GENERATION = "image.generation"
KIND_KNOWLEDGE_SEARCH = "knowledge.search"

# Future-reserved kinds (not yet implemented but reserved so receivers
# don't reject them as unknown during a staged rollout):
KIND_TTS_SYNTHESIZE = "tts.synthesize"
KIND_STT_TRANSCRIBE = "stt.transcribe"
KIND_CODE_EXECUTION = "code.execution"

# Render capability for cast targets (TVs, secondary displays, etc.).
# Single-machine users still advertise this — the local node IS the
# render node, and the diagnostic surface uses the same data shape
# whether or not fabric peers are paired.
KIND_CAST_RENDER = "cast.render"


@dataclass(frozen=True)
class CapabilityBase:
    """Base for every capability. Subclasses override ``kind`` and add
    their own typed fields. ``schema_version`` lets each kind evolve
    independently.
    """

    kind: str
    schema_version: int = 1


@dataclass(frozen=True)
class LLMInferenceCapability(CapabilityBase):
    """A peer that can serve LLM inference for one specific model.

    One instance per loaded model. The peer reports ``loaded=True`` for
    models currently resident on the GPU/CPU; future Phase 2.x can add
    ``loadable=True`` for models on disk but not yet loaded.

    Phase 10 adds cost fields used by RoutingDirector's scoring
    function. Local hardware-hosted models default to 0.0 (sunk
    hardware cost — operator already paid for electricity). Cloud-
    proxied models populate via cost_table.lookup_cost(model_id).
    """

    kind: str = KIND_LLM_INFERENCE
    schema_version: int = 1
    backend: str = ""                 # backend key in provider_registry
    model_id: str = ""                # e.g. "Qwen3.5-72B-A10B-q4"
    model_family: str = ""            # "qwen3" | "gemma4" | etc.
    params_b: float = 0.0             # billions of parameters
    active_params_b: float | None = None  # for MoE: ~10B for A10B variant
    ctx_max: int = 0                  # max context (KV-allocatable)
    loaded: bool = False              # currently resident
    free_slots: int = 0               # llama-server --parallel - active
    device: dict = field(default_factory=dict)  # {"gpu_name", "vram_free_mb"}
    # Phase 10 — cost-aware routing inputs. 0.0 means "free" (local
    # GPU + paired-peer GPU). For cloud-backed models (OpenAI,
    # Anthropic, Together, etc. exposed via a peer-as-gateway) the
    # extractor looks these up from the vendored LiteLLM table.
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0


@dataclass(frozen=True)
class ImageGenerationCapability(CapabilityBase):
    """A peer that can serve image generation for one specific model."""

    kind: str = KIND_IMAGE_GENERATION
    schema_version: int = 1
    backend: str = ""                 # "diffusers" | "openai" | "stability" | ...
    model_id: str = ""                # specific model identifier
    family: str = ""                  # "sd_1.5" | "sdxl" | "flux" | "sd3" | ...
    loaded: bool = False
    max_resolution: str = ""          # "1024x1024" if known


@dataclass(frozen=True)
class KnowledgeSearchCapability(CapabilityBase):
    """A peer that can serve knowledge-pack search for one specific pack."""

    kind: str = KIND_KNOWLEDGE_SEARCH
    schema_version: int = 1
    pack_id: str = ""                 # "wikipedia_en_simple_2026-02"
    pack_name: str = ""               # friendly name
    chunk_count: int = 0
    embedding_dim: int = 0
    active: bool = True               # search is enabled for this pack
    pack_format: str = ""             # "augpack" | "zim" | "augpack+zim"


@dataclass(frozen=True)
class TTSSynthesizeCapability(CapabilityBase):
    """A peer that can serve TTS synthesis for one engine.

    Carries the engine's voice list inline so the receiver can populate
    its voice→provider map directly from heartbeats — no extra round-trip
    fetch against the peer's /api/audio/voices. Languages are tracked
    separately so language-aware routing (the partner-language work in
    [[project_language_partner]]) can target the right peer without
    parsing voice names.

    ``provider_id`` is the receiver-facing handle: ``"kokoro-builtin"``
    for in-process built-ins, the audio_providers row id for external
    sidecars (Chatterbox, Qwen, Fish, etc.). The fabric layer wraps it
    as ``fabric:<node_id>:<provider_id>`` when injecting into a peer's
    local voice map so resolution can round-trip cleanly.

    ``base_url_path`` is the path on the peer's HTTPS edge that serves
    OpenAI-compat audio (typically ``/v1/audio``). The receiver builds
    the full peer URL by joining the peer addr with this path.
    """

    kind: str = KIND_TTS_SYNTHESIZE
    schema_version: int = 1
    engine: str = ""                  # "kokoro" | "pockettts" | "chatterbox" | "qwen" | ...
    provider_id: str = ""             # local id on the peer (e.g. "kokoro-builtin")
    provider_name: str = ""           # friendly label for UI
    base_url_path: str = "/v1/audio"  # path on peer for the OpenAI-compat audio surface
    default_model: str = ""           # provider's default model id
    default_voice: str = ""           # provider's default voice
    voices: list = field(default_factory=list)     # voice names this engine serves
    languages: list = field(default_factory=list)  # ISO codes (en, es, ja, …); empty means "unknown / multi"
    streaming: bool = True            # supports streamed audio chunks
    in_process: bool = False          # True = bundled engine (Kokoro/Pocket); False = HTTP sidecar


@dataclass(frozen=True)
class CastRenderCapability(CapabilityBase):
    """A node that can render and encode media for cast targets.

    One instance per node. The local node always advertises one,
    even on a single-machine deployment with no fabric peers — the
    same data drives the diagnostic surface ("here's what my box can
    do") that powers future cross-device routing.

    Field design:

    - ``tier`` is the coarse routing hint: lite / standard / heavy.
      The director can ask "send heavy render jobs to a heavy node"
      without hard-coding hardware-specific combinations. Computed at
      extract time from the hardware signals below.
    - ``hw_encoder`` is the dedicated video encoder family. NVENC on
      NVIDIA, QSV on Intel iGPU, AMF on AMD. Dedicated silicon is
      separate from compute — encoding doesn't slow LLM inference, so
      a heavy LLM node can still serve as the cast-encode node.
    - ``can_*`` flags name the *outputs* this node can produce. Future
      output kinds (game stream, VR composition) extend by adding
      another flag with a schema_version bump — no fork needed.

    Forward compatibility: every field has a sensible default so older
    receivers parsing a future schema_version cap don't trip on missing
    fields. New flags should default ``False`` so a peer without the
    code path simply opts out.
    """

    kind: str = KIND_CAST_RENDER
    schema_version: int = 1

    # Routing tier — lite/standard/heavy. See class docstring.
    tier: str = "lite"

    # Hardware signal
    cpu_threads: int = 0
    gpu_vendor: str = ""              # "nvidia" | "amd" | "intel" | ""
    gpu_model: str = ""
    gpu_vram_gb: float = 0.0
    hw_encoder: str = ""              # "nvenc" | "qsv" | "amf" | ""
    max_concurrent_streams: int = 1

    # Output capabilities — what this node can actually produce.
    can_render_html: bool = False     # HTML/SVG → image (headless browser)
    can_render_vrm: bool = False      # 3D VRM avatar render
    can_encode_video: bool = False    # H.264/H.265 hardware encode
    can_stream_webrtc: bool = False   # peer→TV WebRTC stream setup


@dataclass(frozen=True)
class STTTranscribeCapability(CapabilityBase):
    """A peer that can serve STT transcription for one engine.

    Same shape principles as the TTS capability: enough metadata in the
    heartbeat for the receiver to make a routing decision without a
    follow-up fetch. Streaming-capable engines (Moonshine, Deepgram)
    advertise ``streaming=True``; batch-only engines (faster-whisper
    over HTTP) advertise ``streaming=False``.
    """

    kind: str = KIND_STT_TRANSCRIBE
    schema_version: int = 1
    engine: str = ""                  # "moonshine" | "deepgram" | "whisper" | ...
    provider_id: str = ""             # local id on the peer
    provider_name: str = ""
    base_url_path: str = "/v1/audio"
    default_model: str = ""
    languages: list = field(default_factory=list)  # ISO codes; empty = "multi / unknown"
    streaming: bool = False           # native streaming STT (WS) vs batch
    in_process: bool = False          # True = bundled (Moonshine); False = HTTP sidecar


# ── Serialisation ─────────────────────────────────────────────────


_CAPABILITY_TYPES: dict[str, type[CapabilityBase]] = {
    KIND_LLM_INFERENCE: LLMInferenceCapability,
    KIND_IMAGE_GENERATION: ImageGenerationCapability,
    KIND_KNOWLEDGE_SEARCH: KnowledgeSearchCapability,
    KIND_TTS_SYNTHESIZE: TTSSynthesizeCapability,
    KIND_STT_TRANSCRIBE: STTTranscribeCapability,
    KIND_CAST_RENDER: CastRenderCapability,
}


def serialise(capability: CapabilityBase) -> dict[str, Any]:
    """Capability → JSON-serialisable dict. Just asdict() today,
    but factored as a function so future schema evolutions can
    normalise fields here without touching every call site.
    """
    return asdict(capability)


def deserialise(data: dict[str, Any]) -> CapabilityBase | None:
    """JSON dict → Capability subclass. Returns ``None`` for unknown
    ``kind`` -- callers should ignore unknown capabilities rather
    than raise, so a peer running a newer version that advertises a
    kind we don't know about doesn't crash us.
    """
    if not isinstance(data, dict):
        return None
    kind = data.get("kind", "")
    cls = _CAPABILITY_TYPES.get(kind)
    if cls is None:
        return None
    # Filter to the fields the dataclass actually accepts; ignore
    # extra fields gracefully. This means a peer adding new optional
    # fields in a higher schema_version is forward-compatible with
    # our parser.
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    try:
        return cls(**filtered)
    except (TypeError, ValueError):
        # Schema mismatch (e.g. type drift). Skip rather than raise --
        # a single bad capability shouldn't drop the whole heartbeat.
        return None


def deserialise_list(items: list[Any]) -> list[CapabilityBase]:
    """Bulk-deserialise a wire-form capability list. Unknown kinds
    are silently dropped (same contract as deserialise()).
    """
    if not isinstance(items, list):
        return []
    out: list[CapabilityBase] = []
    for item in items:
        cap = deserialise(item)
        if cap is not None:
            out.append(cap)
    return out
