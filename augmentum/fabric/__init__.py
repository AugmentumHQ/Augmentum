"""Fabric layer: cross-instance peer coordination for distributed Augmentum.

Phase 0 shipped identity primitives. Phase 1 added transport: peer
pairing + WebSocket connections + the in-memory coordinator. Phase 2
adds capability advertisement: extractors that surface what each
peer can serve (LLM models, image pipelines, knowledge packs) into a
queryable registry that Phase 3's routing director will consume.

Every fabric code path is gated by the ``fabric_enabled`` setting
(default False), so a solo install never executes fabric logic.

See docs/superpowers/specs/2026-05-15-fabric-design.md for the full
design. See the integration plan synthesized in the session that
landed phase 0 for the regression-prevention discipline.
"""

from __future__ import annotations

from augmentum.fabric.capabilities import (
    CapabilityBase,
    ImageGenerationCapability,
    KnowledgeSearchCapability,
    KIND_IMAGE_GENERATION,
    KIND_KNOWLEDGE_SEARCH,
    KIND_LLM_INFERENCE,
    LLMInferenceCapability,
    deserialise,
    deserialise_list,
    serialise,
)
from augmentum.fabric.client import FabricClient
from augmentum.fabric.coordinator import FabricCoordinator, PeerLiveState
from augmentum.fabric.director import RoutingDirector
from augmentum.fabric.extractors import (
    ImageCapabilityExtractor,
    KnowledgeSearchExtractor,
    LLMCapabilityExtractor,
)
from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.peer_auth import (
    PairedPeer,
    PairRequest,
    PairRequestError,
)
from augmentum.fabric.peer_middleware import FabricPeerMiddleware
from augmentum.fabric.protocol import (
    PROTOCOL_VERSION,
    FabricEnvelope,
    FabricProtocolError,
)

__all__ = [
    "CapabilityBase",
    "FabricClient",
    "FabricCoordinator",
    "FabricEnvelope",
    "FabricIdentity",
    "FabricPeerMiddleware",
    "FabricProtocolError",
    "ImageCapabilityExtractor",
    "ImageGenerationCapability",
    "KnowledgeSearchCapability",
    "KnowledgeSearchExtractor",
    "KIND_IMAGE_GENERATION",
    "KIND_KNOWLEDGE_SEARCH",
    "KIND_LLM_INFERENCE",
    "LLMCapabilityExtractor",
    "LLMInferenceCapability",
    "PROTOCOL_VERSION",
    "PairRequest",
    "RoutingDirector",
    "PairRequestError",
    "PairedPeer",
    "PeerLiveState",
    "deserialise",
    "deserialise_list",
    "serialise",
]
