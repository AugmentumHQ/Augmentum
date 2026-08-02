"""Capability discovery frontdesk and host adapters."""

from augmentum.capabilities.frontdesk import (
    CapabilityContext,
    CapabilityFrontdesk,
    build_default_frontdesk,
    context_from_request,
)

__all__ = [
    "CapabilityContext",
    "CapabilityFrontdesk",
    "build_default_frontdesk",
    "context_from_request",
]
