"""Voice pipeline component resolver — picks client vs server vs peer dispatch.

The resolver is a pure function. Given a component (vad/stt/tts/denoise),
a consumer surface (call/companion/narration/readaloud), the client's
declared capabilities, the install-wide or per-user policy, and any
admin-pinned provider — it returns a single ``provider_id`` string that
the dispatch layer (audio_routes.py, voice_routes.py) consumes.

Provider ID grammar:
  - ``"server"``                       → server-side default (Silero VAD,
                                          local Moonshine / external STT, Kokoro/Pocket TTS, …)
  - ``"client:<engine>"``              → run in the browser (e.g.
                                          ``"client:silero-wasm"``)
  - ``"fabric:<node_id>:<provider>"``  → forwarded to a peer node
  - ``"<provider_id>"``                → explicit local provider row
                                          (e.g. ``"chatterbox-tts"``)

Policy values, one per surface:
  - ``"auto"``    → client if capability advertised, server otherwise
  - ``"local"``   → require client capability; raise ``ResolverError`` if missing
  - ``"server"``  → always server-side
  - ``"custom"``  → defer to the existing per-component routing knobs
                    (voice_routing_mode, stt_routing_mode, etc.) — i.e. the
                    resolver returns ``"server"`` and the caller proceeds
                    through the pre-resolver routing path unchanged

Test coverage in ``tests/test_pipeline_resolver.py``.
"""

from __future__ import annotations

from typing import Literal

Component = Literal["vad", "stt", "tts", "denoise"]
Surface = Literal["call", "companion", "narration", "readaloud"]
PolicyMode = Literal["auto", "local", "server", "custom"]

VALID_COMPONENTS: tuple[Component, ...] = ("vad", "stt", "tts", "denoise")
VALID_SURFACES: tuple[Surface, ...] = ("call", "companion", "narration", "readaloud")
VALID_MODES: tuple[PolicyMode, ...] = ("auto", "local", "server", "custom")


class ResolverError(ValueError):
    """Raised when ``mode='local'`` but the client did not advertise the capability."""


def _first_client_engine(
    client_caps: dict[str, list[str]] | None, component: Component
) -> str | None:
    """Return the first engine the client advertised for this component.

    ``None`` when capabilities are missing, the component key isn't present,
    or the list is empty. Caller treats ``None`` as "no client engine."
    """
    if not client_caps:
        return None
    engines = client_caps.get(component)
    if not engines or not isinstance(engines, list):
        return None
    for engine in engines:
        if isinstance(engine, str) and engine.strip():
            return engine.strip()
    return None


def resolve(
    component: Component,
    surface: Surface,
    *,
    client_caps: dict[str, list[str]] | None = None,
    policy: PolicyMode = "auto",
    pinned_provider: str = "",
) -> str:
    """Pick a single provider for one component on one surface.

    Args:
        component: One of ``"vad" | "stt" | "tts" | "denoise"``.
        surface: One of ``"call" | "companion" | "narration" | "readaloud"``.
        client_caps: The browser's advertised capability map. Empty dict /
            ``None`` means the client has no local primitives — server only.
        policy: The user/admin policy for this surface.
        pinned_provider: A specific provider ID to force. Wins over policy
            and capability detection. Empty string means no pin.

    Returns:
        A provider ID string (see grammar in the module docstring).

    Raises:
        ResolverError: When ``policy == "local"`` but the client did not
            advertise an engine for ``component``.
        ValueError: When ``component`` / ``surface`` / ``policy`` are unknown.
    """
    if component not in VALID_COMPONENTS:
        raise ValueError(f"unknown component: {component!r}")
    if surface not in VALID_SURFACES:
        raise ValueError(f"unknown surface: {surface!r}")
    if policy not in VALID_MODES:
        raise ValueError(f"unknown policy: {policy!r}")

    # Explicit pin always wins — admin override, debugging, A/B testing.
    if pinned_provider:
        return pinned_provider

    if policy == "server":
        return "server"

    # Custom mode defers to the legacy per-component routing knobs. The
    # resolver returns "server" so the caller's existing dispatch (which
    # consults voice_routing_mode / stt_routing_mode / etc.) proceeds
    # unchanged.
    if policy == "custom":
        return "server"

    client_engine = _first_client_engine(client_caps, component)

    if policy == "local":
        if client_engine is None:
            raise ResolverError(
                f"policy=local for {surface}/{component} but client advertised no engine"
            )
        return f"client:{client_engine}"

    # policy == "auto" — prefer client when available, fall back to server.
    if client_engine is not None:
        return f"client:{client_engine}"
    return "server"
