"""Mode-prefix stripping at the resolver choke point.

Mode prefixes (``d/``, ``a/``, ``n/``, ``p/``, ``g/``, ``c/``) are normally
consumed by ``RequestClassifier._check_model_prefix``, which mutates
``request.model`` before resolution. Ingresses that never run the classifier
— the /v1/messages tools path (Claude Code), ``X-Augmentum-Mode``
header-override turns — used to pass the prefixed name straight to
``resolve_backend_with_fabric``, where it could never match a catalog entry
and (with fabric wired) raised::

    model 'd/deepseek-v4-pro' is not served by any local backend or
    connected fabric peer (...)

even though bare ``deepseek-v4-pro`` resolved fine. The resolver now strips
mode prefixes itself, mirroring the classifier's semantics on every path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from augmentum.models.provider_registry import ProviderRegistry


def _map_registry(model_map: dict, backends: dict) -> ProviderRegistry:
    reg = ProviderRegistry.__new__(ProviderRegistry)
    reg._backends = backends
    reg._model_map = model_map
    reg._model_pins = {}
    reg._lb_registry = None
    reg._default = ""
    reg._model_unverified = set()
    reg._unverified_served_logged = set()
    reg._fabric_director = None

    async def _noop_refresh(*a, **k):
        return reg._model_map

    reg.refresh_model_map = _noop_refresh  # type: ignore[method-assign]
    return reg


class _FakeDirector:
    """Fabric director whose peers advertise nothing (the failing setup)."""

    async def maybe_route_llm(self, **kwargs):
        return None

    def peer_diagnostic_for_llm(self, model):
        return {"connected_peers": ["peer-a"], "offline_peers": [], "peers": {}}


def test_direct_prefix_resolves_to_mapped_backend():
    # The reported bug: ``d/deepseek-v4-pro`` from a classifier-less ingress
    # must resolve exactly like the bare name does.
    cloud = SimpleNamespace(name="deepseek")
    reg = _map_registry({"deepseek-v4-pro": "deepseek"}, {"deepseek": cloud})

    backend, model = asyncio.run(reg.resolve_backend_for_model("d/deepseek-v4-pro"))
    assert backend is cloud
    assert model == "deepseek-v4-pro"


def test_all_mode_prefixes_strip():
    cloud = SimpleNamespace(name="deepseek")
    reg = _map_registry({"deepseek-v4-pro": "deepseek"}, {"deepseek": cloud})

    for prefix in ("p/", "a/", "n/", "g/", "c/", "d/"):
        backend, model = asyncio.run(
            reg.resolve_backend_for_model(f"{prefix}deepseek-v4-pro")
        )
        assert backend is cloud, prefix
        assert model == "deepseek-v4-pro", prefix


def test_prefix_with_backend_suffix_strips_too():
    # ``d/model@backend`` — the classifier would have stripped ``d/`` before
    # the ``@``-suffix branch ever saw it; the resolver must match.
    cloud = SimpleNamespace(name="deepseek")
    reg = _map_registry(
        {"deepseek-v4-pro@deepseek": "deepseek"}, {"deepseek": cloud}
    )

    backend, model = asyncio.run(
        reg.resolve_backend_for_model("d/deepseek-v4-pro@deepseek")
    )
    assert backend is cloud
    assert model == "deepseek-v4-pro"


def test_prefixed_model_does_not_raise_with_fabric_wired():
    # The exact regression: fabric director present, no peer serves the
    # model, but the BARE name is locally mapped. The prefixed request must
    # route to the local mapping instead of tripping the fabric fail-fast
    # ModelUnavailableError.
    cloud = SimpleNamespace(name="deepseek")
    reg = _map_registry({"deepseek-v4-pro": "deepseek"}, {"deepseek": cloud})
    reg._fabric_director = _FakeDirector()

    backend, model = asyncio.run(
        reg.resolve_backend_with_fabric("d/deepseek-v4-pro")
    )
    assert backend is cloud
    assert model == "deepseek-v4-pro"
