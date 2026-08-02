"""Service-app manifests + the generic dispatcher (apps-as-data, phase 1).

Spec: docs/superpowers/specs/2026-07-18-marketplace-service-os-design.md
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.marketplace.manifest import (
    ManifestError,
    parse_manifest,
    to_service_definition,
)


def _minimal(**over) -> dict:
    m = {
        "manifest_version": 1,
        "service": {
            "id": "navidrome",
            "name": "Navidrome",
            "image": "deluan/navidrome:0.52.5",
            "port": 4533,
            "healthcheck": {"path": "/ping", "timeout_s": 30},
        },
        "browser": {"after_install": "setup_page", "path": "/",
                    "credentials": "generated"},
    }
    m.update(over)
    return m


class TestParseManifest:
    def test_minimal_valid(self):
        m = parse_manifest(_minimal())
        assert m.service_id == "navidrome"
        assert m.internal_port == 4533
        assert m.healthcheck_path == "/ping"
        assert m.browser_after_install == "setup_page"

    def test_browser_block_required(self):
        payload = _minimal()
        del payload["browser"]
        with pytest.raises(ManifestError, match="browser"):
            parse_manifest(payload)

    def test_latest_tag_rejected(self):
        payload = _minimal()
        payload["service"]["image"] = "deluan/navidrome:latest"
        with pytest.raises(ManifestError, match="latest"):
            parse_manifest(payload)

    def test_untagged_image_rejected(self):
        payload = _minimal()
        payload["service"]["image"] = "deluan/navidrome"
        with pytest.raises(ManifestError, match="no tag"):
            parse_manifest(payload)

    def test_digest_pinned_accepted(self):
        payload = _minimal()
        payload["service"]["image"] = "deluan/navidrome@sha256:" + "a" * 64
        assert parse_manifest(payload).image.endswith("a" * 64)

    def test_registry_port_not_mistaken_for_tag(self):
        payload = _minimal()
        payload["service"]["image"] = "registry.local:5000/navidrome:0.52.5"
        assert parse_manifest(payload).image.startswith("registry.local:5000/")

    def test_unknown_integration_hook_dropped_not_fatal(self):
        payload = _minimal(
            integration={"quantum_sync": {"x": 1}, "media_connect": {}},
        )
        m = parse_manifest(payload)
        assert "quantum_sync" not in m.integration
        assert "media_connect" in m.integration

    def test_bad_version_rejected(self):
        with pytest.raises(ManifestError, match="manifest_version"):
            parse_manifest(_minimal(manifest_version=99))

    def test_env_prompts_parsed(self):
        payload = _minimal()
        payload["service"]["env_prompts"] = [
            {"key": "ND_DEFAULTLANGUAGE", "label": "Language", "default": "en"},
        ]
        m = parse_manifest(payload)
        assert m.env_prompts[0].key == "ND_DEFAULTLANGUAGE"

    def test_to_service_definition_maps_core_fields(self):
        sd = to_service_definition(parse_manifest(_minimal()))
        assert sd.id == "navidrome"
        assert sd.image == "deluan/navidrome:0.52.5"
        assert sd.internal_port == 4533
        assert sd.health_endpoint == "/ping"
        assert sd.category.value == "service"

    def test_media_hook_maps_to_media_category(self):
        payload = _minimal(integration={"media_connect": {"provider": "jellyfin"}})
        sd = to_service_definition(parse_manifest(payload))
        assert sd.category.value == "media"


class TestCatalogGate:
    """T4: a service listing with a bad manifest never enters the catalog."""

    def test_valid_service_listing_builds(self):
        from augmentum.marketplace.catalog_loader import _validate_and_build
        listing = _validate_and_build({
            "id": "svc-navidrome",
            "title": "Navidrome",
            "kind": "service",
            "install_via": "service_manifest",
            "install_payload": _minimal(),
        })
        assert listing.kind == "service"

    def test_missing_browser_block_rejected(self):
        from augmentum.marketplace.catalog_loader import _validate_and_build
        payload = _minimal()
        del payload["browser"]
        with pytest.raises(ValueError, match="invalid service manifest"):
            _validate_and_build({
                "id": "svc-bad",
                "title": "Bad",
                "kind": "service",
                "install_via": "service_manifest",
                "install_payload": payload,
            })

    def test_wrong_install_via_rejected(self):
        from augmentum.marketplace.catalog_loader import _validate_and_build
        with pytest.raises(ValueError, match="service_manifest"):
            _validate_and_build({
                "id": "svc-bad2",
                "title": "Bad2",
                "kind": "service",
                "install_via": "media_server",
                "install_payload": _minimal(),
            })


def _fake_request(admin: bool = True, mgr=None):
    app = SimpleNamespace(state=SimpleNamespace(
        service_manager=mgr, http_client=None,
    ))
    req = MagicMock()
    req.app = app
    req.scope = {"user": SimpleNamespace(id="u1", role="admin" if admin else "user")}
    return req


def _fake_mgr():
    mgr = MagicMock()
    mgr.catalog = MagicMock()
    mgr.catalog.register_runtime = MagicMock()
    mgr.get_definition = MagicMock(return_value=None)
    mgr.enable_service = AsyncMock()
    mgr.disable_service = AsyncMock()
    return mgr


class TestGenericDispatcher:
    @pytest.mark.asyncio
    async def test_install_provisions_and_registers_runtime_definition(self, monkeypatch):
        from augmentum.marketplace import install_dispatchers as d
        monkeypatch.setattr("augmentum.auth.guards.is_admin", lambda r: True)
        mgr = _fake_mgr()
        req = _fake_request(mgr=mgr)
        result = await d._install_service_manifest(req, _minimal(), "u1")
        assert result == "navidrome"
        mgr.catalog.register_runtime.assert_called_once()
        mgr.enable_service.assert_awaited_once()
        _, kwargs = mgr.enable_service.await_args
        assert kwargs.get("env_overrides") is None

    @pytest.mark.asyncio
    async def test_install_rejects_invalid_manifest(self, monkeypatch):
        from fastapi import HTTPException

        from augmentum.marketplace import install_dispatchers as d
        monkeypatch.setattr("augmentum.auth.guards.is_admin", lambda r: True)
        req = _fake_request(mgr=_fake_mgr())
        bad = _minimal()
        bad["service"]["image"] = "x:latest"
        with pytest.raises(HTTPException) as exc:
            await d._install_service_manifest(req, bad, "u1")
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_install_admin_only(self, monkeypatch):
        from fastapi import HTTPException

        from augmentum.marketplace import install_dispatchers as d
        monkeypatch.setattr("augmentum.auth.guards.is_admin", lambda r: False)
        req = _fake_request(admin=False, mgr=_fake_mgr())
        with pytest.raises(HTTPException) as exc:
            await d._install_service_manifest(req, _minimal(), "u1")
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_env_prompt_answers_filtered_to_declared_keys(self, monkeypatch):
        from augmentum.marketplace import install_dispatchers as d
        monkeypatch.setattr("augmentum.auth.guards.is_admin", lambda r: True)
        mgr = _fake_mgr()
        req = _fake_request(mgr=mgr)
        payload = _minimal()
        payload["service"]["env_prompts"] = [{"key": "ND_LANG", "label": "L"}]
        payload["_install_options"] = {
            "env": {"ND_LANG": "en", "EVIL_INJECTED": "1"},
        }
        await d._install_service_manifest(req, payload, "u1")
        _, kwargs = mgr.enable_service.await_args
        assert kwargs["env_overrides"] == {"ND_LANG": "en"}

    @pytest.mark.asyncio
    async def test_uninstall_disables_service_preserving_volumes(self, monkeypatch):
        from augmentum.marketplace import install_dispatchers as d
        monkeypatch.setattr("augmentum.auth.guards.is_admin", lambda r: True)
        mgr = _fake_mgr()
        req = _fake_request(mgr=mgr)
        await d._uninstall_service_manifest(req, _minimal(), "u1")
        mgr.disable_service.assert_awaited_once_with("navidrome")

    @pytest.mark.asyncio
    async def test_media_hook_routes_through_shared_connect_helper(self, monkeypatch):
        from augmentum.marketplace import install_dispatchers as d
        monkeypatch.setattr("augmentum.auth.guards.is_admin", lambda r: True)
        connected = AsyncMock(return_value="srv_row_1")
        monkeypatch.setattr(d, "_connect_media_server", connected)
        mgr = _fake_mgr()
        req = _fake_request(mgr=mgr)
        payload = _minimal(integration={"media_connect": {"provider": "jellyfin"}})
        result = await d._install_service_manifest(req, payload, "u1")
        assert result == "srv_row_1"
        assert connected.await_args.kwargs["provider"] == "jellyfin"


class TestRuntimeCatalogRegistration:
    def test_runtime_definition_never_shadows_catalog(self, tmp_path):
        from augmentum.providers.catalog import ProviderCatalog
        from augmentum.providers.models import ServiceCategory, ServiceDefinition

        cat = ProviderCatalog(path=tmp_path / "missing.json")  # empty catalog
        sd = ServiceDefinition(
            id="navidrome", name="n", description="", category=ServiceCategory.SERVICE,
            image="deluan/navidrome:0.52.5", internal_port=4533, host_port=0,
        )
        cat.register_runtime(sd)
        assert cat.get("navidrome") is sd
        # Second registration with same id refuses to shadow.
        sd2 = ServiceDefinition(
            id="navidrome", name="other", description="", category=ServiceCategory.SERVICE,
            image="x:1", internal_port=1, host_port=0,
        )
        cat.register_runtime(sd2)
        assert cat.get("navidrome") is sd


class TestFrontDoorGeneralization:
    """Phase 2: every service app gets the HTTPS front door, not just media."""

    def test_allocator_picks_free_port(self):
        from augmentum.providers.caddy_front_door import (
            FRONT_DOOR_PORT_MIN,
            allocate_front_door_port,
        )
        assert allocate_front_door_port(set()) == FRONT_DOOR_PORT_MIN
        assert allocate_front_door_port({FRONT_DOOR_PORT_MIN}) == FRONT_DOOR_PORT_MIN + 1

    def test_allocator_exhausted_returns_zero(self):
        from augmentum.providers.caddy_front_door import (
            FRONT_DOOR_PORT_MAX,
            FRONT_DOOR_PORT_MIN,
            allocate_front_door_port,
        )
        used = set(range(FRONT_DOOR_PORT_MIN, FRONT_DOOR_PORT_MAX + 1))
        assert allocate_front_door_port(used) == 0

    @pytest.mark.asyncio
    async def test_dispatcher_allocates_https_port(self, monkeypatch):
        from augmentum.marketplace import install_dispatchers as d
        from augmentum.providers.caddy_front_door import (
            FRONT_DOOR_PORT_MAX,
            FRONT_DOOR_PORT_MIN,
        )
        monkeypatch.setattr("augmentum.auth.guards.is_admin", lambda r: True)
        mgr = _fake_mgr()
        mgr.catalog.list_all = MagicMock(return_value=[])
        req = _fake_request(mgr=mgr)
        await d._install_service_manifest(req, _minimal(), "u1")
        (sd,), _ = mgr.catalog.register_runtime.call_args
        assert FRONT_DOOR_PORT_MIN <= sd.https_port <= FRONT_DOOR_PORT_MAX

    @pytest.mark.asyncio
    async def test_manager_applies_front_door_for_service_category(self, monkeypatch):
        from augmentum.providers.manager import ServiceManager
        from augmentum.providers.models import ServiceCategory, ServiceDefinition

        applied = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "augmentum.providers.caddy_front_door.apply_front_door", applied,
        )
        mgr = ServiceManager(MagicMock(), None)
        sd = ServiceDefinition(
            id="uptime-kuma", name="k", description="",
            category=ServiceCategory.SERVICE,
            image="louislam/uptime-kuma:2.4.0", internal_port=3001,
            host_port=0, https_port=6805,
        )
        await mgr._apply_front_door_if_media(sd)
        applied.assert_awaited_once()
        # Generic service apps get the ACCESS gate — forward_auth gates access,
        # the app keeps its own login (the unbounded <svc>.<gate_domain> door).
        assert applied.await_args.kwargs["gate_mode"] == "access"

    @pytest.mark.asyncio
    async def test_manager_skips_front_door_without_port(self, monkeypatch):
        from augmentum.providers.manager import ServiceManager
        from augmentum.providers.models import ServiceCategory, ServiceDefinition

        applied = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "augmentum.providers.caddy_front_door.apply_front_door", applied,
        )
        mgr = ServiceManager(MagicMock(), None)
        sd = ServiceDefinition(
            id="x", name="x", description="", category=ServiceCategory.SERVICE,
            image="a:1", internal_port=1, host_port=0, https_port=0,
        )
        await mgr._apply_front_door_if_media(sd)
        applied.assert_not_awaited()


class TestShippedCatalog:
    """Every shipped service listing must survive the real validation gate
    — a catalog edit that breaks a manifest fails here, not at boot."""

    def test_all_shipped_listings_validate(self):
        import json
        from pathlib import Path

        from augmentum.marketplace.catalog_loader import _validate_and_build
        path = Path("data/marketplace/listings.json")
        doc = json.loads(path.read_text(encoding="utf-8"))
        for entry in doc["listings"]:
            listing = _validate_and_build(entry)
            if listing.kind == "service":
                # Redundant with the gate, but explicit: pinned image.
                image = listing.install_payload["service"]["image"]
                assert ":latest" not in image


class TestBundles:
    """Phase 3: a profile IS a bundle — members install via their own
    dispatchers, per-member isolation, honest receipts."""

    @staticmethod
    def _bundle_request(store, mgr=None):
        req = _fake_request(mgr=mgr or _fake_mgr())
        req.app.state.marketplace_store = store
        return req

    @staticmethod
    def _listing(id_, kind="service", install_via="service_manifest", payload=None):
        return SimpleNamespace(
            id=id_, kind=kind, install_via=install_via,
            install_payload=payload if payload is not None else _minimal(),
        )

    @pytest.mark.asyncio
    async def test_bundle_installs_each_member(self, monkeypatch):
        from augmentum.marketplace import install_dispatchers as d
        calls = []

        async def fake_svc(request, artifact, user_id):
            calls.append(artifact["service"]["id"])
            return "ok"

        monkeypatch.setitem(d.DISPATCHER_REGISTRY, "service_manifest", fake_svc)
        store = MagicMock()
        listings = {
            "mkt:a": self._listing("mkt:a"),
            "mkt:b": self._listing("mkt:b"),
        }
        store.get = AsyncMock(side_effect=lambda i: listings.get(i))
        req = self._bundle_request(store)
        summary = await d._install_bundle(
            req, {"members": ["mkt:a", "mkt:b"]}, "u1",
        )
        assert len(calls) == 2
        assert "installed 2/2" in summary

    @pytest.mark.asyncio
    async def test_bundle_isolates_member_failure(self, monkeypatch):
        from fastapi import HTTPException

        from augmentum.marketplace import install_dispatchers as d

        async def flaky(request, artifact, user_id):
            if artifact["service"]["id"] == "navidrome":
                raise HTTPException(status_code=500, detail="pull failed")
            return "ok"

        monkeypatch.setitem(d.DISPATCHER_REGISTRY, "service_manifest", flaky)
        good = _minimal()
        good["service"] = dict(good["service"], id="uptimekuma")
        store = MagicMock()
        listings = {
            "mkt:bad": self._listing("mkt:bad"),          # navidrome → fails
            "mkt:good": self._listing("mkt:good", payload=good),
        }
        store.get = AsyncMock(side_effect=lambda i: listings.get(i))
        req = self._bundle_request(store)
        summary = await d._install_bundle(
            req, {"members": ["mkt:bad", "mkt:good"]}, "u1",
        )
        # One failed, one landed — the receipt says both, loudly.
        assert "installed 1/2" in summary
        assert "mkt:bad" in summary and "pull failed" in summary

    @pytest.mark.asyncio
    async def test_bundle_rejects_nested_bundles(self):
        from fastapi import HTTPException

        from augmentum.marketplace import install_dispatchers as d
        store = MagicMock()
        store.get = AsyncMock(return_value=self._listing(
            "mkt:inner", kind="bundle", install_via="bundle",
            payload={"members": ["x"]},
        ))
        req = self._bundle_request(store)
        # The only member is a nested bundle → everything failed → raises
        # (all-failed bundles are an error, not a quiet no-op receipt).
        with pytest.raises(HTTPException) as exc:
            await d._install_bundle(req, {"members": ["mkt:inner"]}, "u1")
        assert "nested bundles" in str(exc.value.detail)

    @pytest.mark.asyncio
    async def test_bundle_all_failed_raises(self):
        from fastapi import HTTPException

        from augmentum.marketplace import install_dispatchers as d
        store = MagicMock()
        store.get = AsyncMock(return_value=None)  # nothing resolvable
        req = self._bundle_request(store)
        with pytest.raises(HTTPException) as exc:
            await d._install_bundle(req, {"members": ["mkt:ghost"]}, "u1")
        assert "not in catalog" in str(exc.value.detail)

    def test_catalog_gate_accepts_bundle_shape(self):
        from augmentum.marketplace.catalog_loader import _validate_and_build
        listing = _validate_and_build({
            "id": "mkt:bundle-core",
            "title": "Core",
            "kind": "bundle",
            "install_via": "bundle",
            "install_payload": {"members": ["mkt:uptime-kuma"]},
        })
        assert listing.kind == "bundle"

    def test_catalog_gate_rejects_empty_members(self):
        from augmentum.marketplace.catalog_loader import _validate_and_build
        with pytest.raises(ValueError, match="members"):
            _validate_and_build({
                "id": "mkt:bundle-bad",
                "title": "Bad",
                "kind": "bundle",
                "install_via": "bundle",
                "install_payload": {"members": []},
            })
