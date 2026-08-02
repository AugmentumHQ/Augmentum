"""Service-app aftercare — truthful install state, boot rehydration,
and secret persistence (the deleted-container / restart class bugs)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.marketplace.runtime_rehydrate import rehydrate_manifest_services
from augmentum.proxy.discover_routes import _enrich_service_listing

pytestmark = pytest.mark.asyncio


def _manifest_payload(svc_id: str = "memos") -> dict:
    return {
        "manifest_version": 1,
        "service": {
            "id": svc_id,
            "name": svc_id.title(),
            "image": "neosmemo/memos:0.29.1",
            "port": 5230,
            "volumes": {"data": "/var/opt/memos"},
        },
        "browser": {"after_install": "setup_page", "path": "/",
                    "credentials": "user_set"},
    }


def _listing(svc_id: str = "memos") -> SimpleNamespace:
    return SimpleNamespace(
        id=f"mkt:{svc_id}", kind="service",
        install_payload=_manifest_payload(svc_id),
    )


# ── _enrich_service_listing ──────────────────────────────────────────

class TestEnrichServiceListing:
    def _request(self, definition=None):
        mgr = MagicMock()
        mgr.get_definition.return_value = definition
        return SimpleNamespace(app=SimpleNamespace(
            state=SimpleNamespace(service_manager=mgr)))

    def test_installed_tracks_managed_row_not_user_record(self):
        # No managed_services row → NOT installed, no matter what the
        # per-user install record claims (the stale-state bug).
        d = {"install_payload": _manifest_payload()}
        _enrich_service_listing(self._request(), d, active_defs=set())
        assert d["installed"] is False

    def test_installed_service_gets_front_door_capabilities(self):
        sd = SimpleNamespace(https_port=9443, host_port=0)
        d = {"install_payload": _manifest_payload()}
        _enrich_service_listing(self._request(sd), d, active_defs={"memos"})
        assert d["installed"] is True
        assert d["capabilities"]["https_port"] == 9443
        assert d["capabilities"]["service_id"] == "memos"

    def test_missing_definition_degrades_to_installed_only(self):
        # Definition gone (pre-rehydrate window) → installed stays
        # truthful, capabilities just aren't attached.
        d = {"install_payload": _manifest_payload()}
        _enrich_service_listing(self._request(None), d, active_defs={"memos"})
        assert d["installed"] is True
        assert "capabilities" not in d or "https_port" not in (d.get("capabilities") or {})


# ── rehydrate_manifest_services ──────────────────────────────────────

class TestRehydrate:
    def _mgr(self, *, registered=None, config=None):
        mgr = MagicMock()
        mgr.get_definition.side_effect = lambda sid: (registered or {}).get(sid)
        mgr.read_config_json = AsyncMock(return_value=config or {})
        mgr.catalog.register_runtime = MagicMock()
        return mgr

    def _store(self, listings, active):
        store = MagicMock()
        store.install_wide_active_service_definitions = AsyncMock(return_value=active)
        store.list_for_discover = AsyncMock(return_value=listings)
        return store

    async def test_rehydrates_installed_service_with_persisted_port(self):
        mgr = self._mgr(config={"https_port": 9444})
        store = self._store([_listing()], {"memos"})
        count = await rehydrate_manifest_services(store, mgr)
        assert count == 1
        (sd,), _ = mgr.catalog.register_runtime.call_args
        assert sd.id == "memos"
        assert sd.https_port == 9444

    async def test_skips_not_installed_and_already_registered(self):
        mgr = self._mgr(registered={"memos": object()})
        store = self._store([_listing(), _listing("uninstalled")], {"memos"})
        count = await rehydrate_manifest_services(store, mgr)
        assert count == 0
        mgr.catalog.register_runtime.assert_not_called()

    async def test_no_active_rows_is_a_noop(self):
        mgr = self._mgr()
        store = self._store([_listing()], set())
        assert await rehydrate_manifest_services(store, mgr) == 0
        store.list_for_discover.assert_not_called()


# ── env_overrides persistence in ServiceManager ──────────────────────

class TestEnvOverridePersistence:
    async def test_read_and_update_config_json_roundtrip(self):
        from augmentum.providers.manager import ServiceManager

        rows = {"cfg": json.dumps({"volume_overrides": {"/x": "/y"}})}

        class FakeCursor:
            async def fetchone(self):
                return (rows["cfg"],)

        db = MagicMock()

        async def execute(sql, params=()):
            if sql.strip().startswith("SELECT"):
                return FakeCursor()
            # UPDATE — capture the merged blob
            rows["cfg"] = params[0]
            return FakeCursor()

        db.execute = AsyncMock(side_effect=execute)
        db.commit = AsyncMock()

        mgr = ServiceManager(docker=MagicMock(), db=db)
        await mgr.update_config_json("svc", {"env_overrides": {"PASSWORD": "s3cret"},
                                             "https_port": 9443})
        merged = await mgr.read_config_json("svc")
        # Old keys survive, new keys land — a recreate keeps its secrets.
        assert merged["volume_overrides"] == {"/x": "/y"}
        assert merged["env_overrides"] == {"PASSWORD": "s3cret"}
        assert merged["https_port"] == 9443
