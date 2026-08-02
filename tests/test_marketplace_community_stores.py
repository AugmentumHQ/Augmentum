"""Community app stores (service-OS phase 4) — add-by-URL, namespace
enforcement, gate parity with the shipped catalog, per-store isolation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.marketplace.loaders.stores import (
    add_store,
    list_stores,
    remove_store,
    slug_for_url,
    sync_store,
)


class _FakeSettings:
    def __init__(self):
        self.data: dict[str, str] = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value):
        self.data[key] = value


def _http_returning(payload: dict, status: int = 200):
    http = MagicMock()
    body = json.dumps(payload).encode("utf-8")
    resp = SimpleNamespace(
        content=body,
        status_code=status,
        raise_for_status=lambda: None,
    )
    http.get = AsyncMock(return_value=resp)
    return http


def _svc_entry(id_="uptime", image="louislam/uptime-kuma:2.4.0"):
    return {
        "id": id_,
        "title": "Kuma",
        "kind": "service",
        "install_via": "service_manifest",
        "install_payload": {
            "manifest_version": 1,
            "service": {"id": "uptime-kuma", "image": image, "port": 3001},
            "browser": {"after_install": "setup_page", "path": "/"},
        },
    }


class TestRegistry:
    @pytest.mark.asyncio
    async def test_add_requires_https(self):
        with pytest.raises(ValueError, match="https"):
            await add_store(_FakeSettings(), url="http://insecure.example/apps.json")

    @pytest.mark.asyncio
    async def test_add_list_roundtrip_and_duplicate_rejected(self):
        st = _FakeSettings()
        entry = await add_store(st, url="https://apps.example/store.json", name="Ex")
        assert entry["slug"] == slug_for_url("https://apps.example/store.json")
        assert (await list_stores(st))[0]["url"] == "https://apps.example/store.json"
        with pytest.raises(ValueError, match="already"):
            await add_store(st, url="https://apps.example/store.json")

    @pytest.mark.asyncio
    async def test_remove_delists_publisher(self):
        st = _FakeSettings()
        entry = await add_store(st, url="https://apps.example/store.json")
        mstore = MagicMock()
        mstore.delist_missing_for_publisher = AsyncMock(return_value=3)
        delisted = await remove_store(st, mstore, entry["slug"])
        assert delisted == 3
        assert await list_stores(st) == []
        mstore.delist_missing_for_publisher.assert_awaited_once_with(
            set(), publisher=f"community:{entry['slug']}",
        )


class TestSync:
    @pytest.mark.asyncio
    async def test_sync_namespaces_ids_and_publisher(self, monkeypatch):
        monkeypatch.setattr(
            "augmentum.utils.safe_http.check_ssrf", AsyncMock(),
        )
        mstore = MagicMock()
        upserted = []
        mstore.upsert = AsyncMock(side_effect=lambda row: upserted.append(row))
        mstore.delist_missing_for_publisher = AsyncMock(return_value=0)
        http = _http_returning({"listings": [
            # Claims an official id AND an official publisher — both get
            # overridden into the community namespace.
            {**_svc_entry("mkt:uptime-kuma"), "publisher": "augmentum"},
        ]})
        entry = {"slug": "ex-123", "url": "https://apps.example/store.json"}
        stats = await sync_store(mstore, http, entry)
        assert stats["loaded"] == 1
        listing = upserted[0]
        assert listing.publisher == "community:ex-123"
        assert listing.id == "community:ex-123:mkt:uptime-kuma"

    @pytest.mark.asyncio
    async def test_sync_gate_rejects_bad_manifest_entries(self, monkeypatch):
        monkeypatch.setattr(
            "augmentum.utils.safe_http.check_ssrf", AsyncMock(),
        )
        mstore = MagicMock()
        mstore.upsert = AsyncMock()
        mstore.delist_missing_for_publisher = AsyncMock(return_value=0)
        http = _http_returning({"listings": [
            _svc_entry("good"),
            _svc_entry("bad", image="evil/app:latest"),   # unpinned → gate
        ]})
        stats = await sync_store(
            mstore, http, {"slug": "s1", "url": "https://x.example/s.json"},
        )
        assert stats == {"loaded": 1, "skipped": 1, "delisted": 0}

    @pytest.mark.asyncio
    async def test_sync_rejects_oversize_payload(self, monkeypatch):
        monkeypatch.setattr(
            "augmentum.utils.safe_http.check_ssrf", AsyncMock(),
        )
        http = MagicMock()
        resp = SimpleNamespace(
            content=b"x" * (2 * 1024 * 1024 + 10),
            status_code=200,
            raise_for_status=lambda: None,
        )
        http.get = AsyncMock(return_value=resp)
        with pytest.raises(ValueError, match="exceeds"):
            await sync_store(
                MagicMock(), http,
                {"slug": "s1", "url": "https://x.example/s.json"},
            )

    @pytest.mark.asyncio
    async def test_sync_delists_entries_the_store_dropped(self, monkeypatch):
        monkeypatch.setattr(
            "augmentum.utils.safe_http.check_ssrf", AsyncMock(),
        )
        mstore = MagicMock()
        mstore.upsert = AsyncMock()
        mstore.delist_missing_for_publisher = AsyncMock(return_value=2)
        http = _http_returning({"listings": [_svc_entry("only-one")]})
        stats = await sync_store(
            mstore, http, {"slug": "s1", "url": "https://x.example/s.json"},
        )
        assert stats["delisted"] == 2
        args, kwargs = mstore.delist_missing_for_publisher.await_args
        assert args[0] == {"community:s1:only-one"}
        assert kwargs["publisher"] == "community:s1"
