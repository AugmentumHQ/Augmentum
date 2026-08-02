"""Tests for augmentum/utils/http_client.py — shared HTTP client factory."""

from __future__ import annotations

import httpx

from augmentum.utils.http_client import (
    DEFAULT_TIMEOUT,
    SharedHTTPClient,
    is_local_url,
    normalize_base_url,
)


class TestIsLocalUrl:
    """Verify local URL detection logic."""

    def test_localhost(self):
        assert is_local_url("http://localhost:8080") is True

    def test_127_0_0_1(self):
        assert is_local_url("http://127.0.0.1:8080") is True

    def test_0_0_0_0(self):
        assert is_local_url("http://0.0.0.0:6100") is True

    def test_ipv6_loopback(self):
        assert is_local_url("http://[::1]:8080") is True

    def test_docker_internal(self):
        assert is_local_url("http://host.docker.internal:3000") is True

    def test_dot_local(self):
        assert is_local_url("http://myservice.local:8080") is True

    def test_cloud_url_not_local(self):
        assert is_local_url("https://api.openai.com/v1") is False

    def test_cloud_url_anthropic(self):
        assert is_local_url("https://api.anthropic.com/v1") is False

    def test_case_insensitive(self):
        assert is_local_url("http://LOCALHOST:8080") is True


class TestNormalizeBaseUrl:
    """Verify URL normalization strips trailing /v1."""

    def test_strips_trailing_v1(self):
        assert normalize_base_url("https://api.example.com/v1") == "https://api.example.com"

    def test_strips_trailing_v1_slash(self):
        assert normalize_base_url("https://api.example.com/v1/") == "https://api.example.com"

    def test_strips_trailing_slash(self):
        assert normalize_base_url("https://api.example.com/") == "https://api.example.com"

    def test_no_change_needed(self):
        assert normalize_base_url("https://api.example.com") == "https://api.example.com"


class TestSharedHTTPClient:
    """Verify SharedHTTPClient construction and SSL behavior."""

    def test_constructs_with_default_timeout(self):
        client = SharedHTTPClient()
        assert client._timeout == DEFAULT_TIMEOUT

    def test_constructs_with_custom_timeout(self):
        timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=3.0)
        client = SharedHTTPClient(timeout=timeout)
        assert client._timeout == timeout

    async def test_local_url_creates_unverified_client(self):
        client = SharedHTTPClient()
        async with client.get("http://localhost:8080") as c:
            # The unverified client should have been created
            assert client._unverified is not None
            assert client._verified is None
        await client.close()

    async def test_cloud_url_creates_verified_client(self):
        client = SharedHTTPClient()
        async with client.get("https://api.openai.com/v1") as c:
            assert client._verified is not None
        await client.close()

    async def test_close_is_idempotent(self):
        client = SharedHTTPClient()
        await client.close()
        await client.close()  # Should not raise
