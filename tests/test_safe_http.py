"""Tests for augmentum/utils/safe_http.py — SSRF prevention."""

from __future__ import annotations

import pytest

from augmentum.utils.safe_http import (
    SafeHttpClient,
    SafeHttpError,
    _is_ip_blocked,
    check_ssrf,
    parse_ssrf_allowlist,
)


class TestIsIpBlocked:
    """Verify IP blocking for private/reserved ranges."""

    def test_blocks_loopback_127(self):
        assert _is_ip_blocked("127.0.0.1") is True

    def test_blocks_loopback_127_x(self):
        assert _is_ip_blocked("127.255.255.255") is True

    def test_blocks_10_range(self):
        assert _is_ip_blocked("10.0.0.1") is True
        assert _is_ip_blocked("10.255.255.255") is True

    def test_blocks_172_16_range(self):
        assert _is_ip_blocked("172.16.0.1") is True
        assert _is_ip_blocked("172.31.255.255") is True

    def test_does_not_block_172_32(self):
        assert _is_ip_blocked("172.32.0.1") is False

    def test_blocks_192_168_range(self):
        assert _is_ip_blocked("192.168.0.1") is True
        assert _is_ip_blocked("192.168.255.255") is True

    def test_blocks_link_local(self):
        assert _is_ip_blocked("169.254.0.1") is True

    def test_blocks_ipv6_loopback(self):
        assert _is_ip_blocked("::1") is True

    def test_blocks_ipv6_unique_local(self):
        assert _is_ip_blocked("fc00::1") is True
        assert _is_ip_blocked("fd12::1") is True

    def test_blocks_ipv6_link_local(self):
        assert _is_ip_blocked("fe80::1") is True

    def test_allows_public_ip(self):
        assert _is_ip_blocked("8.8.8.8") is False
        assert _is_ip_blocked("1.1.1.1") is False

    def test_allows_public_ip_v6(self):
        assert _is_ip_blocked("2607:f8b0:4004:800::200e") is False

    def test_blocks_invalid_ip(self):
        assert _is_ip_blocked("not-an-ip") is True


class TestCheckSsrf:
    """Verify async SSRF checks."""

    async def test_blocks_private_ip_literal(self):
        with pytest.raises(SafeHttpError, match="Blocked IP"):
            await check_ssrf("http://10.0.0.1/api")

    async def test_blocks_loopback_literal(self):
        with pytest.raises(SafeHttpError, match="Blocked IP"):
            await check_ssrf("http://127.0.0.1/api")

    async def test_blocks_bad_scheme(self):
        with pytest.raises(SafeHttpError, match="Blocked scheme"):
            await check_ssrf("ftp://example.com/file")

    async def test_blocks_no_hostname(self):
        with pytest.raises(SafeHttpError):
            await check_ssrf("http://")

    async def test_allowlist_permits_hostname(self):
        # A private IP that is allowlisted should pass
        await check_ssrf("http://10.0.0.1/api", allowlist=["10.0.0.1"])

    async def test_allowlist_permits_cidr(self):
        await check_ssrf("http://172.18.0.5/api", allowlist=["172.18.0.0/16"])


class TestParseSsrfAllowlist:
    """Verify allowlist parsing."""

    def test_empty_string(self):
        assert parse_ssrf_allowlist("") == []

    def test_single_entry(self):
        assert parse_ssrf_allowlist("searxng") == ["searxng"]

    def test_multiple_entries(self):
        result = parse_ssrf_allowlist("searxng,executor,ollama")
        assert result == ["searxng", "executor", "ollama"]

    def test_strips_whitespace(self):
        result = parse_ssrf_allowlist(" searxng , executor ")
        assert result == ["searxng", "executor"]


class TestSafeHttpClient:
    """Verify SafeHttpClient URL validation."""

    def test_blocks_ftp_scheme(self):
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked scheme"):
            client._validate_url("ftp://example.com")

    def test_blocks_file_scheme(self):
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked scheme"):
            client._validate_url("file:///etc/passwd")

    def test_allows_http(self):
        client = SafeHttpClient()
        hostname = client._validate_url("http://example.com/page")
        assert hostname == "example.com"

    def test_allows_https(self):
        client = SafeHttpClient()
        hostname = client._validate_url("https://example.com/page")
        assert hostname == "example.com"


class TestFetchBytes:
    """fetch_bytes must inherit the same SSRF protections as fetch — it's the
    binary-safe sibling used for avatar/image downloads."""

    async def test_blocks_bad_scheme(self):
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked scheme"):
            await client.fetch_bytes("file:///etc/passwd")

    async def test_blocks_loopback_literal(self):
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked"):
            await client.fetch_bytes("http://127.0.0.1/avatar.png")

    async def test_blocks_private_ip_literal(self):
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked"):
            await client.fetch_bytes("http://10.0.0.1/avatar.png")
