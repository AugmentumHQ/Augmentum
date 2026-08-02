"""MCP client config catalog.

Eight popular MCP clients with paste-ready snippets pointing back at
Augmentum's own ``/mcp`` server. The accept handler doesn't mutate
Augmentum state — these chips return a ready-to-paste config string
the user puts in their client's config file.

User-scoped (no admin gate): showing a config snippet for the user's
own desktop client is not a privileged action.

Schema sources verified 2026-06-02:
- Claude Desktop:  ``mcpServers.<name>.{url, headers}``
                   ``claude_desktop_config.json``
- Cursor:          ``mcpServers.<name>.{url, headers}``    ``~/.cursor/mcp.json``
- Cline:           ``mcpServers.<name>.{url, headers}``    ``cline_mcp_settings.json``
- Continue:        modern is ``config.yaml`` with top-level ``mcpServers``
                   list. Streamable HTTP shape: ``- name + type:
                   streamable-http + url``. Legacy
                   ``experimental.modelContextProtocolServers`` in
                   ``config.json`` still works but is deprecated.
- Windsurf:        ``mcpServers.<name>.{serverUrl, headers}``
                   ``~/.codeium/windsurf/mcp_config.json``
- VS Code native:  ``servers.<name>.{type:"http", url, headers}``
                   ``.vscode/mcp.json`` or settings.json ``mcp.servers``
- Zed (2026 docs): ``context_servers.<name>.{url, headers}`` —
                   native field added in 2026; older docs/examples
                   use an ``mcp-remote`` npx bridge instead.
- Generic:         same as Claude Desktop shape (the de-facto common
                   denominator).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from augmentum.offers.catalog.base import (
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request


log = get_logger(__name__)


KIND: str = "mcp_client_config"


# ── URL resolution ───────────────────────────────────────────────


def _augmentum_mcp_url(request: Request) -> str:
    """Derive the user-facing Augmentum /mcp URL from the request.

    We can't hardcode this — Augmentum runs behind reverse proxies,
    on custom ports, on Tailscale, etc. The request's own URL is the
    authoritative origin.
    """

    # ``request.url`` is the resolved request URL; the base URL drops
    # path + query. Strip trailing slash for clean concatenation.
    base = str(request.base_url).rstrip("/")
    return f"{base}/mcp"


# ── Snippet builders ─────────────────────────────────────────────


def _pretty_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2)


def _build_mcpservers_json(url: str) -> str:
    """The most common shape — Claude Desktop / Cursor / Cline / Generic.

    Drops the placeholder API key into the Authorization header so the
    snippet works end-to-end the moment the user pastes their actual
    key in place.
    """

    return _pretty_json({
        "mcpServers": {
            "augmentum": {
                "url": url,
                "headers": {"Authorization": "Bearer sk-aug-YOUR-KEY-HERE"},
            },
        },
    })


def _build_continue_yaml(url: str) -> str:
    # Continue moved to config.yaml in 2026; the YAML form is the
    # supported one for new installs. Keep this hand-written rather
    # than pulling in PyYAML — the shape is trivial.
    return (
        "# Append to ~/.continue/config.yaml\n"
        "mcpServers:\n"
        "  - name: augmentum\n"
        "    type: streamable-http\n"
        f"    url: {url}\n"
        "    requestOptions:\n"
        "      headers:\n"
        "        Authorization: Bearer sk-aug-YOUR-KEY-HERE\n"
    )


def _build_windsurf_json(url: str) -> str:
    return _pretty_json({
        "mcpServers": {
            "augmentum": {
                "serverUrl": url,
                "headers": {"Authorization": "Bearer sk-aug-YOUR-KEY-HERE"},
            },
        },
    })


def _build_vscode_json(url: str) -> str:
    return _pretty_json({
        "servers": {
            "augmentum": {
                "type": "http",
                "url": url,
                "headers": {"Authorization": "Bearer sk-aug-YOUR-KEY-HERE"},
            },
        },
    })


def _build_zed_json(url: str) -> str:
    # 2026 docs show native ``url`` + ``headers`` directly under
    # context_servers.<name>. Older examples use an mcp-remote
    # subprocess bridge — Zed will prompt for OAuth if Authorization
    # is unset, so we ship the bearer-header shape pre-filled.
    return _pretty_json({
        "context_servers": {
            "augmentum": {
                "url": url,
                "headers": {"Authorization": "Bearer sk-aug-YOUR-KEY-HERE"},
            },
        },
    })


# ── Catalog records ──────────────────────────────────────────────


SnippetBuilder = Callable[[str], str]


_CLIENTS: list[dict[str, Any]] = [
    {
        "id": "claude-desktop",
        "title": "Claude Desktop",
        "hint": (
            "Paste into ~/Library/Application Support/Claude/claude_desktop_config.json "
            "(macOS) or %APPDATA%\\Claude\\claude_desktop_config.json (Windows). "
            "Restart Claude Desktop."
        ),
        "file": "claude_desktop_config.json",
        "language": "json",
        "build": _build_mcpservers_json,
    },
    {
        "id": "cursor",
        "title": "Cursor",
        "hint": (
            "Paste into ~/.cursor/mcp.json (global) or .cursor/mcp.json in a "
            "project folder (workspace). Cursor reloads MCP servers automatically."
        ),
        "file": "~/.cursor/mcp.json",
        "language": "json",
        "build": _build_mcpservers_json,
    },
    {
        "id": "cline",
        "title": "Cline (VS Code)",
        "hint": (
            "Open the Cline panel → click the MCP icon → 'Configure MCP Servers'. "
            "Pastes into cline_mcp_settings.json; Cline reloads on save."
        ),
        "file": "cline_mcp_settings.json",
        "language": "json",
        "build": _build_mcpservers_json,
    },
    {
        "id": "continue",
        "title": "Continue (VS Code / JetBrains)",
        "hint": (
            "Append to ~/.continue/config.yaml (YAML is the supported format "
            "as of 2026; the older JSON 'experimental.modelContextProtocolServers' "
            "still works but is deprecated). Continue reloads on save."
        ),
        "file": "~/.continue/config.yaml",
        "language": "yaml",
        "build": _build_continue_yaml,
    },
    {
        "id": "windsurf",
        "title": "Windsurf (Codeium)",
        "hint": (
            "Paste into ~/.codeium/windsurf/mcp_config.json. Use "
            "Windsurf Settings → Cascade → MCP Servers → 'Refresh' to reload. "
            "Note Windsurf uses 'serverUrl' not 'url'."
        ),
        "file": "~/.codeium/windsurf/mcp_config.json",
        "language": "json",
        "build": _build_windsurf_json,
    },
    {
        "id": "vscode",
        "title": "VS Code (native MCP)",
        "hint": (
            "Create .vscode/mcp.json in your project (or settings.json under "
            "'mcp.servers'). VS Code prompts to enable the server on first use."
        ),
        "file": ".vscode/mcp.json",
        "language": "json",
        "build": _build_vscode_json,
    },
    {
        "id": "zed",
        "title": "Zed",
        "hint": (
            "Append to ~/.config/zed/settings.json under 'context_servers'. "
            "Zed reloads on save. With no Authorization header, Zed prompts "
            "for OAuth instead — pre-filling the bearer matches Augmentum's "
            "sk-aug-* key model."
        ),
        "file": "~/.config/zed/settings.json",
        "language": "json",
        "build": _build_zed_json,
    },
    {
        "id": "generic",
        "title": "Generic MCP client",
        "hint": (
            "Most other clients accept this shape. The URL and Authorization "
            "header are the only fields that matter; everything else is "
            "wrapping. If your client expects something different, only the "
            "URL + bearer token from the snippet are load-bearing."
        ),
        "file": "(client-specific)",
        "language": "json",
        "build": _build_mcpservers_json,
    },
]


# ── Entry factory ────────────────────────────────────────────────


def _make_entry(record: dict[str, Any]) -> CatalogEntry:
    client_id: str = record["id"]
    title: str = record["title"]
    hint: str = record["hint"]
    file_label: str = record["file"]
    language: str = record["language"]
    build_snippet: SnippetBuilder = record["build"]

    async def _preview(_target_id: str, _user_id: str) -> OfferPreview | None:
        return OfferPreview(
            label=f"Connect {title} to Augmentum",
            hint=f"Paste a config snippet into {file_label}.",
            details={
                "transport": "client-config",
                "file": file_label,
                "language": language,
                "scope": "user",
            },
        )

    async def _accept(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        # Allow the user/model to override the URL via extra (e.g.
        # a Tailscale hostname instead of the local origin) — but
        # default to whatever the request resolves to.
        extra = payload.get("extra") or {}
        url = ""
        if isinstance(extra, dict):
            url = str(extra.get("url") or "")
        if not url:
            url = _augmentum_mcp_url(request)
        snippet = build_snippet(url)
        return {
            "ok": True,
            "kind": "snippet",
            "client": client_id,
            "file": file_label,
            "language": language,
            "snippet": snippet,
            "hint": hint,
            "next_step": (
                f"Open {file_label}, paste this snippet, replace the "
                "bearer token with your sk-aug-* API key, and reload "
                f"{title}."
            ),
        }

    return CatalogEntry(
        kind=KIND,
        target_id=client_id,
        title=f"Connect {title}?",
        scope="user",
        build_preview=_preview,
        accept=_accept,
    )


ENTRIES: list[CatalogEntry] = [_make_entry(r) for r in _CLIENTS]


register_kind(KIND, ENTRIES)
