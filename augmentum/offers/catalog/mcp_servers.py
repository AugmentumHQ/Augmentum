"""MCP server install catalog.

Two tiers of entries:

* **Tier 1 — verified vendor remote (Streamable HTTP).** The vendor
  hosts the server; we just paste their URL. OAuth lives between the
  user and the vendor; Augmentum never sees the credentials.
  Sources (2026-06-02 sweep):
  - Linear     https://mcp.linear.app/mcp
  - Notion     https://mcp.notion.com/mcp
  - Stripe     https://mcp.stripe.com
  - Atlassian  https://mcp.atlassian.com/v1/mcp        (Jira + Confluence)
  - Cloudflare https://api.mcp.cloudflare.com/mcp      (~2,500 API endpoints)
  - Sentry     https://mcp.sentry.dev/mcp
  - Intercom   https://mcp.intercom.com/mcp
  - Monday     https://mcp.monday.com/mcp
  - HF Hub     https://huggingface.co/mcp
  - GitHub     https://api.githubcopilot.com/mcp/      (GH Copilot's MCP)

* **Tier 2 — Anthropic reference (stdio via npx/uvx).** Subprocess
  servers maintained by the MCP steering group. Install command runs
  through MCPClientManager.connect_stdio. The 7 currently-maintained:
  Everything (test-only, skipped), Fetch, Filesystem, Git, Memory,
  SequentialThinking, Time.

The Gmail entry is intentionally a user-bring-your-own placeholder —
there's no canonical hosted Gmail MCP as of 2026-06-02. The accept
handler stubs the row so the user can finish OAuth in Settings →
Providers → MCP.

All entries are admin-scoped because MCP install grants tool access
to anything the chat LLM picks up; this matches the auth posture of
``/v1/mcp/connect``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from augmentum.config import settings
from augmentum.offers.catalog.base import (
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request


log = get_logger(__name__)


KIND: str = "mcp_server"


# ── Persistence helpers ──────────────────────────────────────────


def _read_persisted_servers() -> list[dict[str, Any]]:
    raw = (getattr(settings, "mcp_servers", "") or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, dict) and s.get("name")]
    except (json.JSONDecodeError, ValueError, TypeError):
        log.warning("mcp_servers_persisted_invalid_json")
    return []


async def _persist_servers(
    request: "Request", servers: list[dict[str, Any]],
) -> None:
    """Match the persistence shape used by ``/v1/mcp/connect``."""

    store = getattr(request.app.state, "settings_store", None)
    serialized = json.dumps(servers)
    if store is not None:
        try:
            await store.set("mcp_servers", serialized)
        except Exception:
            log.warning("mcp_servers_persist_failed", exc_info=True)
    # Live update so subsequent /v1/mcp/servers reads see the new row
    # without waiting for a process restart.
    object.__setattr__(settings, "mcp_servers", serialized)


def _server_already_present(name: str) -> bool:
    for entry in _read_persisted_servers():
        if entry.get("name") == name:
            return True
    return False


# ── Generic install handlers (shared by every entry) ─────────────


async def _install_http_server(
    name: str, url: str, request: "Request",
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Persist + live-connect a Streamable HTTP MCP server.

    Mirrors the path ``/v1/mcp/connect`` takes for HTTP: append to
    ``mcp_servers`` settings, then ``client.connect_http`` + register
    tools so the new server is reachable in this process without
    waiting for a restart. Errors during live-connect are non-fatal:
    the persisted row guarantees the server is wired on next start.
    """

    if _server_already_present(name):
        return {
            "ok": True,
            "already_installed": True,
            "name": name,
            "next_step": "Open Settings → Providers → MCP to manage.",
        }

    new_row: dict[str, Any] = {"name": name, "url": url}
    if headers:
        new_row["headers"] = dict(headers)
    existing = _read_persisted_servers()
    existing.append(new_row)
    await _persist_servers(request, existing)

    # Live connect — best effort. Persistence is the source of truth.
    live_tools: list[str] = []
    live_error: str = ""
    client = getattr(request.app.state, "mcp_client", None)
    if client is not None:
        try:
            await client.connect_http(name, url, headers=headers)
            from augmentum.mcp.bridge import register_mcp_tools

            live_tools = register_mcp_tools(
                client, name, request.app.state.tool_registry,
            )
        except Exception as exc:
            live_error = str(exc)[:200]
            log.warning(
                "mcp_offer_live_connect_failed",
                server=name, error=live_error,
            )

    log.info("offer_mcp_server_installed", target_id=name, transport="http")
    return {
        "ok": True,
        "name": name,
        "transport": "http",
        "tools_connected": live_tools,
        "live_error": live_error,
        "next_step": (
            "Complete OAuth at the vendor's URL if prompted, "
            "then the tools are usable."
            if not live_error else
            "Settings → Providers → MCP — finish auth + retry."
        ),
    }


async def _install_stdio_server(
    name: str, command: str, request: "Request",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Persist + live-connect a stdio MCP server (npx / uvx / etc.)."""

    if _server_already_present(name):
        return {
            "ok": True,
            "already_installed": True,
            "name": name,
            "next_step": "Open Settings → Providers → MCP to manage.",
        }

    new_row: dict[str, Any] = {"name": name, "command": command}
    if args:
        new_row["args"] = list(args)
    if env:
        new_row["env"] = dict(env)
    existing = _read_persisted_servers()
    existing.append(new_row)
    await _persist_servers(request, existing)

    live_tools: list[str] = []
    live_error: str = ""
    client = getattr(request.app.state, "mcp_client", None)
    if client is not None:
        try:
            await client.connect_stdio(name, command, args=args, env=env)
            from augmentum.mcp.bridge import register_mcp_tools

            live_tools = register_mcp_tools(
                client, name, request.app.state.tool_registry,
            )
        except Exception as exc:
            live_error = str(exc)[:200]
            log.warning(
                "mcp_offer_live_connect_failed",
                server=name, error=live_error,
            )

    log.info("offer_mcp_server_installed", target_id=name, transport="stdio")
    return {
        "ok": True,
        "name": name,
        "transport": "stdio",
        "tools_connected": live_tools,
        "live_error": live_error,
        "next_step": (
            "First connect may take 5-15s while the package downloads."
            if not live_error else
            "Settings → Providers → MCP — check that npx/uvx is on PATH."
        ),
    }


# ── Per-entry preview + accept factory ───────────────────────────


def _http_entry(
    target_id: str,
    title: str,
    url: str,
    hint: str,
    icon: str = "",
) -> CatalogEntry:
    """Build a CatalogEntry for a vendor remote-HTTP MCP server.

    The factory keeps each entry below to a single readable line at
    the call site; everything offer-mechanics-related is in shared
    helpers.
    """

    async def _preview(_target_id: str, _user_id: str) -> OfferPreview | None:
        if _server_already_present(target_id):
            return None
        return OfferPreview(
            label=f"{title} (Streamable HTTP)",
            hint=hint,
            details={
                "transport": "http",
                "url": url,
                "auth": "OAuth at vendor",
                "scope": "admin",
            },
        )

    async def _accept(payload: dict[str, Any], request: "Request") -> dict[str, Any]:
        # Allow the model to forward user-supplied custom headers via
        # ``extra.headers`` (e.g. a pre-issued bearer token), but URL
        # itself comes from the curated catalog — the model can't
        # redirect a vendor offer to a third-party URL.
        extra = payload.get("extra") or {}
        headers_in = extra.get("headers") if isinstance(extra, dict) else None
        headers: dict[str, str] | None = None
        if isinstance(headers_in, dict):
            headers = {str(k): str(v) for k, v in headers_in.items()}
        return await _install_http_server(target_id, url, request, headers=headers)

    return CatalogEntry(
        kind=KIND,
        target_id=target_id,
        title=f"Install {title} MCP?",
        scope="admin",
        build_preview=_preview,
        accept=_accept,
        icon=icon,
    )


def _stdio_entry(
    target_id: str,
    title: str,
    command: str,
    args: list[str],
    hint: str,
    icon: str = "",
    requires_arg_extra: str = "",
) -> CatalogEntry:
    """Build a CatalogEntry for a stdio (npx / uvx) MCP server.

    ``requires_arg_extra`` names an ``extra`` key the user must
    provide for the install to work (e.g. ``"path"`` for filesystem,
    ``"repository"`` for git). When set and missing, the offer still
    installs but ``next_step`` tells the user to add the value in
    Settings.
    """

    async def _preview(_target_id: str, _user_id: str) -> OfferPreview | None:
        if _server_already_present(target_id):
            return None
        return OfferPreview(
            label=f"{title} (stdio)",
            hint=hint,
            details={
                "transport": "stdio",
                "command": command,
                "args_template": list(args),
                "scope": "admin",
                "requires": requires_arg_extra or "",
            },
        )

    async def _accept(payload: dict[str, Any], request: "Request") -> dict[str, Any]:
        # If the entry requires an argument (path, repository, etc.)
        # pull it from extras; otherwise the entry connects with just
        # its template args. The args list is rebuilt from the
        # template so a missing extra produces a no-arg server (which
        # the user can fix later via Settings) rather than crashing.
        extra = payload.get("extra") or {}
        runtime_args = list(args)
        if requires_arg_extra and isinstance(extra, dict):
            user_val = extra.get(requires_arg_extra)
            if isinstance(user_val, str) and user_val:
                runtime_args.append(user_val)
        return await _install_stdio_server(
            target_id, command, request, args=runtime_args,
        )

    return CatalogEntry(
        kind=KIND,
        target_id=target_id,
        title=f"Install {title} MCP?",
        scope="admin",
        build_preview=_preview,
        accept=_accept,
        icon=icon,
    )


# ── Gmail (Phase 1 reference — user must bring their own URL) ────


async def _gmail_preview(target_id: str, user_id: str) -> OfferPreview | None:
    if _server_already_present("gmail"):
        return None
    return OfferPreview(
        label="Gmail (bring-your-own server URL)",
        hint=(
            "No canonical hosted Gmail MCP as of 2026. Creates a stub "
            "entry; provide your own server URL + OAuth header in "
            "Settings → MCP."
        ),
        details={"transport": "http", "scope": "admin", "byo_url": True},
    )


async def _gmail_accept(
    payload: dict[str, Any], request: "Request",
) -> dict[str, Any]:
    if _server_already_present("gmail"):
        return {
            "ok": True,
            "already_installed": True,
            "next_step": "Open Settings → Providers → MCP to manage.",
        }

    extra = payload.get("extra") or {}
    url = str(extra.get("url") or "") if isinstance(extra, dict) else ""
    headers_in = extra.get("headers") if isinstance(extra, dict) else None
    headers: dict[str, str] = {}
    if isinstance(headers_in, dict):
        for k, v in headers_in.items():
            headers[str(k)] = str(v)

    if url:
        return await _install_http_server("gmail", url, request, headers=headers)

    # Stub — write {name: gmail, url: ""} so the row appears in
    # Settings for the user to fill in.
    new_row: dict[str, Any] = {"name": "gmail", "url": "", "headers": headers}
    existing = _read_persisted_servers()
    existing.append(new_row)
    await _persist_servers(request, existing)
    log.info("offer_mcp_server_stub_installed", target_id="gmail")
    return {
        "ok": True,
        "server_added": "gmail",
        "needs_configuration": True,
        "next_step": (
            "Open Settings → Providers → MCP and provide the Gmail "
            "server URL + OAuth header."
        ),
    }


# ── Catalog ──────────────────────────────────────────────────────


ENTRIES: list[CatalogEntry] = [
    # ─ Vendor remote (Streamable HTTP, OAuth at vendor) ─
    _http_entry(
        target_id="linear",
        title="Linear",
        url="https://mcp.linear.app/mcp",
        hint="Issues, projects, cycles, comments — OAuth.",
    ),
    _http_entry(
        target_id="notion",
        title="Notion",
        url="https://mcp.notion.com/mcp",
        hint="Pages, databases, search — OAuth. Local server is being sunset.",
    ),
    _http_entry(
        target_id="stripe",
        title="Stripe",
        url="https://mcp.stripe.com",
        hint="Read-only payments / customers / invoices — OAuth. Never paste the full secret key.",
    ),
    _http_entry(
        target_id="atlassian",
        title="Atlassian (Jira + Confluence)",
        url="https://mcp.atlassian.com/v1/mcp",
        hint="JQL search, ticket CRUD, Confluence pages — OAuth. (SSE deprecated 2026-06-30.)",
    ),
    _http_entry(
        target_id="cloudflare",
        title="Cloudflare",
        url="https://api.mcp.cloudflare.com/mcp",
        hint="DNS, Workers, R2, KV, Zero Trust — ~2,500 API endpoints via two tools.",
    ),
    _http_entry(
        target_id="sentry",
        title="Sentry",
        url="https://mcp.sentry.dev/mcp",
        hint="Issues, events, releases — OAuth at sentry.dev.",
    ),
    _http_entry(
        target_id="intercom",
        title="Intercom",
        url="https://mcp.intercom.com/mcp",
        hint="Customer conversations, contacts, tickets — OAuth.",
    ),
    _http_entry(
        target_id="monday",
        title="Monday.com",
        url="https://mcp.monday.com/mcp",
        hint="Boards, items, columns, updates — OAuth.",
    ),
    _http_entry(
        target_id="hugging_face",
        title="Hugging Face Hub",
        url="https://huggingface.co/mcp",
        hint="Models, datasets, spaces — read-only browse + search.",
    ),
    _http_entry(
        target_id="github_copilot",
        title="GitHub (Copilot MCP)",
        url="https://api.githubcopilot.com/mcp/",
        hint="Repos, PRs, issues, code search — GitHub Copilot's official MCP. OAuth.",
    ),

    # ─ Anthropic reference (stdio via npx / uvx) ─
    _stdio_entry(
        target_id="fetch",
        title="Fetch",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-fetch"],
        hint="LLM-friendly web fetch with HTML→markdown. Reference server.",
    ),
    _stdio_entry(
        target_id="filesystem",
        title="Filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem"],
        hint="Read/write inside one allowed dir. Pass the directory in extra.path.",
        requires_arg_extra="path",
    ),
    _stdio_entry(
        target_id="git",
        title="Git",
        command="uvx",
        args=["mcp-server-git", "--repository"],
        hint="Read / search / manipulate git repos. Pass the repo dir in extra.repository.",
        requires_arg_extra="repository",
    ),
    _stdio_entry(
        target_id="memory",
        title="Memory (knowledge graph)",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
        hint="Persistent KG memory for the model. Separate from Augmentum's own memory store.",
    ),
    _stdio_entry(
        target_id="sequential_thinking",
        title="Sequential Thinking",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequentialthinking"],
        hint="Reflective step-by-step problem-solving tool. Useful for weaker models.",
    ),
    _stdio_entry(
        target_id="time",
        title="Time / Timezone",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-time"],
        hint="Timezone conversion + scheduling helpers.",
    ),

    # ─ User-supplied URL fallback (Phase 1 reference) ─
    CatalogEntry(
        kind=KIND,
        target_id="gmail",
        title="Install Gmail MCP?",
        scope="admin",
        build_preview=_gmail_preview,
        accept=_gmail_accept,
        icon="mail",
    ),
]


register_kind(KIND, ENTRIES)
