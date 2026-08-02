# Connect external MCP servers *into* Augmentum

This is the reverse of [Connect to Claude via MCP](connect-to-claude-mcp.md):
here Augmentum is the MCP **client**, pulling another MCP server's tools into
your box so your chat, companion, coder, and agentic modes can use them — e.g.
a GitHub server, a database server, a third-party search server.

Once connected, the external server's tools are registered in Augmentum's tool
registry and behave like built-in tools everywhere tools are available.

## The one rule that decides everything

Augmentum runs in a `python:3.11-slim` container (Python + `uv`, **no Node**).
That splits MCP servers into two cases:

| Server transport | How to add | Works in the box? |
|---|---|---|
| **Remote HTTP / Streamable-HTTP** (`url`) | UI, API, or config | ✅ Yes — the reliable path |
| **stdio subprocess** (`command` + `args`, e.g. `npx …`, `uvx …`) | env var only | ⚠️ Only if the command's runtime is in the container |

- **HTTP servers are the recommended path** — they connect over the network, so
  the container runtime doesn't matter.
- **stdio `npx`-based servers** (the most common kind — `@modelcontextprotocol/
  server-github`, `server-filesystem`, etc.) **won't run** as-is, because Node
  isn't in the Augmentum image. `uvx`/`python` stdio servers *may* work (uv is
  present). To use a stdio-only server reliably, wrap it as HTTP — see below.

All of this is **admin-only**: an MCP server exposes its tools to every tenant
on the box, so adding one is an install-wide decision.

## Easiest: add a remote HTTP server in the UI

1. **Settings → MCP** → the **Connect New Server** form.
2. Enter a **name** (e.g. `github`), the server **URL** (e.g.
   `https://my-mcp-host/mcp/`), and — if the server needs auth — paste the
   **Authorization header** value (e.g. `Bearer sk-…`) into the optional field.
3. Click **Connect**. The URL is SSRF-validated, the server's tools register,
   and a confirmation shows how many tools you just made available **to everyone
   on this box**. The server appears in **Connected MCP Servers** with a
   **Disconnect** button, and all exposed tools are listed (and filterable)
   under **All MCP Tools**.

The connection **persists across restarts** (saved to the `mcp_servers` config).

> Adding a server is **admin-only** and **install-wide** — its tools become
> available to every user on the box. Only connect servers you trust, and use
> Disconnect to remove one. The optional auth field is masked.

## Via the API (HTTP servers, headers supported)

For servers that need an auth header (the UI form covers name + URL; the API
covers headers), `POST /api/mcp/connect` (admin):

```bash
curl https://YOUR_HOST:PORT/api/mcp/connect \
  -H "Authorization: Bearer sk-aug-XXXX" \
  -H "Content-Type: application/json" \
  -d '{
        "name": "github",
        "url": "https://my-mcp-host/mcp/",
        "headers": { "Authorization": "Bearer <that-server-token>" }
      }'
```

- The URL is **SSRF-checked** before connecting (blocks internal/metadata IPs).
- stdio (`command`) bodies are **rejected** by this API for security — use the
  env var below.
- Other endpoints: `GET /api/mcp/servers` (list), `DELETE /api/mcp/servers/{name}`
  (disconnect), `GET /api/mcp/tools` (all tools from connected servers).

## stdio servers (env var only)

For security, subprocess servers can only be set via the
`AUGMENTUM_MCP_SERVERS` environment variable — a JSON array. Remember the
runtime must exist in the container (Python/`uvx` ok; `npx`/Node not, unless you
add it to the image):

```bash
# A Python-based stdio server (uv is in the image):
AUGMENTUM_MCP_SERVERS='[
  {"name": "my-tool", "command": "uvx", "args": ["some-python-mcp-server"]}
]'
```

HTTP servers can go here too (and survive without the UI):

```bash
AUGMENTUM_MCP_SERVERS='[
  {"name": "github", "url": "https://my-mcp-host/mcp/",
   "headers": {"Authorization": "Bearer <token>"}}
]'
```

Set it in your `.env` / compose environment and restart.

## Using an `npx`/stdio-only server anyway (the HTTP wrapper trick)

If the server you want only ships as stdio + `npx`, don't fight the container —
**run it as a small sidecar that exposes HTTP**, then point Augmentum at that
URL. Tools like `mcp-proxy` / `supergateway` wrap a stdio MCP server as an
HTTP/SSE endpoint. Run one in its own Node container (or on the host), e.g. a
compose service that runs `npx <stdio-server>` behind an HTTP gateway, then add
its URL via the UI/API above. This keeps the Augmentum image lean and gives you
any MCP server over the reliable HTTP path.

## Where the tools show up

After connecting, the external tools are in Augmentum's tool registry, so
they're available wherever tools run: passthrough chat (with tools enabled),
analytical, agentic, coder, and the companion's tool loop. Confirm what's live
with `GET /api/mcp/tools` or the **Connected MCP Servers** list in Settings.
