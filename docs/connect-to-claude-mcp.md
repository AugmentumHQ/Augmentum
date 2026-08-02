# Connect your Augmentum box to Claude (and other MCP clients)

Augmentum runs a built-in **MCP server** — so Claude Desktop, Claude Code,
Cursor, or any MCP-compatible agent can use *your* box's tools and **your
private memory** directly. Point Claude at your box and it can search the web,
recall and store facts in your personal memory, generate images, work your
media library, and more — all on your hardware, scoped to you.

This is the differentiator: most "local AI" exposes a chat endpoint. Augmentum
also exposes its **cross-modal tools + private memory** over the open MCP
standard, so the agent you already use gets access to your box.

## What you get

Once connected, the client sees Augmentum's tools, including:

- **`memory_recall`** / **`memory_store`** — search and write your personal
  semantic memory (scoped to your user; never shared across tenants).
- **Web search**, **Python execution**, **math verification**, file/text tools.
- **Image generation**, **media** controls, **browse**, and (where available)
  bug-finder tools — whatever is registered on your box.
- Your **character cards** and **knowledge packs** as MCP *resources*, and your
  **prompt presets / reasoning flows** as MCP *prompts*.

## Prerequisites

1. **MCP server enabled** — on by default (`mcp_enabled`). Confirm via
   `GET /api/capabilities` → `"mcp_enabled": true`. The endpoint is also listed
   there under `endpoints.mcp`.
2. **An API key** — create one in **Settings → API Keys** (`sk-aug-…`). The MCP
   endpoint is auth-gated; the client authenticates with this key.
3. **Your endpoint URL** — `https://YOUR_HOST:PORT/mcp/` (note the trailing
   slash). Transport is **Streamable HTTP**; auth is the
   `Authorization: Bearer sk-aug-…` header.

> Replace `YOUR_HOST:PORT` with your box (e.g. `https://192.168.1.50:6443`) and
> `sk-aug-XXXX` with your key throughout.

## Claude Code (CLI)

```bash
claude mcp add --transport http augmentum https://YOUR_HOST:PORT/mcp/ \
  --header "Authorization: Bearer sk-aug-XXXX"
```

- `--transport http` is correct (Streamable HTTP).
- Add `--scope user` to make it available across all your projects (writes to
  `~/.claude.json`); the default scope is local/personal.
- Manage it: `claude mcp list`, `claude mcp get augmentum`, `claude mcp remove augmentum`.

## Claude Desktop

Edit `claude_desktop_config.json`:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Recent versions (native remote HTTP):**

```json
{
  "mcpServers": {
    "augmentum": {
      "type": "http",
      "url": "https://YOUR_HOST:PORT/mcp/",
      "headers": { "Authorization": "Bearer sk-aug-XXXX" }
    }
  }
}
```

**Older versions (via the `mcp-remote` bridge):**

```json
{
  "mcpServers": {
    "augmentum": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "--url", "https://YOUR_HOST:PORT/mcp/",
               "--header", "Authorization: Bearer sk-aug-XXXX"]
    }
  }
}
```

Restart Claude Desktop after editing.

## Cursor / other MCP clients

Any client that supports **Streamable HTTP** MCP servers:

```json
{
  "mcpServers": {
    "augmentum": {
      "type": "streamable-http",
      "url": "https://YOUR_HOST:PORT/mcp/",
      "headers": { "Authorization": "Bearer sk-aug-XXXX" }
    }
  }
}
```

## Security notes

- The `/mcp/` endpoint is **not public** — it's gated by Augmentum's auth
  middleware. A request with no valid `sk-aug-…` key never reaches a tool.
- **Multi-tenant safe:** `memory_recall`/`memory_store` read the user from the
  authenticated request on every call, so each key only ever touches its own
  user's memory. Memory tools refuse to run for the anonymous tenant.
- Treat the API key like a password — it grants tool + memory access to your
  box. Revoke it anytime in **Settings → API Keys**.

## Troubleshooting

- **404 / "not found":** make sure the URL ends in **`/mcp/`** (trailing slash).
  `/mcp` without the slash redirects; some clients don't follow it.
- **TLS / certificate errors (common on a LAN box):** if your box uses a
  self-signed certificate over HTTPS, MCP clients may reject the connection.
  Either install a trusted cert (e.g. via a reverse proxy / Let's Encrypt on a
  real hostname), trust the self-signed cert on the client machine, or — for
  `mcp-remote` — consult its flags for self-signed handling. Plain `http://`
  on a trusted LAN also works if your deployment allows it.
- **401 Unauthorized:** the key is missing, wrong, or revoked. Re-check the
  `Authorization: Bearer sk-aug-…` header and the key in Settings → API Keys.
- **Confirm the surface:** `GET https://YOUR_HOST:PORT/api/capabilities` shows
  `mcp_enabled`, the `endpoints.mcp` path, and everything else your box offers.
