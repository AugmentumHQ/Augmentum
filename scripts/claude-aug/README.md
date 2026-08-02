# claude-aug — Claude Code on your local Augmentum

Runs Claude Code against your local Augmentum proxy (local models) while
leaving plain `claude` untouched (still Anthropic, still your subscription).
Fully isolated: own config dir, own sessions, own tool permissions.

## Install

Windows (PowerShell):
```powershell
.\scripts\claude-aug\install.ps1
```

Linux/macOS:
```bash
./scripts/claude-aug/install.sh
```

Prompts for your Augmentum API key (UI → Settings → API Keys) if not
passed via `-ApiKey` / `--api-key`. Re-running updates in place and keeps
existing config values. Then open a new terminal and run `claude-aug`.

## What you get

- `claude-aug [--profile deep|fast|mixed] [--model X] [--small Y]` wrapper.
  Scoped env — never leaks into the parent shell or plain `claude`.
- **Full config isolation** via `CLAUDE_CONFIG_DIR=~/.augmentum/claude-config`.
- **ATP toolkit over MCP**: all whitelisted Augmentum tools (web_search,
  web_fetch, research, python_exec, memory_recall, image_search, ...) exposed
  to Claude Code via `atp-mcp-bridge.py`, pre-allowed, no prompts.
- **No silent web-search failures**: the built-in WebSearch/WebFetch are
  Anthropic *server-side* tools and dead on a local backend — they're denied,
  and `CLAUDE.md` routing guidance points the model at the ATP equivalents.
- Health check with container auto-start, status banner, statusline showing
  `AUG | <model> @ <server>`.

## Server-side requirement (once)

Augmentum's own `.env` must map Claude Code's hardcoded `claude-*` model IDs
to your local models (the installer prints the exact lines):
```
AUGMENTUM_ANTHROPIC_ALIAS_HAIKU=<small-model>
AUGMENTUM_ANTHROPIC_ALIAS_SONNET=<small-model>
AUGMENTUM_ANTHROPIC_ALIAS_OPUS=<main-model>
AUGMENTUM_ANTHROPIC_ALIAS_DEFAULT=<main-model>
```

## Files

| File | Purpose |
|---|---|
| `install.ps1` / `install.sh` | Idempotent installer |
| `claude-aug.ps1` / `claude-aug.sh` | Shell wrappers (portable, read `claude.env`) |
| `claude.env.template` | Per-user config template |
| `atp-mcp-bridge.py` | stdio MCP server bridging `/v1/tools` (stdlib only) |
| `statusline.ps1` | Statusline for aug sessions (Windows) |
| `CLAUDE.md` | Tool-routing guidance loaded into every aug session |
