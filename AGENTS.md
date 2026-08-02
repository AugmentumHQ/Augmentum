# AGENTS.md — for AI coding agents

If you're an AI coding agent (Claude Code, Cursor, Aider, Codex, Cline, …) and a
user pointed you at this repo, start here. This file orients you to **install and
run Augmentum, navigate the codebase, and answer questions about it.** For the
full development standards and architecture, read **[`CLAUDE.md`](CLAUDE.md)**
(same content whether you're Claude or another agent). For end-user guides, see
**[`docs/`](docs/README.md)**.

## What this is

Augmentum is a **self-hosted personal AI platform** — not a proxy. It serves its
own models (bundled `llama-server`), ships a multi-surface web UI, and speaks
OpenAI / Ollama / Anthropic-compatible APIs so any client can plug in. One shared
local memory + identity underlies everything. Python 3.11, SQLite (aiosqlite),
vanilla-JS UI, Docker Compose, multi-tenant, privacy-first. Large surface:
~1,400 routes across ~40 subsystems, 7 dispatch modes.

## Install & run (what to do if asked to "install and run this")

Requires **Docker** (with Compose). Two paths:

**A — Cloned repo (recommended when you already have the repo):**
```bash
./setup.sh        # interactive wizard; writes .env + .augmentum.conf (or setup.bat on Windows)
./start.sh -d     # builds locally + starts detached (start.bat on Windows)
```

**B — Pull prebuilt images (no build):**
```bash
echo "AUGMENTUM_VARIANT=cpu" > .env      # or gpu (NVIDIA)
docker compose pull && docker compose up -d
```

Then:
- Open **http://localhost:6100/ui** — the first account you register becomes the
  **admin** (registration then closes).
- Health check: the `augmentum` container is healthy when
  `curl -sf http://localhost:6100/` succeeds (that's the container healthcheck).
- Logs: `docker compose logs -f augmentum`.
- Stop: `docker compose down`. Rebuild after code changes: `./start.sh build`.

Notes for the agent doing the install:
- Binds to `127.0.0.1` by default. Only set `AUGMENTUM_BIND_HOST=0.0.0.0` if the
  user explicitly wants LAN access — and tell them to register the admin
  immediately after first launch (see the README security note).
- CPU variant works on any x86_64 host; GPU variant needs NVIDIA + adds image gen.
- On Apple Silicon the images are `linux/amd64` (QEMU by default) — see the
  Apple-Silicon note in the README for the faster route.

## Codebase map (where to look to answer questions)

Everything server-side is under `augmentum/`; the UI is under `ui/`.

| To understand… | Look in |
| --- | --- |
| **HTTP/WS API surface** | `augmentum/proxy/*_routes.py` (~110 route modules); routers registered in `augmentum/proxy/server.py` |
| **The 7 dispatch modes** | `augmentum/modes/` (`passthrough/`, `analytical/` = UARF, `narrative/`, `agentic/`, `coder/`); a classifier routes each request |
| **Coder (IDE agent)** | `augmentum/coder/` (tools in `tools.py`, containers in `containers.py`, browser, preview, permissions, ACP) |
| **Companion (autonomous)** | `augmentum/companion_runtime/` (verbs, behavior loops, native_loop, safety_floor) |
| **Model serving / backends** | `augmentum/models/` (`llama_server_manager.py`, slots A/B/C, `provider_registry.py`, `openai_compat.py`) |
| **State & DB** | `augmentum/state/` — SQLite via aiosqlite; schema evolves via numbered files in `augmentum/state/migrations/` |
| **Open tools (ATP)** | `augmentum/tools/` + `augmentum/proxy/atp_routes.py` (`/v1/tools`) |
| **Voice / avatar / vision** | `augmentum/voice/`, `augmentum/avatar/`, `augmentum/vision/` |
| **Network subsystems** | `augmentum/fabric/` (capability federation), `augmentum/connect/` + `augmentum/calling/` (calls/threads) |
| **Knowledge packs** | `augmentum/knowledge/` (ZIM / augpack hybrid retrieval) |
| **Settings** | defined in `augmentum/config.py`; registered for the UI in `augmentum/proxy/config_routes.py`; stored in the `settings` table |
| **Frontend** | `ui/scripts/` (vanilla ES modules, one area per folder); entry `ui/index.html` |
| **A specific feature** | there's almost always a package `augmentum/<name>/` — e.g. `calendar/`, `ocr/`, `powers/`, `intent/`, `game_agent/`, `dream/`, `marketplace/` |

Fast ways to locate things: grep the routes (`augmentum/proxy/*_routes.py`) for a
URL, grep `augmentum/config.py` for a setting, or check a subsystem's own
`README.md` if present (e.g. `augmentum/intent/README.md`, `augmentum/powers/README.md`).

## Making changes correctly

`CLAUDE.md` is the source of truth for conventions. The essentials:

- **Multi-tenant:** every user-scoped table has a `user_id` column; every CRUD
  function takes `*, user_id: str = ""`; routes pass
  `request.scope.get("user").id`. New tables/features must follow this.
- **Persistence is server-side** (SQLite); a new table needs a migration file in
  `augmentum/state/migrations/`. localStorage is a cache only.
- **New route file →** register its router in `server.py`. **New setting →** add
  to `config.py` and `config_routes.py`.
- **Verify wide:** the surface is highly interconnected — a change in one place
  routinely affects others; check the whole relevant path, not one file.
- **Fix the class, not the symptom**, and **never auto-select** on the user's
  behalf (models/providers/voices) — surface the choice.
- Python is ruff-compliant with `from __future__ import annotations`; JS is
  framework-free ES modules.

Run tests with `python -m pytest tests/ -x` (from a venv, `pip install -e ".[dev]"`),
lint with `ruff check augmentum/`.

## Answering user questions about Augmentum

- **"What can it do?"** → the README's "What it can do" section + `docs/README.md`.
- **"How do I use X?"** → the matching guide in `docs/` (coder, companion, voice,
  discover, powers, knowledge packs, connect…).
- **"How does X work internally?"** → the subsystem package under `augmentum/<X>/`
  and the relevant "Architecture Patterns" section in `CLAUDE.md`.
- **Status/maturity:** this is a single-developer, self-tested beta that hasn't
  had external peer review — be honest about that; don't imply production-grade.
