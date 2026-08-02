---
name: run-augmentum
description: >
  Build, run, smoke-test, and screenshot Augmentum (the local FastAPI/Docker
  intelligence-layer proxy). Use when asked to start Augmentum, bring the
  stack up, verify it's healthy, take a screenshot of the UI, or interact
  with the running app from the agent side. Triggers on "run augmentum",
  "start the stack", "smoke test", "is augmentum up", "screenshot the UI",
  "boot the proxy", "docker compose up", "check augmentum health".
---

Augmentum is a FastAPI proxy delivered as a Docker Compose stack rooted at
this repo. The agent driver is `.claude/skills/run-augmentum/smoke.sh` — it
brings the stack up (if not already), probes the unauthenticated HTTP
surface, and headless-screenshots `/ui/` via Chrome. All paths below are
relative to the repo root.

## Prerequisites

- **Docker** with `compose` v2. Docker Desktop on Windows; Docker Engine
  on Linux. Verified with `Docker 29.x` + `compose v5.0.2`.
  ```bash
  docker version --format '{{.Server.Version}}'
  docker compose version | head -1
  ```
- **Bash** (git-bash on Windows is fine — that is the verified host).
- **Chrome / Chromium** for the screenshot step. Optional; the smoke
  script auto-detects and skips with a warning if absent.
  Verified path on Windows: `/c/Program Files/Google/Chrome/Application/chrome.exe`.
- `.augmentum.conf` must exist at the repo root (lists the compose files
  the stack uses — created by `setup.sh` / `setup.bat`). If it's missing,
  run setup once before anything else.

## Setup

If `.augmentum.conf` does not exist, run setup. It writes both that file
and `.env`:

```bash
./setup.sh        # Linux / macOS / git-bash
# or: setup.bat   # Windows cmd
```

Setup is one-shot and idempotent. `.env` is gitignored.

## Build

Only needed when a `Dockerfile*` or vendored llama-server pin changes. The
default `start.*` path re-uses existing images.

```bash
./start.sh build     # rebuild + start (foreground)
# or: start.bat build
```

First-time builds pull `augmentum-llama-server` from GHCR (~30s) if a
binary is published for the pinned `LLAMA_SERVER_VERSION`; otherwise it
falls back to a local CUDA compile that takes **30–50 minutes**.

## Run (agent path)

The driver does boot + smoke + screenshot in one shot:

```bash
./.claude/skills/run-augmentum/smoke.sh
```

What it does, in order:

1. Checks `docker inspect augmentum-augmentum-1` health. If not `healthy`,
   invokes `./start.sh -d` and polls for up to ~120s.
2. Probes three unauthenticated endpoints (any failure → exit 1):
   - `GET /api/version` — expects `{"version":"…"}`
   - `GET /api/auth/status` — expects `{"setup_required":…,…}`
   - `GET /` — expects `Ollama is running` (Ollama-compat shim)
3. Screenshots `http://localhost:6100/ui/` via headless Chrome to
   `tmp/run-skill/ui-root.png` (1280×800, 8s virtual-time budget).

Artifacts:

| path | contents |
|---|---|
| `/tmp/augmentum-smoke.log` | timestamped probe transcript |
| `tmp/run-skill/ui-root.png` | UI screenshot (Sign-In page on a fresh stack) |

Flags:

| flag | effect |
|---|---|
| `--no-boot` | skip the boot+wait; just probe what's running |
| `--no-shot` | skip the Chrome screenshot |

Verified single-command run from this session (existing healthy stack):

```text
ok:   /api/version -> 200   body: {"version":"0.1.0"}
ok:   /api/auth/status -> 200   body: {"setup_required":false,"authenticated":false,"user":null}
ok:   / -> 200   body: Ollama is running
ok:   screenshot -> .../tmp/run-skill/ui-root.png (14184 bytes)
smoke PASS
```

### Authenticated probes (out of smoke scope)

`/api/health` returns the deep backend report but is gated by auth — it
responds `401 {"error":"Unauthorized"}` to an anonymous request. To hit
it you need a session cookie or a long-lived API key from the UI's
Settings → Augmentum API Keys panel; pass as `Authorization: Bearer <key>`.
The smoke deliberately does not depend on this so it works on a fresh
post-setup stack with no user signed in yet.

## Run (human path)

```bash
./start.sh           # foreground; Ctrl-C to stop
./start.sh -d        # detached
./start.sh down      # stop the stack
./start.sh logs -f   # tail logs
./start.sh ps        # what's running
```

`start.bat` is the equivalent for Windows cmd. Both read `.augmentum.conf`
to build the `-f compose.*.yaml` flag list.

Surfaces:

- `http://localhost:6100` — direct uvicorn (what the smoke uses)
- `https://localhost:6443` — Caddy-fronted TLS (self-signed by default;
  use `curl -k`)

## Test

```bash
python -m pytest tests/ -x
```

The audit/health system has its own scanner — orthogonal to the running
stack but worth running before claiming a session done:

```bash
python .claude/skills/augmentum-dev/scripts/audit.py --quiet
```

## Gotchas

- **`/api/health` is auth-gated.** Treat the Docker `HEALTHCHECK` status
  (`docker inspect ... --format '{{.State.Health.Status}}'`) and
  `/api/version` as the agent's "is it up" signals. Reaching for
  `/api/health` from a script will get a 401 and look like a broken stack.
- **`/` returns `Ollama is running`, not the UI.** This is the
  Ollama-compatibility shim (so an Ollama client can hit the root path).
  The UI is at **`/ui/`** (trailing slash matters — bare `/ui` is a 307
  redirect). Don't screenshot `/`; you'll get plain text.
- **Chrome screenshot paths must be Windows-style on Windows hosts.**
  `chrome.exe --screenshot=/some/posix/path.png` fails with
  `Access is denied (0x5)` and silently exits. The smoke uses `cygpath
  -w` to translate; if you call chrome directly, pass `C:\…\file.png`.
- **First boot is slow if `augmentum-llama-server` isn't cached.**
  `start.sh` tries `docker pull ghcr.io/augmentumhq/augmentum-llama-server:<ver>`
  first (~30s) and falls back to a local CUDA build that runs **30–50
  minutes**. Skip this branch entirely by using `compose.yaml` only
  (remove `compose.dev.yaml` from `.augmentum.conf`) — production runs
  use the prebuilt `augmentum` image from GHCR.
- **HTTPS uses a self-signed cert.** `curl -k` for 6443. The cert SAN
  list is auto-populated from LAN + Tailscale interfaces by `start.*`;
  override with `AUGMENTUM_TLS_EXTRA_SANS=IP:1.2.3.4,...` in `.env`.
- **`augmentum-ws-*` containers are coder-mode workspaces, not core
  stack.** They survive `start.sh down` of the main stack and have
  their own lifecycle (created/torn down by the coder route). Don't
  panic if you see a dozen of them in `docker ps`.

## Troubleshooting

- **`{"error":"Unauthorized"}` from every probe** — you're hitting an
  auth-gated path. Use `/api/version`, `/api/auth/status`, or `/` for
  unauthenticated smoke.
- **`405 Method Not Allowed` with `Allow: GET`** — you used `curl -I`
  (HEAD) against a GET-only route. Use `curl -sI` only for header
  inspection; for the smoke use `curl -sf` (GET).
- **`augmentum-augmentum-1 status=absent`** — the stack isn't up. Run
  `./start.sh -d` and re-check `docker ps`. If it never appears,
  `./start.sh logs augmentum` will show the boot failure (most often
  a missing migration or a port conflict on 6100).
- **Screenshot file is 0 bytes** — Chrome silently exited. Re-run with
  the screenshot path as a Windows path (`C:\…`) and watch stderr for
  the access-denied line. Verified workaround is in `smoke.sh`.
- **`No configuration found. Run setup first`** — `.augmentum.conf` is
  missing. Run `./setup.sh` once; it's idempotent.
- **Port 6100 already bound** — usually a stale `augmentum-augmentum-1`
  from a crashed run. `docker rm -f augmentum-augmentum-1` then
  `./start.sh -d`.
