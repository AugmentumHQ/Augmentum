# Coder mode

Coder mode is a full IDE agent that works in a **real, isolated Docker
container** — not a sandbox toy. It reads and writes files, runs shell commands
and tests, drives git, controls a real browser, and can even run a dev server and
see the result. You keep long-running **workspaces** so context and state persist
between sessions.

## Getting into coder mode

Two ways:

- **The Coder surface** in the web UI — the full experience (workspace panel,
  file tree, terminal, live preview, diff review).
- **A model prefix** — send a request to any Augmentum endpoint with the model
  name prefixed `c/` (e.g. `c/llama3`) to route that turn through the coder loop.

## Workspaces & tooling profiles

Each coder workspace is its own container. When you create one, pick a **tooling
profile** for the tools it needs:

| Profile | For |
| --- | --- |
| `standard` | Fast baseline — Python, JS, Go, Rust. |
| `power` | Adds process/network inspection, modern package managers, native build/debug helpers. |
| `browser` | Power + browser automation via the shared browser service. |
| `pentest` | Power + authorized-security CLI (nmap, sqlmap, Metasploit, SecLists, …). Authorized testing only. |
| `creative` | Power + Blender headless + glTF pipeline for 3D asset creation. |

Profiles are prebaked images, so a workspace on a profile starts fast. Installs
you add later (`apt`, `pip`, `npm`) are captured and survive container recreation.

## What it can do

- **Files & code** — read, write, structured edits, patch application, grep,
  glob, symbol search, file outlines.
- **Shell & terminals** — run commands, and open *persistent* terminal sessions
  you can send input to and snapshot.
- **Tests** — auto-detects pytest / npm / go / cargo and parses results.
- **Git** — status, diff, commit, and a review-the-diff flow before you accept.
- **A browser it can see** — a Chrome-DevTools-Protocol browser: navigate,
  click, type, screenshot, read the console, extract content — so it can verify
  its own web work by *looking*.
- **Live preview** — publish a workspace port and Augmentum proxies the dev
  server back to you with auth, so you watch changes render as the agent works.
- **Bug-finder** — a dedicated pass that hunts for and verifies real bugs.
- **Sub-agents** — dispatch parallel explore/implement sub-agents for big tasks.

## Permissions

Each workspace has its own **permission policy** — what the agent may run without
asking, what needs confirmation, and what's blocked. Sensitive actions
(host-network access, destructive commands) are gated, and there's a permission
audit trail. Tune it per workspace so an experimental workspace can be looser
than one pointed at real code.

## Shaping how it works — Powers

Pin a **Power** to bias the run: a test-authoring routine, a migration-safety
checklist, a security-audit lens. Pin with `/power <id>` or from the Powers
panel, and see [Powers](powers.md) to create your own (including the "turn this
into a power" shortcut).

## Driving it from your own editor

Coder speaks the **Agent Client Protocol (ACP)** and the standard OpenAI-compatible
API, so it works both directions:

- Point **Claude Code / Cursor / Cline** at your Augmentum endpoint and prefix
  the model with `c/` to run the coder loop from your editor.
- Or use the built-in Coder surface and let it dispatch external agents.

## Tips

- Give the workspace the right profile up front — recreating to change it is
  slower than picking `power`/`browser`/`creative` when you know you'll need it.
- Use **live preview** for anything web — the agent verifying by screenshot beats
  guessing.
- For a repeated review/verification routine, capture it as a **Power** so future
  runs start with that discipline.
