---
name: Dependency Doctor
description: >
  Dependency, package-manager, environment, and installation triage for workspaces.
kind: guidance
activation_policy: controller
activation_windows:
  - pre_plan
  - verify_failed
modes:
  - coder
triggers:
  - dependency
  - install
  - package manager
  - module not found
  - lockfile
  - environment
preferred_tools:
  - file_read
  - shell_read
  - shell_exec
  - service_logs
verification_recipe:
  - Identify the manifest and lockfile before installing.
  - Prefer the workspace's package manager and existing scripts.
  - Verify with the narrowest build, import, or test command.
memory_writes:
  - category: runtime
    key: package_manager
  - category: command
    key: install_command
success_criteria:
  - Package-manager choice is explained from workspace evidence.
  - Install or environment changes are verified or explicitly marked pending.
tags:
  - dependencies
  - environment
  - setup
---

# Dependency Doctor

Use this Power when the task is blocked by missing packages, broken
lockfiles, environment mismatch, or setup uncertainty. The default
failure mode is to install the wrong package, with the wrong manager,
into the wrong scope — be patient and read the manifest first.

## Workflow

1. **Identify the manifest first** — `pyproject.toml`, `requirements.txt`,
   `package.json`, `go.mod`, `Cargo.toml`, etc. Whatever's there governs.
2. **Identify the package manager** — read the lockfile name
   (`uv.lock`, `poetry.lock`, `package-lock.json`, `pnpm-lock.yaml`,
   `yarn.lock`, `bun.lockb`, `Cargo.lock`). The lockfile is more
   reliable than the user's habits.
3. **Use the workspace's existing scripts** — if `package.json` has a
   `"scripts"` block, prefer `npm run install:dev` over a raw install.
4. **Install with the right manager** — never `pip install` in a
   `uv`-managed repo; never `npm install` in a `pnpm` repo. Lockfile
   drift caused by wrong-manager installs is hard to detect and
   poisons future builds.
5. **Verify with the narrowest command** — `python -c "import x"`,
   `node -e "require('x')"`, `go build ./pkg/...`. Don't run the full
   test suite to verify one install.
6. **Record the decision** via `observe`: `category="env"` with the
   exact install command + manager so future sessions don't redo this.

## Common gotchas

- **Multiple Pythons**: `python` may point at system Python; `python3`
  or the venv's `python` may be different. Use the venv-resolved path
  the workspace profile names.
- **Devcontainer vs host**: in our coder mode, `/workspace` is a
  container — installs inside the container don't affect the host.
  Always confirm you're inside the container (`env_info` /
  `container_info`).
- **Optional extras**: `pip install foo` and `pip install foo[bar]`
  install different things. Check the manifest for the extras the
  project actually needs.
- **Native deps**: a Python or Node package may need a system lib
  (gcc, libpq, libffi). If `pip install` fails compiling, the fix is
  often `apt install <lib>-dev` first.

## Guardrails

- Never `pip install --upgrade` package versions casually — the
  lockfile pins for a reason.
- Never delete a lockfile to "regenerate" without explicit user
  consent — that's a one-way change.
- Never globally install in a venv-managed repo; respect the venv.
- Never assume a global tool exists — `which jq`, `which docker`
  before relying on them.

## Good outputs

- "Identified `uv.lock` + `pyproject.toml`. Installed `httpx` via
  `uv add httpx`. Verified with `python -c 'import httpx'`. Recorded
  install_command observation."
- "Build failed compiling psycopg2-binary. Installed `libpq-dev`
  first, then `pip install -r requirements.txt`. Pinned the libpq
  requirement to a constraint observation."
- "Skipped install — `package.json` lists `react` but it's already in
  `node_modules` per `npm ls react`. No action needed."

