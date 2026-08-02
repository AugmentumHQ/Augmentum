# Contributing to Augmentum

Welcome. This guide is the practical "how do I actually do X"
companion to:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 5-minute request-flow + substrate primer
- [`docs/integration-weave.md`](docs/integration-weave.md) — the cross-modal view (what the substrates share)
- [`docs/subsystems.md`](docs/subsystems.md) — per-subsystem deep dives (~35 sections)

If you're a brand-new contributor, read those first, then come back
here.

## Setup

```bash
git clone https://github.com/<your-fork>/augmentum.git
cd augmentum
python -m venv .venv
.venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1   |  bash: . .venv/bin/activate
pip install -e ".[dev]"
```

**Tested platforms.** The proxy + test suite are exercised on
`x86_64 Linux` (CI), `x86_64 Windows` (maintainer host), and
`arm64 macOS` is supported on a best-effort basis (no live Mac in
the maintainer's loop — see [`docs/MAC_HARDENING.md`](docs/MAC_HARDENING.md)
for the platform-support punch list). Most heavy deps
(`torchaudio<2.9`, `onnxruntime`, `kokoro-onnx`, `wespeakerruntime`,
`webrtc-noise-gain`) ship arm64 wheels; a small number compile from
source on first install, so make sure Xcode Command Line Tools are
present (`xcode-select --install`) before `pip install -e ".[dev]"`
on macOS. GPU acceleration on Apple Silicon falls back to CPU/MPS
inside the bundled `llama-server` and to CPU for ONNX-Runtime models
— functional, just slower than the CUDA path.

For full local runs, you'll also want Docker (the workspace +
provider services + searxng all run via compose). The `setup.bat` /
`setup.sh` wizard writes a `.augmentum.conf` listing which compose
overlays you want to enable.

```bash
start.bat                 # start core stack (no rebuild)
start.bat build           # start + rebuild images
docker compose up -d      # equivalent if you'd rather not use start.bat
```

You can develop the proxy without Docker — just point augmentum at
an OpenAI-compatible backend (Ollama / LM Studio / your own
`llama-server`) via the model-settings UI or `.env`.

## Run the test suite

With the venv active:

```bash
pytest tests/ -q                       # full offline suite
pytest tests/test_coder_routes.py -q   # one module
pytest -m "not live and not slow" -q   # exclude expensive tests
pytest tests/live/ -q                  # live integration (needs running services)
```

See [`docs/testing.md`](docs/testing.md) for the rulebook (what to
mock vs hit real, when to mark `@pytest.mark.live`, etc.).

## The health audit (run this before every commit)

```bash
python .claude/skills/augmentum-dev/scripts/audit.py            # default — score, delta, hotspots
python .claude/skills/augmentum-dev/scripts/audit.py --quiet    # summary only
python .claude/skills/augmentum-dev/scripts/audit.py --smoke    # also verify imports + migrations apply
python .claude/skills/augmentum-dev/scripts/audit.py --with-tests   # also actually run pytest
python .claude/skills/augmentum-dev/scripts/audit.py --verbose      # also show per-subsystem flag breakdown
python .claude/skills/augmentum-dev/scripts/audit.py --update-baseline   # commit current state as the new baseline
```

Score interpretation: 90+ EXCELLENT, 75-89 GOOD, 60-74 FAIR,
40-59 DEGRADED, <40 CRITICAL. The audit fails (exit 1) if any
metric regressed vs the committed baseline at
`.claude/skills/augmentum-dev/references/audit_baseline.json`.

If your PR legitimately adds findings (e.g. shipping a new subsystem
grows the orphaned-route count until UI is wired), bump the baseline
in the same PR and explain why in the commit message. **Do not** bump
the baseline to silence regressions you didn't cause.

The audit is wired through:

- `validate_wiring.py`  — settings 4-layer integrity, route
                          registration, migration ordering
- `dead_code.py`        — orphaned endpoints + ghost frontend calls
- `code_quality.py`     — silent JS catches, WS contract, dead CSS,
                          console.log
- `runtime_checks.py`   — empty-model fetches, unhandled fetch
                          failures, app.state guard usage
- `security_check.py`   — SSRF / XSS / SQL surface
- `db_safety.py`        — destructive SQL + missing user_id scoping
- `db_health.py`        — schema drift, orphaned rows, FK integrity
- `test_coverage.py`    — module + route coverage
- `red_team_scan.py`    — adversarial patterns
- pip-audit             — dependency CVEs (install with
                          `pip install pip-audit`)
- exception validation  — stale entries in security_exceptions.json
- doc-fact verification — claims in CLAUDE.md / SKILL.md vs reality

## Adding a setting

```bash
python .claude/skills/augmentum-dev/scripts/scaffold_setting.py \
       my_feature_enabled bool false
```

The script prints copy-pasteable boilerplate for all four layers a
setting must touch:

1. **`augmentum/config.py`** — Python default on the `Settings` model
2. **`augmentum/proxy/config_routes.py`** — API validation (in
   `_TOOL_SETTINGS` for booleans, `_STRING_SETTINGS` for strings with
   max-length cap)
3. **`augmentum/proxy/server.py`** — `_SETTINGS_RESTORE_MAP` entry so
   the value is rehydrated from the SQLite settings store on startup
4. **`ui/scripts/settings.js`** — frontend default + load/save +
   change handler

A setting that exists in only some layers silently fails (UI saves
locally, server reverts on restart). `validate_wiring.py` catches
this — every entry in `_TOOL_SETTINGS` / `_STRING_SETTINGS` must have
a matching `_SETTINGS_RESTORE_MAP` entry and a corresponding
camelCase key in `settings.js`'s `DEFAULTS`.

## Adding a route

```bash
python .claude/skills/augmentum-dev/scripts/scaffold_route.py bookmarks
```

Generates `augmentum/proxy/bookmarks_routes.py` with the standard
shape and prints the import + `app.include_router(...)` lines you
need to paste into `server.py`. Run `validate_wiring.py` after to
confirm the registration took.

## Adding a migration

```bash
python .claude/skills/augmentum-dev/scripts/gen_migration.py \
       "add_my_feature_table"
```

Auto-detects the next number and creates
`augmentum/state/migrations/NNN_add_my_feature_table.sql`. Pattern:

```sql
CREATE TABLE IF NOT EXISTS my_feature (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id  TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (NNN, 'add_my_feature_table');
```

Rules:

- Always `CREATE TABLE IF NOT EXISTS`. Always wrap `ALTER TABLE …
  ADD COLUMN` in a try/except — the runner gracefully skips
  duplicate-column errors so re-running an existing install is
  idempotent.
- Foreign keys to `ui_sessions` need `get_or_create_session()` called
  first by the inserting code path.
- **User data tables get a `user_id` column.** Every CRUD function
  that reads/writes the table accepts `*, user_id: str = ""` and
  scopes its query when `user_id` is non-empty. See
  [`docs/security_model.md`](docs/security_model.md) for why.

## Frontend rules

- **Every `${...}` interpolation in an `innerHTML =` template literal
  must use `escapeHtml()`.** Backtick-injection is a real XSS vector.
  `code_quality.py` flags unsafe ones.
- **Sessions are loaded as metadata stubs on init**, then full data
  on demand via `ensureSessionLoaded()`. Don't fetch all sessions'
  trees up front.
- **Save patterns:** `saveSessions()` is debounced; `_flushActiveSession()`
  is the synchronous flush after a message; `navigator.sendBeacon`
  on unload sends only the active session (stays under the 64KB
  limit).

## Code style

- Python: ruff-clean. `from __future__ import annotations` at the top
  of every module. Type hints on every function signature.
  `structlog` for logging via `get_logger(__name__)`.
- JS: vanilla, no framework. ES modules. DOM access only — no jQuery,
  no React, nothing transpiled.
- Comments: write the **why**, not the **what**. The code already
  shows what. If a comment is just paraphrasing the line below it,
  delete it.

## Commit + PR

- Commit messages: `type(scope): subject` first line (e.g.
  `fix(coder): shell injection in git remote endpoint`), body
  explaining **why**, not **what**. Look at the recent log for the
  house style.
- Run `audit.py --smoke` (or `--with-tests` if you touched logic)
  before pushing. Failing audits will get bounced.
- One commit per coherent change. If a PR is doing five things, it
  should be five commits — that's how `git bisect` and `git revert`
  stay useful.

## Where to ask

For now: open a GitHub issue. Augmentum is pre-launch, so the
contributor base is small — your question won't get lost.
