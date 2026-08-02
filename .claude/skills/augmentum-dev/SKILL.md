---
name: augmentum-dev
description: >
  Augmentum development accelerator. Use this skill whenever implementing features,
  adding settings, creating routes, writing migrations, or modifying the UI in
  Augmentum. Also use when onboarding, debugging wiring issues, or reviewing PRs.
  Triggers on: "add a setting", "new endpoint", "new feature", "scaffold",
  "validate wiring", "health check", "project health", "why isn't my setting working",
  "contribution guide", "db health", "database corruption", "migration safety",
  "images/artifacts disappeared", "orphaned files", any feature implementation in this codebase.
---

# Augmentum Development Accelerator

You are working on **Augmentum** — a FastAPI intelligence-layer proxy between LLM frontends and backends. Before writing code for any feature, understand the wiring patterns below. After writing code, run the validation scripts to catch mistakes.

## Quick Commands

**Project health audit** — full-scope health system. Bundles every individual scanner, runs dependency-CVE scan, validates audit-infrastructure (stale exception entries, doc-fact accuracy), computes a 0-100 health score, surfaces cross-tool hotspots, optionally runs a runtime smoke test or the actual pytest suite. **Run this at the start of any non-trivial session and again before claiming a task is done.**

```bash
python ${CLAUDE_SKILL_DIR}/scripts/audit.py            # default: full pass, score + delta
python ${CLAUDE_SKILL_DIR}/scripts/audit.py --quiet    # summary only
python ${CLAUDE_SKILL_DIR}/scripts/audit.py --verbose  # also show per-subsystem flag breakdown
python ${CLAUDE_SKILL_DIR}/scripts/audit.py --smoke    # also verify create_app() imports + migrations apply (~30s)
python ${CLAUDE_SKILL_DIR}/scripts/audit.py --with-tests  # also run pytest offline subset (up to 10min)
python ${CLAUDE_SKILL_DIR}/scripts/audit.py --skip-deps   # skip pip-audit (faster, but loses CVE check)
python ${CLAUDE_SKILL_DIR}/scripts/audit.py --trend 10    # show last 10 runs from history
python ${CLAUDE_SKILL_DIR}/scripts/audit.py --format=json     # machine-readable
python ${CLAUDE_SKILL_DIR}/scripts/audit.py --format=markdown # PR-comment friendly
python ${CLAUDE_SKILL_DIR}/scripts/audit.py --update-baseline # commit current state as the new baseline
python ${CLAUDE_SKILL_DIR}/scripts/audit.py --no-history      # skip history append (dry run)
```

**What it checks (30 metrics across 12 categories):**
- core scanners: wiring, dead_code, code_quality, runtime, security, coverage, red_team, db_contention, **db_safety** (SQLite footguns — `AUTOINCREMENT`, non-idempotent `CREATE`, non-WAL `journal_mode` in the state layer, live-DB `shutil.copy`, unbounded `DELETE`/`DROP`; see the *Database health & safety* section), **async_blocking** (event-loop blockers — `time.sleep`/`requests.*`/`subprocess.run` and sync embedding calls made directly inside `async def`; the static counterpart to the `event_loop_stall` runtime watchdog)
- **deps**: pip-audit CVEs against pyproject.toml (skipped gracefully if pip-audit not installed; install with `pip install pip-audit`)
- **exceptions**: `security_exceptions.json` entries that reference files no longer in repo (audit-infrastructure rot)
- **doc_facts**: verifiable claims in `CLAUDE.md` / `SKILL.md` checked against reality (e.g. user-scoped table count, highest migration)
- **doc_coverage**: code⟷doc SET drift — subsystem packages / dispatch modes / providers that exist in the tree but have no doc entry (see [Self-Maintaining Docs](#self-maintaining-docs-how-the-skill-tracks-the-codebase-on-its-own)); each spec is an independent `<spec>_undocumented` metric
- **smoke** (--smoke): does `create_app()` import + do all migrations apply on a fresh `:memory:` DB
- **tests** (--with-tests): does pytest pass on the offline test set

**Score interpretation:**
- 90+ EXCELLENT — ship-quality
- 75-89 GOOD — solid, minor debt
- 60-74 FAIR — working but notable debt to clear
- 40-59 DEGRADED — debt accumulating, focus needed
- <40 CRITICAL — address immediately

**Pre-task** — `audit.py --quiet` (records snapshot in history; quick).
**Pre-merge** — `audit.py --smoke --with-tests --verbose` (the full thing).
**CI** — `audit.py --format=json --smoke --with-tests` (parseable; non-zero exit on regression / smoke fail / test fail).

When you intentionally land work that legitimately adds findings (e.g. shipping a new subsystem grows orphaned-route count until UI is wired), bump the baseline in the same commit so the next session starts with an accurate floor. Do not bump it to silence regressions you didn't cause.

**Cross-tool hotspots** are files flagged by 2+ different scanners (e.g. `media_routes.py` showing up in dead_code AND runtime AND security). These are higher-priority than any single-tool finding because the same file is failing multiple checks — fix one and you usually fix several.

**Per-subsystem breakdown** (with `--verbose`) groups all flags by their parent dir, telling you where the debt is concentrated. Often `ui/scripts/` or one specific subsystem dominates the list — focus there for the biggest score impact per fix.

**Validate current wiring** (catch missing registrations, orphaned settings, broken round-trips):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/validate_wiring.py
```

**Live ground-truth acceptance** (LIVE stack required — asserts observed
reality, not code shape: real HTTP routes, real containers, real browser
sessions, real DOM state. This is the rung that catches what unit tests
can't — e.g. the Docker-bridge-isolation and page-reopen-state bugs of
2026-07-17 were only findable here). Run after changing the coder browser
stack, workspace profiles, the builds behavior gate, or anything
container-networking-adjacent:
```bash
python ${CLAUDE_SKILL_DIR}/scripts/live_acceptance.py --list        # registry
python ${CLAUDE_SKILL_DIR}/scripts/live_acceptance.py --init        # write config template
python ${CLAUDE_SKILL_DIR}/scripts/live_acceptance.py --suite browser
python ${CLAUDE_SKILL_DIR}/scripts/live_acceptance.py --suite browser --allow-disruption  # + stop/start drills
```
Ground rules baked in: LLM-dependent checks read the model from
`live_acceptance.local.json` (git-ignored) or `--model` and SKIP when
unset — the model is never auto-selected. Disruptive checks (they stop
services) only run with `--allow-disruption`. The suite refuses to start
below `min_free_memory_gb` free host RAM (default 8) so it can never
stack onto an active model load. Every run mints a scoped bench session
and always revokes it, deletes its test workspaces, and closes its
browser sessions — pass or fail. `bench_harness.py` is the reusable
substrate (token mint/revoke, in-container python via stdin, docker
helpers) for writing new checks; add checks with the `@check(suite)`
decorator, keep one ground truth per check, and put anything flaky
behind `depends_on` rather than retries.

**Generate next migration**:
```bash
python ${CLAUDE_SKILL_DIR}/scripts/gen_migration.py "description of change"
```

**DB safety scan** (static — bundled into `audit.py`; run standalone after touching migrations or DB code):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/db_safety.py
python ${CLAUDE_SKILL_DIR}/scripts/db_safety.py --verbose  # show suppressed entries
```

**Live DB health & reconciliation** (integrity_check, FK check, WAL size, file⟷row drift — run after any corruption/recovery, or when "my X disappeared"):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/db_health.py            # auto-finds the DB
python ${CLAUDE_SKILL_DIR}/scripts/db_health.py --db /data/augmentum.db --data-dir /data   # --full for integrity_check; --json for machine output
# In Docker: docker exec <container> python /path/to/db_health.py --db /data/augmentum.db
```
It auto-discovers each file-backed table's store dir (e.g. `image_generations` → `image_output/`, `artifacts` → `data/artifacts/`); `--data-dir` overrides the root if the layout differs.

**Generate interactive health dashboard**:
```bash
python ${CLAUDE_SKILL_DIR}/scripts/health_report.py
```

`audit.py` runs `validate_wiring.py` along with the rest of the suite — prefer it for end-of-task checks. Run `validate_wiring.py` standalone only when you want the fast 2-second wiring-only signal mid-edit.

**Security check** (context-aware — understands what's intentional):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/security_check.py
python ${CLAUDE_SKILL_DIR}/scripts/security_check.py --verbose  # show suppressed findings
```

**Dead code & coverage check** (orphaned endpoints, ghost calls, untested routes):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/dead_code.py
python ${CLAUDE_SKILL_DIR}/scripts/dead_code.py --verbose  # list suppressed entries
```
Reviewed false positives (a ghost call the URL-matcher mis-reads, a route that intentionally has no JS caller) go in `dead_code_suppressions.json` — `ghost_calls`: `"METHOD /api/path"`, `orphaned_endpoints`: `"METHOD /api/path"`. Don't suppress real findings; for the rolling accepted *count* of mid-build orphans, use `audit.py --update-baseline` instead.

**Code quality check** (silent catches, WS contract, console.log, CSS/JS alignment, tech debt):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/code_quality.py
python ${CLAUDE_SKILL_DIR}/scripts/code_quality.py --verbose  # include CSS class details
```

**Runtime bug-pattern scan** (catches what wiring validation misses — empty models, silent exceptions, unhandled fetch):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/runtime_checks.py
python ${CLAUDE_SKILL_DIR}/scripts/runtime_checks.py --verbose  # show suppressed findings
```

**Async event-loop blocking scan** (static — bundled into `audit.py`; run after touching any `async def` in `augmentum/`):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/async_blocking.py
python ${CLAUDE_SKILL_DIR}/scripts/async_blocking.py --verbose  # show suppressed findings
```
Flags blocking calls made *directly* inside `async def` (never inside a nested `def`/`lambda` — the `to_thread` offload idiom is correctly ignored; awaited calls are never flagged). **ERROR**: `time.sleep`, `requests.*`, `subprocess.run`/`call`/`check_*`, `os.system`, top-level `httpx.*`. **WARNING**: `subprocess.Popen` (fork/exec) + synchronous `EmbeddingService.embed_*` calls (the 2026-06-13 event-loop-stall class). Fix by wrapping in `ctx.run_in_thread()` / `asyncio.to_thread()`, or switching to the async API. Reviewed false positives → `async_blocking_suppressions.json` (`findings`: `"path:line"`).

**Scaffold a new setting** (prints copy-pasteable 4-layer boilerplate):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/scaffold_setting.py my_feature_enabled bool false
python ${CLAUDE_SKILL_DIR}/scripts/scaffold_setting.py my_model_override str ""
```

**Scaffold a new route** (generates route file + prints server.py registration lines):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/scaffold_route.py bookmarks
```

**Test coverage gap check** (finds untested modules, routes, orphaned tests):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/test_coverage.py
python ${CLAUDE_SKILL_DIR}/scripts/test_coverage.py --verbose  # list all untested modules
```

**Red team scan** (adversarial security analysis — SQL injection, XSS, data isolation, token exposure):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/red_team_scan.py
python ${CLAUDE_SKILL_DIR}/scripts/red_team_scan.py --verbose  # include data isolation + context leak checks
```
Shares `references/security_exceptions.json` with `security_check.py` — an entry whose `id` contains the category slug (`sql-fstring-*`, `shell-injection-*`, `weak-random-*`) and lists the file suppresses that finding in *both* scanners (decide once). Add new exceptions there, not by editing the scanner.

**Refresh reference caches** (auto-runs with validate, or standalone):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/refresh_refs.py
```

**Refresh fact-fenced doc claims** (codebase model — rewrites `<!--fact:NAME-->...<!--/-->` blocks in CLAUDE.md / SKILL.md from live SQL queries against the model):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/refresh_docs.py --check    # exit 1 on drift, prints diff
python ${CLAUDE_SKILL_DIR}/scripts/refresh_docs.py --apply    # rewrite stale values in place
python ${CLAUDE_SKILL_DIR}/scripts/refresh_docs.py --list     # list registered facts + values
```

**Self-healing:** a `Stop` hook (`scripts/heal_docs_hook.sh`, wired in `.claude/settings.json`) runs `refresh_docs.py --apply` automatically when an agent finishes a turn — but only if a fact-source file (migration, route, `config.py`, `settings.js`, test) changed since the last heal, guarded by an mtime stamp so the common no-op turn costs ~0.3s instead of ~5s. It rewrites the doc bodies in the working tree but does **not** `git add` them (you stage CLAUDE.md / SKILL.md yourself, per parallel-session commit hygiene). This is why the fact counts above rarely drift anymore; you only need `--apply` by hand if the hook is disabled.

**Skill self-diagnosis** (`skill_doctor.py`) — one command that answers "is this skill still describing reality, and what must a *human* do about it?" It splits the answer into the two autonomy tiers (see [Self-Maintaining Docs](#self-maintaining-docs-how-the-skill-tracks-the-codebase-on-its-own) below) so intervention stays minimal: Tier-1 stale facts (auto-heal, no action) vs Tier-2 coverage gaps (need a description → a human). Exits non-zero when a human has work pending, so it's CI-usable.
```bash
python ${CLAUDE_SKILL_DIR}/scripts/skill_doctor.py            # full self-diagnosis + "HUMAN ACTION REQUIRED"
python ${CLAUDE_SKILL_DIR}/scripts/skill_doctor.py --scaffold # also print paste-ready stub rows for every gap
python ${CLAUDE_SKILL_DIR}/scripts/skill_doctor.py --heal     # apply the Tier-1 fact heal first, then report
python ${CLAUDE_SKILL_DIR}/scripts/skill_doctor.py --json     # machine-readable
```

**Scaffold a doc row** (`scaffold_doc_row.py`) — turn every coverage gap into a paste-ready stub so the residual human step is filling one description cell, not authoring a row:
```bash
python ${CLAUDE_SKILL_DIR}/scripts/scaffold_doc_row.py            # stubs for every spec with a gap
python ${CLAUDE_SKILL_DIR}/scripts/scaffold_doc_row.py subsystems # just one spec
python ${CLAUDE_SKILL_DIR}/scripts/scaffold_doc_row.py --list     # list registered coverage specs
```

**Diagnose codebase health** (model queries with subsystem clustering):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/diagnose.py                       # all queries
python ${CLAUDE_SKILL_DIR}/scripts/diagnose.py orphaned_endpoints   # one query
python ${CLAUDE_SKILL_DIR}/scripts/diagnose.py --list                # list registered queries
```

**Subsystem health card** ("what's the state of feature X?"):
```bash
python ${CLAUDE_SKILL_DIR}/scripts/subsystem_card.py                 # top 10 highest-signal subsystems
python ${CLAUDE_SKILL_DIR}/scripts/subsystem_card.py narrative      # full card for one subsystem
python ${CLAUDE_SKILL_DIR}/scripts/subsystem_card.py --list          # every subsystem with signal
python ${CLAUDE_SKILL_DIR}/scripts/subsystem_card.py --top 20        # top N by signal
```
Each card bundles routes (with orphan flags), settings (with missing-layer attribution), user-scoped tables, and recent file activity for one subsystem. The card layout extends automatically as new ingesters land — no per-subsystem template maintenance.

The model is rebuilt opportunistically into `.augmentum-dev-cache/codebase.db` (gitignored). 11 facts and 2 queries ship today (Phases 0-2):
- **facts**: table count + list, max migration, registration count + first line, endpoints/js_calls/orphan counts, settings count + fully-wired + incomplete
- **queries**: `orphaned_endpoints` (endpoints with no JS reference, clustered by subsystem), `incomplete_settings` (settings wired in 3 of 4 layers, attributed to the missing layer)

Live counts as of this doc render:
- Endpoints: <!--fact:endpoints.count-->1335<!--/--> · JS calls: <!--fact:js_calls.count-->1218<!--/--> · Orphaned (strict join): <!--fact:orphaned_endpoints.count-->464<!--/-->
- Settings: <!--fact:settings.count-->610<!--/--> total · <!--fact:settings.fully_wired-->0<!--/--> fully-wired (4/4 layers) · <!--fact:settings.incomplete-->554<!--/--> incomplete (3/4 layers)
- Tests: <!--fact:test_files.count-->1068<!--/--> files · <!--fact:test_files.test_count_total-->18904<!--/--> `def test_*` functions · <!--fact:untested_routes.count-->24<!--/--> untested route files
- Multi-tenant audit: <!--fact:multi_tenant_audit.count-->219<!--/--> route handlers in user-scoped subsystems with no detected `user_id` wiring (candidate list — review for cross-tenant leak risk)

Add new facts in `facts/registry.py`; new queries as a module in `queries/`; embed facts in docs with the same `<!--fact:NAME-->...<!--/-->` syntax.

### Self-Maintaining Docs: how the skill tracks the codebase on its own

This skill is documentation-as-code, so it rots as Augmentum grows unless the *maintenance itself* is automated. It is, in three tiers — the goal is that a growing codebase needs **near-zero manual doc upkeep**, and whatever residue is genuinely human is named explicitly by `skill_doctor.py`.

**Tier 1 — self-healing facts (zero intervention).** Any countable claim wrapped in `<!--fact:NAME-->…<!--/-->` is rewritten from a live SQL query against the codebase model by the `Stop` hook every turn a fact-source file changes. *Rule: never hand-type a number in these docs — wrap it in a fact.* That is why `endpoints.count`, the user-scoped table list, `modes.count`, `route_modules.count`, and `subsystems.count` cannot drift. Add one in `facts/registry.py` (a `(description, sql, formatter)` tuple) and fence it in the doc.

**Tier 2 — coverage detectors (auto-detect; a human writes the description).** Countable claims self-heal, but a *set the docs should mirror* (every subsystem has a Map row, every mode a Handler row, every provider a card) can't be auto-written — a wrong description is worse than a visible gap, and auto-authoring prose violates "never auto-select on the user's behalf". So the automation is **detection, not fabrication**: `doc_coverage/` diffs a code-derived set against what the docs declare and reports every gap on each audit (`doc_coverage.<spec>_undocumented` metrics, regression-gated against the baseline). Adding a new tracked list is **one `CoverageSpec` in `doc_coverage/specs.py`** — no audit edit (the score auto-weights any new spec). Today's specs: `subsystems`, `modes`, `provider_cards`.

**Tier 3 — scaffolding (fill one cell).** `scaffold_doc_row.py <spec>` emits a paste-ready stub for every gap, so the only human act is replacing `TODO` with a one-line description.

**The operator loop is one command:** `skill_doctor.py` runs Tiers 1–2 and prints a `HUMAN ACTION REQUIRED` block naming exactly what's left (usually: "fill N description cells; everything else self-heals"), with `--scaffold` to emit the stubs and `--heal` to apply the Tier-1 refresh first. Run it at the end of any session that added a subsystem / mode / provider, or whenever you want to know the skill's own drift posture.

**Design rule when extending:** anything countable → a **fact** (Tier 1); any *set* the docs must mirror → a **`CoverageSpec`** (Tier 2, keep the code⟷doc mapping unambiguous or the diagnostics lose trust); reserve humans for semantic judgement only (Tier 3). This is the "one spine, integrate as data not code" discipline — don't add a parallel checker when a fact or a spec will do.

`audit.py` accepts:
- `--use-model` — route doc_facts through the model AND run all model queries with diagnosis output (subsystem breakdowns, sample paths, missing layers).
- `--record-fix-event` — append the current audit's deltas to `fix_events` for pattern memory (Phase 5+ analysis).
- Phase 3 causality is on by default: scanner-self-change tagging means a regression caused by editing the scanner itself (e.g. fixing a parser) is marked `[SELF-CHANGE]` and excluded from the regression count, so the score never punishes work that surfaces previously-hidden metrics.

**Scanner conventions:** `scripts/_common.py` is the shared home — importing it makes stdout/stderr UTF-8-safe (so a `✓` doesn't crash a scanner on a Windows console), and it exports `find_root()`, the `red/yellow/green/...` colors, and `load_suppressions()` / `is_suppressed()`. Every scanner has a companion `*_suppressions.json` (or shares `references/security_exceptions.json`) for *reviewed* false positives — `path` / `path:line` / dir-prefix entries; `--verbose` lists what's suppressed. **Suppress false positives, not real findings**; for accepting a *rolling count* of legitimate growth, bump the audit baseline instead.

---

## Derived Reference Files

These JSON files are **auto-generated** by `refresh_refs.py` (which runs automatically
as part of `validate_wiring.py`). They provide instant lookups without re-reading the
full codebase. Only regenerated when source files have changed (mtime check).

| File | What it contains | Generated from |
|------|-----------------|----------------|
| `references/routes.json` | Every backend endpoint: method, path, handler, file:line | `augmentum/proxy/*_routes.py` |
| `references/frontend_api_calls.json` | Every `fetch()` and WebSocket call from frontend JS | `ui/scripts/**/*.js` |
| `references/settings_map.json` | Full camelCase↔snake_case setting mapping + 4-layer coverage | config.py, config_routes.py, server.py, settings.js |

**Read these files** when you need to quickly answer:
- "Does this endpoint exist?" → check `routes.json`
- "Does the frontend call this API?" → check `frontend_api_calls.json`
- "What's the JS name for this Python setting?" → check `settings_map.json`

---

## Multi-Agent Code Review

When the user asks for a "code review", "audit Connect / narrative / X for
shipping quality", "have subagents review this", or any variant — read the
**[multi-agent review playbook](references/multi_agent_review.md)**. It
documents the partition rules, prompt template, severity rubric, and
consolidation process used for the Connect 2026-06-05 audit. Reuse it
verbatim for new subsystems instead of re-deriving the prompts.

Key points:
- Partition into 4–10 non-overlapping zones (2k–4k LOC each).
- Launch all agents in a single message with `run_in_background: true`,
  `subagent_type: general-purpose`.
- Read-only review — agents must not write code.
- Consolidate by theme, not by reviewer; promote P1→P0 when ≥2 reviewers
  hit the same finding from different zones.
- Save the top findings to memory for next-round verification.

---

## Subsystem Map

Augmentum has grown to <!--fact:subsystems.count-->81<!--/--> top-level subsystem packages under `augmentum/` (served by <!--fact:route_modules.count-->111<!--/--> `*_routes.py` modules). This table covers the major ones — it is **not** hand-exhaustive; the `doc_coverage` audit check (in `audit.py`) flags any subsystem package that has no row here, so gaps surface automatically instead of rotting silently. Deep-dive docs live in [`docs/subsystems.md`](../../../docs/subsystems.md).

| Subsystem | Dir | Routes | Key classes | Purpose |
|-----------|-----|--------|-------------|---------|
| **Modes** | `augmentum/modes/{passthrough,analytical,narrative,agentic,coder,becca_direct,direct}/` | `openai_routes.py`, `ollama_routes.py`, `chat_routes.py` | `PassthroughHandler`, `AnalyticalHandler`, `NarrativeHandler`, `AgenticHandler`, `CoderHandler`, `BeccaDirectHandler`, `DirectHandler` | <!--fact:modes.count-->7<!--/--> request handlers; chosen by `classifier/router.py` (see Handler Pattern below) |
| **Coder** | `augmentum/coder/`, `augmentum/modes/coder/` | `coder_routes.py`, `coder_permission_routes.py`, `coder_review_routes.py` | `ContainerManager`, `CoderState`, `WorkspaceSnapshot`, `ScratchStore` | Containerized agentic coding: Plan/Act loop, semantic indexer, workspace snapshots, git review flow |
| **Promises / Missions** | `augmentum/promises/` | (consumed by coder) | `MissionRunner`, `Promise`, `VerificationKind` | Structured plan with verifiers (replaces free-text plan_steps) |
| **Powers** | `augmentum/powers/` | `power_routes.py` | `PowerManifest`, `PowerController` | Capability packs that bias coder at safe checkpoints (guidance/verifier/workflow) |
| **Jobs** | `augmentum/jobs/` | `jobs_routes.py` | `JobRunner`, `JobContext` | Restart-survivable background queue (gutenberg_fetch, media_sync, …) |
| **MCP** | `augmentum/mcp/` | `mcp_routes.py` | `MCPClient`, `MCPServerConnection` | Stdio/pipe bridge to external MCP tool servers; tool-call forwarding |
| **Reasoning flows** | `augmentum/reasoning/` | `reasoning_routes.py`, `flow_routes.py` | `execute_flow_stream()`, `ReasoningFlow`, `FlowStep` | User-defined step pipelines (classify/search/verify/respond) for analytical mode |
| **Classifier** | `augmentum/classifier/` | (called from routers) | `classify_request()`, `complexity_analyzer`, `narrative_detector` | Chooses mode; header/prefix override > heuristics |
| **Narrative** | `augmentum/modes/narrative/` | `narrative_routes.py`, `knowledge_routes.py`, `character_routes.py`, `persona_routes.py` | `NarrativeEngine`, `NarrativeHandler`, `SessionMemorySettings` | STATE/LEDGER/ARCHIVE 3-layer memory, character cards, personas |
| **Memory** | `augmentum/memory/` | `memory_routes.py` | `MemoryStore`, `EmbeddingService` | sqlite-vec embeddings, consolidation, compaction |
| **Dream** | `augmentum/dream/` | `dream_routes.py` | `DreamEngine`, `DreamScheduler`, `DreamJournal`, `PortraitManager` | Persona introspection cycles + portraits |
| **Documents / RAG** | `augmentum/documents/` | `document_routes.py` | `DocumentStore`, `chunk_with_parents()` | Chunking + FTS5 + embedding retrieval |
| **VFS (File Index)** | `augmentum/vfs/` | `files_routes.py` | `FileIndexService`, `VFSAdapter` | Unified file catalog across uploads, media, docs, artifacts, chat images |
| **Media** | `augmentum/media/` | `media_routes.py` | `MediaServer`, `sync_media_catalog()` | Audiobookshelf/Emby/Jellyfin/Komga/LibriVox/Suwayomi sync + DLNA receivers |
| **Games (web)** | `augmentum/games/` | `games_routes.py` | `GameBrowseResult`, `providers/js13k.py` | Web game discovery (JS13K). itch.io provider was removed. |
| **Game streaming (emulator)** | `augmentum/game_stream/` | `game_stream_routes.py` | `GameStreamSession`, docker adapter | Emulator-based game streaming sessions backed by `compose.game-stream.yaml` (`agsp-streamed`). Phase-1 includes per-user 2/2 live-stream cap. |
| **Titles / saves / marketplace** | `augmentum/titles/`, `augmentum/saves/`, `augmentum/marketplace/` | `titles_routes.py`, `titles_saves_routes.py`, `titles_bios_routes.py`, `titles_marketplace_routes.py`, `marketplace_routes.py` | Title catalog, BIOS classifier, save game store | Per-user game titles, save-game persistence, marketplace listings (server-level table). |
| **Connected devices** | `augmentum/devices/` | `device_routes.py` | `DeviceRegistry`, DLNA adapter, body-atlas substrate | DLNA receivers, cast targets, body-aware avatar substrate (per-VRM voxel grid). |
| **Controllers** | `augmentum/controllers/` | `controllers_routes.py` | Gamepad/HID remapping | Game controller pairing + per-user remap profiles (`controller_remaps`). |
| **Discovery** | `augmentum/discovery/` | `discovery_routes.py` | `filter_and_rank()`, `filter_for_llm()`, `filter_for_docs()` | Consumer-specific quality filtering + recommendation |
| **Image** | `augmentum/image/` | `image_routes.py`, `cloud_image_routes.py`, `chat_image_routes.py` | `ImagePipeline`, provider adapters | Local SD/GGUF + cloud (OpenAI/Together/Stability/BFL/Fal). No model is pre-baked — users install via the model picker on first run. |
| **Voice** | `augmentum/voice/` | `voice_routes.py`, `voice_enrollment_routes.py`, `audio_routes.py` | VAD/STT/TTS pipeline, speaker verification | WebSocket voice loop + TTS provider registry |
| **Avatar** | `augmentum/avatar/` | `avatar_routes.py` | `AvatarStore`, body-atlas voxel grid | VRM 3D avatars + lipsync metadata. 10 bundled VRMs (mode-agnostic visual presets). |
| **Cardsmith** | `augmentum/modes/narrative/cardsmith/` | `cardsmith_routes.py` | Card generation pipeline | LLM-driven character card creation. Long-running endpoints (`/turn`, `/finalize`) — expect 5-40s per call. |
| **Models / Engine v2** | `augmentum/models/` | `model_routes.py` (+ `engine_router`, `llamacpp_router`) | `LlamaServerManager` (subprocess), provider registry | Bundled `llama-server` binary pinned via `LLAMA_SERVER_VERSION`. **Don't revive services/engine/ (v1 retired).** |
| **Resource ledger** | `augmentum/resource/` | `resource_routes.py` | `ResourceLedger`, `TrackedModel`, `ModelProfile` | Cross-subsystem VRAM/RAM tracking |
| **Providers** | `augmentum/providers/` | `provider_routes.py`, `balancer_routes.py` | Provider registry | Backend catalog, Docker service lifecycle (marketplace listings live under Titles now). |
| **Auth** | `augmentum/auth/` | `auth_routes.py` | Argon2id, opaque tokens, raw ASGI middleware | Multi-tenant; see Multi-Tenant section |
| **Knowledge packs** | `augmentum/knowledge/` | `knowledge_routes.py` | `PackManager`, `ZimReader`, `wiki_extractor`, `convert_worker` | `.augpack` (sqlite-vec + FTS5) and `.zim` (Kiwix) corpora. Browseable in Browse panel via sandboxed iframe. Hybrid retrieval = vector + FTS + ZIM keyword merged with RRF. |
| **Browse / Notes** | `augmentum/notes/` | `browse_routes.py`, `notes_routes.py`, `note_intelligence_routes.py` | Web reader, Milkdown notes | Extraction pipeline + domain reputation |
| **Studio (artifacts)** | `augmentum/tools/artifact_*.py` | `artifact_routes.py`, `studio_routes.py` | `artifact_document/presentation/spreadsheet/chart` | DOCX/PPTX/XLSX/chart generation + in-app editing |
| **VFS (File Index)** | `augmentum/vfs/` | `files_routes.py` | `FileIndexService`, `VFSAdapter`, `_FILE_STATS_CACHE` | Unified file catalog. `/api/files/stats` is per-user TTL-cached (~30s) — chip-badge counts may lag writes by up to TTL. |
| **Cache mgmt** | (route only) | `cache_routes.py` | — | Cache inspection / invalidation endpoints. |
| **Grove** | (in `ui_routes.py` + settings) | `grove_routes.py` | — | Control center: theme, ambient music, vitals |
| **YouTube** | — | `youtube_routes.py` | — | Embedded playback + transcript sync |
| **Executor** | — | `executor_routes.py` | Sandbox runner | Code/sandbox execution |
| **Notifications / Metrics** | — | `notification_routes.py`, `metrics_routes.py` | — | SSE notifications (3 endpoints not polled by UI); TTFT/P99 metrics |
| **Companion (Becca)** | `augmentum/companion/`, `augmentum/companion_runtime/` | `companion_routes.py`, `companion_growth_routes.py` | `native_loop.py` (shared FC loop), presence/drive/energy/journal verbs | Autonomous companion — dispatches every mode as a subagent; owns primitives + tick verbs + behavior loops; per-user growth. Default OFF. |
| **Architect (observer)** | `augmentum/architect/` | `architect_routes.py` | `observe_attention()`, presence context | Cross-surface attention choke-point: `reportAttention(topic,payload)` → presence, so the companion knows where the user is. |
| **Self-edit** | `augmentum/selfedit/` | via `capability_routes.py`, `/api/selfedit/*` | `orchestrator`, `native_loop`, `verifier`, `foundry`, `palate`, `promote` | Sovereign self-edit recursion loop (intent→isolate→edit→self-heal→verify→gate→apply→observe→learn); oracle tiers; isolated `growth.db`. |
| **Intent (Action Registry)** | `augmentum/intent/` | `intent_capture_routes.py` | builtin verbs, `app_menu.py` (`app.act`), referent cache | Composable primitive verbs (`note.create`, `navigate.open_surface`, …) via 3 dispatch tiers (regex / embedding / LLM tool). |
| **Bug finder** | `augmentum/bug_finder/` | `bug_finder_routes.py` | `orchestrator`, findings/patterns/runs stores | Autonomous bug-hunting for coder mode (codebase-knowledge + findings ledger). |
| **Fabric** | `augmentum/fabric/` | `fabric_routes.py` | peer identities, device bundles, replay watermarks | Cross-instance peer federation — one device borrows another's GPU/model. |
| **Calling / Connect** | `augmentum/calling/`, `augmentum/connect/` | `connect_routes.py` | call sessions, threads (E2E), presence, contacts | WebRTC user-to-user voice/video calls + text threads. |
| **Game agent** | `augmentum/game_agent/` | `game_agent_routes.py` | `agent`, `perception`, `llm_bridge`, probes/rule_packs | AI plays games via vision (frame → classifier → action loop). |
| **Learning** | `augmentum/learning/` | `learning_routes.py` | vocab state, language-learning drills | Language-learning surface. |
| **Observation** | `augmentum/observation/` | `observation_routes.py` | cross-modal pattern memory | Passive cross-modal pattern memory feeding recommendation / presence. |
| **Personality** | `augmentum/personality/` | `persona_routes.py` | facet activations, doc candidates, associations | Persona-facet substrate (per-user personality doc + facet cooccurrence). |
| **Cast** | `augmentum/cast/` | `cast_routes.py`, `cast_games_routes.py`, `cast_game_proxy_routes.py` | `input_bridge`, cast profiles | Cast-to-TV (screen + game-stream casting, receiver events). |
| **XR / Embodiment** | `augmentum/xr/` | `xr_routes.py` | XR sessions, seats, session events | WebXR binding for the VRM avatar (seats, spatial director). |
| **Training** | `augmentum/training/` | — (CLI + capture) | `capture.py` (live trace capture) | Trace-capture + companion-model training pipeline (see `docs/companion-model-training-design.md`). |
| **Scheduling** | `augmentum/scheduling/` | — (chat/voice/companion entrypoints) | `SchedulerService`, `utils/cron.py` | App-level timed-action substrate (cron engine; multi-user headless dispatch). |
| **Library** | `augmentum/library/` | `library_routes.py`, `library_save_routes.py` | collections, publications, activity | Per-user content library (collections, saved items, publications). |
| **Builds / Projects** | `augmentum/builds/`, `augmentum/projects/` | `build_routes.py` | build runs, project repos/checkouts/refs | Build-run tracking + per-user project repo bindings. |
| **Vision** | `augmentum/vision/` | `vision_routes.py` | `router.py` (VL primary → classifier → SmolVLM fallback) | Image captioning/vision as a capability of the classifier slot. |
| **World model** | `augmentum/world_model/` | `world_routes.py` | world state | Environment/world-state model (game-agent + simulation). |
| **Dance** | `augmentum/dance/` | `dance_routes.py` | dance loops, ratings, history | Avatar dance-loop library + playback. |
| **Security / OCR** | `augmentum/security/`, `augmentum/ocr/` | — (internal) | secret encryption; OCR extraction | Cross-cutting internal services (Fernet secret-at-rest; OCR text extraction). |

**Finding a route fast**: `cat references/routes.json | python -m json.tool | less` or search it with Grep.

---

## The Wiring Contract

Every user-configurable setting in Augmentum must exist in **all four layers**. A setting that exists in only some layers causes silent failures — the UI appears to save but the value doesn't persist, or the server loads a default on restart.

```
                    ┌─────────────────────┐
                    │   settings.js       │  Frontend: DEFAULTS + load + save + sync
                    │   (camelCase)        │
                    └────────┬────────────┘
                             │ PUT /api/config/tools  (or /ui)
                             ▼
                    ┌─────────────────────┐
                    │  config_routes.py   │  Validation: _TOOL_SETTINGS or _STRING_SETTINGS
                    │  (snake_case)       │
                    └────────┬────────────┘
                             │ settings_store.set()
                             ▼
                    ┌─────────────────────┐
                    │    config.py        │  Python defaults: Settings(BaseSettings)
                    │  (snake_case)       │
                    └────────┬────────────┘
                             │ startup restore
                             ▼
                    ┌─────────────────────┐
                    │    server.py        │  _SETTINGS_RESTORE_MAP: type caster
                    │  (snake_case)       │
                    └─────────────────────┘
```

### Adding a new boolean setting: `my_feature_enabled`

**Step 1 — config.py** (Python default):
```python
my_feature_enabled: bool = False
```

**Step 2 — config_routes.py** (API validation):
```python
# In _TOOL_SETTINGS:
"my_feature_enabled": (bool, 0, 1),
```

**Step 3 — server.py** (startup restore):
```python
# In _SETTINGS_RESTORE_MAP:
"my_feature_enabled": _parse_bool,
```

**Step 4 — settings.js** (frontend):
```javascript
// 1. In DEFAULTS:
myFeatureEnabled: false,

// 2. In loadToolSettingsFromBackend():
settings.myFeatureEnabled = data.my_feature_enabled ?? DEFAULTS.myFeatureEnabled;

// 3. In syncToolSettingsToBackend():
my_feature_enabled: settings.myFeatureEnabled,

// 4. Wire the UI control's change handler to call syncToolSettingsToBackend()
```

The naming convention: **snake_case** everywhere in Python, **camelCase** in JavaScript. The mapping happens in the sync functions.

### Adding a string setting

Same flow, but use `_STRING_SETTINGS` instead of `_TOOL_SETTINGS`:
```python
# config_routes.py:
"my_model_override": 256,   # max length

# server.py _SETTINGS_RESTORE_MAP:
"my_model_override": str,
```

### Backend-only (non-UI) settings — how to find and set them

The 4-layer contract is for settings that get a **UI control**. Many settings
(<!--fact:settings.incomplete-->554<!--/--> of <!--fact:settings.count-->610<!--/-->) are **intentionally backend-only** — power-user / dev
toggles wired in 3 layers (config.py + config_routes.py + server.py) with **no
`settings.js`**. This is NOT a defect (despite the `incomplete_settings` query
labeling them "3/4 layers"): most `coder_*`, `selfedit_*`, `engine_*`, and
breaker/gate toggles are deliberately API-only. **A backend-only setting still
has one authoritative doc: its `config.py` field comment** — that's where the
"what it does" lives, so always write a real one-line comment there.

**To SET a backend-only setting** (same endpoint the UI uses — no restart, persists to the settings store):
```bash
# bool / int (in _TOOL_SETTINGS) and strings (in _STRING_SETTINGS):
curl -X PUT http://localhost:6100/api/config/tools \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer <token>' \
  -H 'Origin: http://localhost:6100' \
  -d '{"coder_think_tool_enabled": true}'
```
(From inside the container use `http://127.0.0.1:6100`. Origin must match Host — CSRF.)

**To FIND / list them** (three ways, cheapest first):
- `python .claude/skills/augmentum-dev/scripts/diagnose.py incomplete_settings` — every setting missing the `settings.js` layer (i.e. all backend-only ones), attributed to the missing layer.
- `references/settings_map.json` — full snake↔camel map + per-setting 4-layer coverage (auto-generated by `refresh_refs.py`).
- Grep the source of truth: `_TOOL_SETTINGS` / `_STRING_SETTINGS` in `config_routes.py` (bool/int/string validation) and the field comments in `config.py`.

**When adding a backend-only setting**, do the 3 backend layers, write the
`config.py` comment, and **skip `settings.js`** to match the sibling pattern —
`validate_wiring.py` will list it as "in config_routes but not synced from
settings.js" (a warning, not an error); that's expected and correct for an
API-only toggle.

---

## Route Registration Pattern

Every route file must be:
1. Imported in `server.py` (inside `create_app()` — exact line drifts; locate with the grep below)
2. Registered with `app.include_router()` (~line <!--fact:registrations.first_line-->9083<!--/--> onward — <!--fact:registrations.count-->119<!--/--> registrations today)

The exact line range drifts; locate with:
```bash
grep -n "_routes import router" augmentum/proxy/server.py | head -1   # start of import block
grep -n "app\.include_router" augmentum/proxy/server.py | head -1      # start of registration block
```

```python
# server.py — import section:
from augmentum.proxy.my_routes import router as my_router

# server.py — registration section:
app.include_router(my_router)
```

The route file itself defines the prefix:
```python
router = APIRouter(prefix="/api/myfeature", tags=["myfeature"])
```

**Common mistake**: Creating a route file but forgetting to register it. The validation script catches this.

---

## Migration Pattern

Migrations live in `augmentum/state/migrations/` and are numbered sequentially (highest as of this doc: **<!--fact:migrations.max-->332<!--/-->**). `gen_migration.py` auto-detects the next number — don't hardcode. There is one historical gap at 059 (never committed); not a defect, ignore it. Pattern:

```sql
-- NNN_my_feature.sql  (NNN = next number)
CREATE TABLE IF NOT EXISTS my_feature (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

Rules:
- Always use `IF NOT EXISTS` for **`CREATE TABLE` and `CREATE INDEX`** — migrations must survive a re-run / partial-apply
- **Never `AUTOINCREMENT`** — plain `INTEGER PRIMARY KEY` is the rowid (reuse is harmless); `AUTOINCREMENT` adds a `sqlite_sequence` table + write amplification and was implicated in past corruption (see migration 139, which removed it)
- Always use `ALTER TABLE ... ADD COLUMN` with a try/catch pattern for existing tables
- Foreign keys referencing `ui_sessions` need `get_or_create_session()` called first
- The migration runner executes files in alphabetical order on startup
- A bare `DELETE FROM <t>` (no WHERE) or `DROP TABLE` in a migration is unbounded data loss — only do it for an explicit table rebuild, and `db_safety.py` will flag it (whitelist the rebuild in `db_safety_suppressions.json`)

`db_safety.py` (bundled into `audit.py`) enforces these statically — run `audit.py` after touching migrations.

---

## Database health & safety

The main store is one SQLite file (`{data_dir}/augmentum.db`, WAL mode, via `augmentum/state/backends/sqlite.py`). It has corrupted and been salvaged repeatedly — `data/` accumulates `*.corrupt*` / `*.backup-*` / `*.recover.sql` snapshots, and the `fix(sqlite): …` commit series (FTS5 auto-repair, `VACUUM INTO` backups, `_recover_corrupt_db` fd-leak/once-per-incident gate, bounded shutdown WAL checkpoint, dropping `AUTOINCREMENT`) is the ongoing hardening. Treat the DB as fragile and assume any recovery may have *dropped rows* (a `.recover` salvage skips tables whose pages were in the damaged region) — the **files those rows pointed at survive on disk**, so the symptom is "my images/artifacts vanished" while `/data/image_output` is full.

**Rules:**
- **All main-DB access goes through `augmentum/state/`** (`SQLiteBackend`), so WAL + `busy_timeout` + pragmas are applied consistently. Raw `sqlite3.connect()` is fine for *side* DBs (knowledge packs, ZIM readers, the coder index, the dream journal) — those are their own files — but never open `augmentum.db` ad-hoc.
- **Backups: `VACUUM INTO`**, never `shutil.copy*` of a *live* DB — a copy captures a torn WAL state. (Forensic copies of an already-quarantined/dead DB are fine — they're whitelisted in `db_safety_suppressions.json`.)
- **WAL is mandatory** for the state layer; never `PRAGMA journal_mode` to anything else there. `db_safety.py` treats a non-WAL `journal_mode` in `augmentum/state/` as an *error*.
- **Shutdown checkpoints the WAL with a timeout** — don't remove that; an unbounded checkpoint that gets SIGKILL'd mid-write is a corruption vector.
- **File-backed tables drift.** `image_generations` ↔ `image_output/`, `artifacts`/`documents` ↔ their store dirs. The DB row and the file are not transactional with each other. Never assume row ⟺ file; reconcile.

**Tools:**
- `db_safety.py` — static scanner, bundled into `audit.py`. Run after migrations or anything that opens a DB.
- `db_health.py` — *live* diagnosis: `PRAGMA quick_check`/`integrity_check`, `foreign_key_check`, WAL size, and **file ⟷ row reconciliation** (broken refs + orphaned files). Run it after any corruption/recovery event, or whenever "my X disappeared". In Docker: `docker exec <container> python …/db_health.py --db /data/augmentum.db`.

**Recovering orphaned files** (the bytes are there; the rows aren't): `VACUUM INTO` a fresh snapshot of the live DB first → restore the missing rows from the newest `*.backup-*` DB, filtered to ids whose file still exists (`INSERT OR IGNORE`) → synthesize minimal rows for any files newer than that backup → restart. (This is exactly the 2026-05 image recovery — ~270 PNGs reconnected from a May-9 backup.)

---

## Frontend Patterns

### Template literal safety
Every user-provided string rendered in a template literal MUST use `escapeHtml()`:
```javascript
// WRONG — XSS via backtick injection:
el.innerHTML = `<div>${userName}</div>`;

// RIGHT:
el.innerHTML = `<div>${escapeHtml(userName)}</div>`;
```

`escapeHtml()` escapes: `<`, `>`, `&`, `"`, `` ` ``, and `${`.

### Session lazy loading
Sessions load as metadata stubs on init, full data loads on demand:
```javascript
// Load stubs (fast):
const resp = await fetch('/api/chats/?meta=1');

// Load full session when needed:
const full = await fetch(`/api/chats/${sessionId}`);
```

### Save pattern
```javascript
// Debounced save (most cases):
saveSessions();  // debounces, batches

// Immediate save (after message send):
_flushActiveSession();

// On page unload (must be sync):
navigator.sendBeacon('/api/chats/sync', payload);  // <64KB limit
```

---

## Subsystem Cookbooks (load on demand)

The per-subsystem "how to add a new X" recipes live in
**[`references/subsystem_patterns.md`](references/subsystem_patterns.md)**.
Read that file only when your change touches one of these — otherwise skip it
to keep context lean. The cross-cutting contracts every change must honor
(wiring, route registration, migrations, multi-tenant isolation, the
post-implementation checklist) stay inline above/below.

| Working on… | Section in `subsystem_patterns.md` |
|---|---|
| New Docker service / compose overlay | Docker Overlay Pattern |
| Coder mode internals (containers, state, promises, powers, terminal) | Coder Mode Patterns |
| Long-running background / GPU / I-O work | Background Jobs Pattern |
| External MCP tool servers | MCP Tool Bridge Pattern |
| Analytical-mode step pipelines | Reasoning Flow Pattern |
| llama-server upgrades / thinking-parser families | Engine v2 Pattern |
| Editor AST features (diagnostics, folding, autocomplete) | CodeMind |
| Adding a TTS provider | TTS Provider Pattern |
| Adding an image provider | Image Provider Pattern |
| Workspace / chat code-editor UI | Code Editor Patterns |
| Writing tests (tiers, fixtures, examples) | Testing Patterns |

---

## Internal LLM Call Pattern

When making internal LLM calls (title generation, flow generation, memory extraction, etc.), always pass the user's selected model — never use `model=""`.

```python
# WRONG — backend receives empty string, provider rejects with "Model Not Exist":
request = InternalChatRequest(model="", messages=[...])

# RIGHT — pass the model through from the API request:
request = InternalChatRequest(model=model, messages=[...])
```

**Three-layer threading:**
1. **Frontend** sends `model: app.state.currentModel` in the request body
2. **Route handler** extracts `model = body.get("model", "")` and passes it to the service function
3. **Service function** uses it in `InternalChatRequest(model=model, ...)`

If the model must be resolved without a frontend request (background tasks), use the backend's model list:
```python
available = await backend.list_models()
model = available[0].name if available else ""
```

The `runtime_checks.py` scanner detects `model=""` patterns that aren't overwritten within 5 lines.

---

## Handler Pattern (Modes)

Augmentum has **<!--fact:modes.count-->7<!--/--> modes** routing requests through distinct handlers (<!--fact:modes.list-->agentic, analytical, becca_direct, coder, direct, narrative, passthrough<!--/-->). The classic five extend `BaseHandler`; `becca_direct` / `direct` extend `ModeHandler`:

| Mode | Subdir | Handler | Purpose |
|------|--------|---------|---------|
| Passthrough | `augmentum/modes/passthrough/` | `PassthroughHandler` | Direct proxy + SSOS auto-tools |
| Analytical | `augmentum/modes/analytical/` | `AnalyticalHandler` | UARF 6-phase + tool calling |
| Narrative | `augmentum/modes/narrative/` | `NarrativeHandler` | Story/RP + 3-layer memory + cards |
| Agentic | `augmentum/modes/agentic/` | `AgenticHandler` | Goal-driven plan + artifact tools |
| **Coder** | `augmentum/modes/coder/` | `CoderHandler` | Plan/Act loop + workspace container + mission/promises |
| **Becca-direct** | `augmentum/modes/becca_direct/` | `BeccaDirectHandler` | Companion pipeline — dispatches every mode as a subagent (default OFF) |
| Direct | `augmentum/modes/direct/` | `DirectHandler` | Raw pipe — no classification, straight to the backend |

The mode list above is fact-fenced (`modes.count` / `modes.list`), so it self-heals when a mode package is added or removed. Request routing is through `augmentum/classifier/router.py` (heuristic: explicit header/prefix > complexity > narrative detector > fallback).

Each handler follows:
```python
class MyHandler(BaseHandler):
    async def process_stream(self, request, session, ...):
        ...
    async def save_state(self, session_id):
        ...
    async def load_state(self, session_id):
        ...
```

Handlers are cached per-user-session in `app.state`:
```python
# Cache key is (user_id, session_id) when auth is active:
cache_key = (user_id, session_id) if user_id else session_id
if cache_key not in app.state.narrative_engines:
    app.state.narrative_engines[cache_key] = NarrativeEngine(...)
```

---

## Multi-Tenant Data Isolation Pattern

Augmentum is multi-tenant. Every piece of user data MUST be scoped by `user_id`. This is the #1 security invariant — violating it leaks data between users.

### The Rule

Every function that touches a **user-scoped table** must accept `*, user_id: str = ""` as a keyword-only argument. When `user_id` is non-empty, all queries MUST include `AND user_id = ?`.

**User-scoped tables (<!--fact:tables.user_scoped.count-->200<!--/--> today).** The authoritative, always-current list lives in **`CLAUDE.md`** — the `User-scoped tables (N): ...` line, fact-fenced and rewritten from the live migration model by `refresh_docs.py`. Don't re-enumerate it here (a hand-typed copy rots — this section used to hardcode 76 and drifted to less than half the real count). Confirm any single table with:

```bash
grep -rn "user_id TEXT\|ADD COLUMN user_id" augmentum/state/migrations/
```

`audit.py`'s `doc_facts` check flags drift in the CLAUDE.md list on every run; `refresh_docs.py --apply` rewrites it.

**Server-level tables (NOT scoped):**
`providers`, `app_settings`, `managed_services`, `audio_providers`, `image_providers`, `knowledge_packs`, `settings`, `users`, `domain_reputation`, `resource_snapshots`, `resource_profiles`, `marketplace_listings`

> `voice_mixes` is **user-scoped** as of migration 093 — don't trust older docs that listed it as server-level.

### Three-Layer Scoping

```
Route handler:  user_id = request.scope.get("user").id
       ↓        passes user_id= to every data call
Store/persistence: *, user_id: str = ""
       ↓        appends AND user_id = ? to SQL
Cache keys:     (user_id, session_id) instead of session_id
```

### Adding a New User-Scoped Table

1. **Migration:** Add `user_id TEXT REFERENCES users(id)` column + index
2. **Store functions:** Add `*, user_id: str = ""` to ALL CRUD methods
3. **Route handlers:** Extract `_user_id(request)` and pass it through
4. **Cache keys:** If cached in `app.state`, key by `(user_id, session_id)`
5. **Tests:** Verify User A's data is invisible to User B

### Patterns

```python
# SELECT — append user_id filter:
async def get_item(self, item_id: str, *, user_id: str = ""):
    query = "SELECT * FROM items WHERE id = ?"
    params = [item_id]
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

# INSERT — include user_id column:
async def create_item(self, item_id: str, data: str, *, user_id: str = ""):
    cols = "id, data"
    phs = "?, ?"
    vals = [item_id, data]
    if user_id:
        cols += ", user_id"
        phs += ", ?"
        vals.append(user_id)
    await db.execute(f"INSERT INTO items ({cols}) VALUES ({phs})", vals)

# Shared/builtin items (flows, reasoning) — show user's + global:
async def list_flows(self, *, user_id: str = ""):
    if user_id:
        query = "SELECT * FROM flows WHERE user_id = ? OR user_id IS NULL"
        params = [user_id]
    else:
        query = "SELECT * FROM flows"
        params = []

# Route handler — extract user_id:
def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""
```

---

## Architecture Reference

For deeper documentation, read:
- [`references/subsystem_patterns.md`](references/subsystem_patterns.md) — per-subsystem cookbooks (TTS/image providers, coder, jobs, MCP, engine, editor, testing) — loaded on demand
- [`docs/patterns.md`](../../../docs/patterns.md) — All 15 recurring code patterns with examples
- [`docs/subsystems.md`](../../../docs/subsystems.md) — Mode handlers, voice pipeline, memory system deep dives
- [`docs/testing.md`](../../../docs/testing.md) — Test rulebook: how to write tests for this project
- [`docs/security_model.md`](../../../docs/security_model.md) — Threat model, trust boundaries, security patterns
- [`docs/red_team_review.md`](../../../docs/red_team_review.md) — Adversarial audit notes

---

## Post-Implementation Checklist

After implementing any feature, verify:

- [ ] Run `python ${CLAUDE_SKILL_DIR}/scripts/validate_wiring.py` — no new errors
- [ ] Run `python ${CLAUDE_SKILL_DIR}/scripts/runtime_checks.py` — no new errors
- [ ] Touched an `async def`? Run `python ${CLAUDE_SKILL_DIR}/scripts/async_blocking.py` — no new loop blockers
- [ ] Run `python ${CLAUDE_SKILL_DIR}/scripts/test_coverage.py` — no new coverage gaps
- [ ] Run `python ${CLAUDE_SKILL_DIR}/scripts/red_team_scan.py` — no CRITICAL/HIGH findings on changed files
- [ ] New settings exist in all 4 layers (config.py, config_routes.py, server.py, settings.js)
- [ ] New routes are imported AND registered in server.py
- [ ] New tables have migrations with IF NOT EXISTS
- [ ] **New user-scoped tables** have `user_id TEXT REFERENCES users(id)` column + index in migration
- [ ] **All CRUD on user-scoped tables** accepts `*, user_id: str = ""` and appends `AND user_id = ?`
- [ ] **Route handlers** extract `_user_id(request)` and pass to every data call
- [ ] **Cache keys** use `(user_id, session_id)` not bare `session_id`
- [ ] All user text in template literals uses escapeHtml()
- [ ] Internal LLM calls use the user's selected model (not empty string)
- [ ] New fetch() calls have error handling (.catch or try/catch)
- [ ] UI controls save to server, not just localStorage
- [ ] Data survives page refresh and server restart
- [ ] Textarea+overlay alignment: CSS properties match between .workspace-code and .workspace-code-highlight
- [ ] New TTS providers added to _BUNDLED_IDS and _BUNDLED list with correct endpoint paths
- [ ] New image providers use correct auth header format and payload structure
- [ ] New quick actions added to _QUICK_ACTIONS_CATEGORIES with correct mode (diff/full)
- [ ] New auto-fixers return {code, fixed, changes} and are wired into _silentLint + _autoFixCodeBlock
- [ ] Data survives page refresh and server restart
