# claude-aug integration — working notes / handoff

Status notes for continuing the Claude Code ↔ Augmentum integration.
(Started 2026-07-19 in a session rooted at ~/.augmentum.)

## Done so far

- **Wrappers** (`claude-aug.ps1`/`.sh`): scoped env (snapshot/restore — fixed
  a bug where plain `claude` broke after `claude-aug` in the same shell),
  `--profile deep|fast|mixed`, `--model`, `--small`, health check with
  container auto-start, `CLAUDE_CODE_SUBAGENT_MODEL` for subagent routing.
- **Full config isolation**: `CLAUDE_CONFIG_DIR=~/.augmentum/claude-config`
  (own .claude.json, settings, sessions, skills). Pre-seeded onboarding +
  approved key hash (last 20 chars of API key).
- **ATP → MCP bridge** (`atp-mcp-bridge.py`): stdlib-only stdio MCP server;
  lists `/v1/tools/list` dynamically, forwards to `/v1/tools/call`. All 19
  whitelisted tools work (tested: calculator, web_search via SearXNG).
  Whitelist additions in `atp_routes.py` propagate to clients automatically.
- **Silent-failure fix**: built-in WebSearch/WebFetch are Anthropic
  *server-side* tools — dead on a local backend. Denied in aug settings;
  CLAUDE.md routes the model to `mcp__atp__web_search` etc. instead.
- **Rigor pack** (`skills/`): plan-gate, adversarial-verify (verification
  delegated to ATP tools, not model judgment), scope-fence, deep-answer.
  Purpose: make local models (Qwen3.6, Ornith-35B, DeepSeek quants) feel
  Opus-class via process instead of parameters.
- **Installer** (`install.ps1`/`install.sh`): idempotent, generates all
  user-specific config (paths, key hash, container name), tested end-to-end.

## Key codebase facts (from a deep survey)

- `proxy/harness.py`: `X-Augmentum-Harness` header triggers per-turn memory
  injection (800-token budget) + background learning capture (human-gated).
  The wrapper already sends `X-Augmentum-Harness: claude_code`.
- `proxy/anthropic_routes.py`: claude-* tier aliasing (`anthropic_alias_haiku/
  sonnet/opus/default`) with tier-aware thinking (haiku=off, opus=preserve_thinking);
  KV prefix-cache pinning for sub-1s TTFT on 35B models.
- `proxy/atp_routes.py`: explicit whitelist (19 of 90+ registry tools),
  health-gated listing.
- `coder/browser.py` + `browser_sidecar.py`: agent-browser (vercel-labs,
  v0.32.1, pinned in AGENT_BROWSER_VERSION) runs in `augmentum-browser-1`
  with persistent Chrome sessions. NOT yet exposed via ATP.
- Ornith models are served (Ornith-1.0-35B/9B GGUFs) but absent from
  model_catalog.py — dropped-in GGUFs, not catalog entries.

## Known issue: memory scope bleed (CONFIRMED)

`proxy/harness.py` uses one flat HARNESS_SCOPE per user — all projects AND
all harnesses share one memory pool, and procedural conventions are injected
"ALWAYS" on every turn. Result: conventions learned in one project bleed into
unrelated projects, and models parrot conventions constantly. The wrappers
now send `X-Augmentum-Project: <cwd basename>` (alongside the harness header)
so the server can scope memory to {harness}:{project} once implemented.

## Done: Browser tools via ATP (2026-07-19, verified live)

`augmentum/tools/browser_tools.py` — 5 registry Tools (`browser_navigate`,
`browser_screenshot`, `browser_action`, `browser_evaluate`, `browser_wait`)
driving the agent-browser sidecar directly via `browser_sidecar.run_cli`/
`pull_file` (no refactor of browser.py; the evaluate wrapper is shared).
Key decisions:

- **No workspace**: sessions are per-user, server-side derived
  (`atp-<user_id>` via `session_for_workspace(uid, prefix="atp")`) —
  tenant boundary preserved; page state persists across calls
  (navigate → evaluate → screenshot without re-passing url works).
- **Screenshots → ArtifactStore**: returns `/api/artifacts/<id>/download`
  (fetch with the same x-api-key), never base64.
- **localhost rewrite**: harness-side `localhost`/`127.0.0.1` URLs are
  rewritten to `host.docker.internal` (the sidecar's vantage on the host).
- **ATP-only surfaces**: `SurfaceExposure(chat=False, coder=False,
  flow=False)` so registering them widens no chat/coder/flow surface.
- **Async health gating**: sidecar liveness needs an await; ATP routes now
  prefer `health_check_async()` when a tool defines it (`_tool_healthy` in
  atp_routes.py). Sync `health_check` stays cache-based for other surfaces.
- Registered in server.py (artifact-store block); whitelisted in ATP_TOOLS
  (24 tools total now). Verified: container restart → /v1/tools/list shows
  all 5 → navigate/evaluate/screenshot(PNG downloads)/action(click
  navigated)/wait all round-trip through POST /v1/tools/call.

## Done: Vision/OCR via ATP (2026-07-19, plumbing verified; live call pending a vision model)

`augmentum/tools/vision_tools.py` — `vision_describe` (VisionRouter,
QUALITY workload) + `ocr_extract` (docling `extract_page_script`,
assemble=False). Input is an artifact id / `/api/artifacts/<id>/download`
URL (ownership-checked via `ArtifactStore.get(user_id=...)`) or an
http(s) URL — raw filesystem paths rejected (multi-tenant boundary).
This closes the loop: `browser_screenshot` → artifact URL →
`vision_describe(image=<that url>)`. Both whitelisted in ATP_TOOLS (26).

Health gates verified honest on the current box: no vision provider was
live (primary down, SmolVLM STOPPED, classifier not vision-serving) and
no docling container → both tools hide from /list and /call returns
"currently unavailable". Artifact/url/id resolution + user-isolation +
raw-path rejection verified with a stubbed router. **Open**: one live
`vision_describe` + `ocr_extract` round-trip once a vision-capable
model / the ocr container is up (didn't auto-start heavy services).

## Done: Doctor (2026-07-19, built client-side, verified live)

`scripts/claude-aug/doctor.py` — standalone stdlib deployment-health checker.
The augmentum-dev audit answers "is the code right"; the doctor answers "is
the world right, right now". 15 checks across infra/proxy/atp/harness/hygiene
layers; PASS/WARN/FAIL/SKIP with evidence + one-line remedy; `--format=json`;
`--fix` (safe remedies only: container start, ps1 BOM restore); `--skip-live`.
Installed to ~/.augmentum/doctor.py by both installers; runnable as
`claude-aug doctor`. Distinguishes slow-vs-dead (proxy >5s = WARN not FAIL).

- Stdout carries stable `live.<metric>=N` lines (0 = clean). **TODO for a
  repo session**: add a Tool entry + trivial parser in
  .claude/skills/augmentum-dev/scripts/audit.py so runtime health joins the
  0-100 score as category "live" (weights for the new metrics too).
- CONVENTION: when an integration item lands, grow EXPECTED_ATP_TOOLS in
  doctor.py in the same diff (baseline-bump discipline). Browser tools are
  already in the manifest (24 expected live). vision_describe/ocr_extract
  are whitelisted-but-gated — deliberately NOT in the manifest until a
  vision/ocr backend runs by default; the doctor's "new unlisted" note will
  surface them the day they come alive.
- Field note: on its first real run the doctor caught the n8n restart loop,
  proxy warm-up latency, and observed the browser tools landing (manifest
  diff) — while a parse race in its own concurrency design was found and
  fixed (same-layer checks run concurrently: they must be self-contained,
  no intra-layer ctx dependencies; cross-layer ctx reads are safe).

## Full ATP sweep results (2026-07-19, all 24 live tools called with real args)

21/24 round-trip clean. Latencies: instant (calc/json/hash/units/math/py),
0.5-5s (web_fetch/wikipedia/youtube/browser_*), ~11s (image_search,
research — 6 cited sources). Browser chain verified: navigate → evaluate →
wait → screenshot(artifact URL) → action(click actually navigated).
Client-side fixes made from findings: bridge now unwraps ATP's HTTP-500
JSON error bodies (was showing opaque "HTTP Error 500" to the model),
truncates outputs >60k chars with a narrowing hint, and claude-config
CLAUDE.md now documents per-tool gotchas + the UNTRUSTED-content policy +
the screenshot fetch recipe (curl with x-api-key → Read).

**Server-side bugs for a repo session:**
- `consistency_check`: listed as healthy but /call fails "Backend call
  failed:" (empty detail) — its health_check doesn't verify the backing
  LLM slot. A tool that lies; make health reflect the backend.
- `search_files`: HTTP 500 with EMPTY error message on a plain query —
  both the failure and the blank error shape need fixing.
- `math_verify`: equation input ("x**2-4=0") → raw "invalid syntax" 500;
  a friendlier error ("numeric expressions only; use python_exec+sympy to
  solve") would save every harness model a retry loop.
- `image_search` output says "already displayed to the user inline" —
  chat-UI phrasing leaking to harness callers; make phrasing surface-aware.
- `wikipedia` relevance: returned "Identity fusion" for "reciprocal rank
  fusion" — consider title-match confidence or multi-result output.

## Done: Memory scope isolation (item 3) + memory_store (item 4) — 2026-07-19, verified live

**Scopes** (`proxy/harness.py`): global `harness` scope now holds ONLY the
seeded defaults; learned/harvested memories live in
`harness:<harness>:<project>` (project from `X-Augmentum-Project`, sanitized)
or the shared `harness:default` when the header is absent. Migration 315
moved all legacy non-seed flat-scope rows to `harness:default` (verified in
DB: 23 seeds stayed global, 11 learned rows moved; nothing deleted).
`memory/store.py` isolation is now prefix-aware (`is_isolated_scope` /
`_isolation_sql`) so `harness:*` sub-scopes stay unreachable from the
general pool in all 5 filter sites. Recall = global seeds (procedural) +
project scope; facts = project scope only. Staged candidates carry
`target_scope`/`supersedes_baseline_scope`; `promote_candidate` writes into
them (legacy records fall back to `harness:default`).

**Injection taming**: new setting `harness_conventions_mode` (config.py,
default `"first_turn"`): full conventions block only when the transcript has
no assistant message yet; facts-only afterwards. `"always"` restores old
behavior. Live-verified: turn 1 injects (conventions=True), turn 2 in
another project injects nothing.

**memory_store** (`tools/harness_memory.py`, whitelisted in ATP, in
doctor's EXPECTED_ATP_TOOLS): stages ONE candidate through the harvest
pipeline (never a live write), secret-regex-guarded; harness/project
identity is header-derived by the ATP route (`_inject_user_context` now
also sets ctx.harness/ctx.project — body can't spoof it). Live loop
verified: memory_store → staged in `harness:claude_code:augmentum` →
promote via /api/harness/harvest/promote → row in project scope → next
briefing injected it (items 5→6) and a different project did NOT see it.

**Two real recall bugs found & fixed while verifying** (memory/store.py):
1. Vector KNN fetched `limit*2` neighbors GLOBALLY (all users/scopes) then
   post-filtered — a small isolated scope never survived the cut, so
   strict-scope recall was silently blind. Now over-fetches (k≥256) when a
   scope filter will discard candidates.
2. The FTS type filter was appended as bare SQL after `MATCH ?`
   (`AND memory_type:(...)`) — a SQL parse error, so EVERY
   memory_types-filtered recall's FTS leg silently died (caught + logged at
   debug). Filter moved inside the MATCH expression where FTS5 column
   syntax is valid.

## Done: Research delegation (item 5) + artifacts (item 6) — 2026-07-19, verified live

**Research**: `flow_deep_research` (the existing FlowTool over the Deep
Research custom flow) + new `task_status` poll tool
(`tools/flow_status.py`) are whitelisted. Launch returns a task id
immediately; poll `task_status {task_id}` until completed/failed — results
are user-scoped (`get_task(user_id=...)`) and expire ~1h after completion.
Live-verified: launch → poll → full cited synthesis ran on local slots.
Two real bugs fixed on the way:
- FlowTool launches NEVER passed `tool_registry` into
  `BackgroundChainManager.launch` → every FlowTool-initiated flow died
  instantly with "Backend or tool registry not available"
  (handler_factory.py `_launcher` now binds it). This was broken for chat
  function-called flows too, not just ATP.
- The poll tool must NOT be named `flow_status`: the flow re-sync
  unregisters every `flow_*` tool on flow CRUD (learned live — it vanished
  seconds after registration). Hence `task_status`.
- FlowTool now uses `Tool.extract_user_id` so the ATP `_context` user
  reaches the task (was `_user_id`-only → anonymous tasks via ATP).

**Artifacts**: `create_document` / `create_chart` / `create_spreadsheet`
whitelisted. Verified: `create_document` (needs `title` + `sections:[{
heading, level, content}]`, NOT freeform `content`) produced a real PDF
with `/api/artifacts/<id>/download`. All five joined doctor's
EXPECTED_ATP_TOOLS (30 tools live).

ATP surface is now 31 whitelisted (30 live; vision_describe/ocr_extract
health-gated off until a vision/ocr backend runs).

## Done: Sidecar HTTPS via real CA trust (2026-07-19, verified live)

Replaced `AGENT_BROWSER_IGNORE_HTTPS_ERRORS=1` with proper trust:
- Dockerfile.browser: + `libnss3-tools`; entrypoint imports mounted root
  CA(s) into Chrome's NSS db (`~/.pki/nssdb`) then execs the sleep loop.
- compose.browser.yaml: ro-mounts the **caddy_data volume** at `/caddy-ca`
  (NOT `/data/caddy/pki/authorities/local` — verified live that :6443
  serves a leaf signed by the custom "Augmentum Local Root CA" at
  `/data/ca.crt`, not caddy's internal PKI; the internal root is imported
  too if it ever becomes readable). Private keys are 0600 root →
  unreadable to the sidecar's uid-1000 user. Ignore flag now defaults 0
  (break-glass only). Also added `extra_hosts: host.docker.internal:
  host-gateway` for plain-Linux parity.
- compose.yaml: `DNS:host.docker.internal` added to the base SAN list —
  first nav failed with ERR_CERT_COMMON_NAME_INVALID because the leaf
  had no such SAN; the SAN-hash mechanism auto-reissued the leaf on
  caddy restart (root CA unchanged, so the NSS import stays valid).
- live_acceptance.py: new `https_ca_trust` check (browser suite) navs
  https://host.docker.internal:6443/ui/ with a throwaway session —
  PASSES (title "Augmentum", ignore flag 0).
- Suite status: sidecar_running / https_ca_trust / workspace_create_slim /
  ladder_engine_sidecar / session_cleanup_on_delete PASS.
  `gate_real_llm_binding` (0-1/3, real-LLM-quality-bound — only the
  classifier model is up) and `dual_session_isolation` (b-counter flake,
  varies per run, plain-http pages so not TLS-related) FAIL — pre-existing
  /environmental, left open.
- Gotcha for operators: `docker compose up -d` with a PARTIAL -f list
  recreates augmentum-main without the dev overlay (stale baked code, no
  ATP) — always pass the full COMPOSE_FILES set from .augmentum.conf.

Bonus: with compose.ocr.yaml up, `ocr_extract` went live and the item-2
open verification is CLOSED — real screenshot artifact → OCR text with
bboxes round-tripped via ATP. (`vision_describe` still awaits a vision-
capable model.)

## Done: Capability wave 2 — meta-tier, packs, sandbox, agent bridge (2026-07-19/20, verified live)

All maximally reused from existing substrates; 36 tools now in /list.

- **Meta-tier (atp_routes.py)**: `GET /v1/tools/discover?q=` searches the
  long tail; eligibility = any registry tool ALREADY chat-exposed
  (`_discoverable`: same user, same _context isolation as chat, so no new
  capability boundary). `POST /call` accepts discovered tools. Schemas
  returned only when ≤5 matches (context protection). The bridge's
  `atp_discover`/`atp_call` MCP wrappers can now be pure pass-throughs.
- **pack_search** (`tools/pack_search_atp.py`): thin subclass of the
  coder's PackSearchTool (inert workspace args; ATP-only surfaces;
  health = packs installed). The self-hosted Context7. Verified live
  (devdocs python pack answered asyncio TaskGroup).
  Gotcha: import `augmentum.coder.tools` BEFORE `coder.knowledge_tools`
  (circular-import trip otherwise).
- **sandbox_shell** (`tools/sandbox_tools.py`): per-user persistent Docker
  sandbox reusing coder workspaces (`create_workspace(tooling_profile=
  "standard")`, id remembered at settings key `atp.sandbox.<user_id>`,
  auto-recreate if deleted). Verified live incl. cross-call persistence.
- **Agent bridge** (migration 316, `proxy/agent_bridge.py`,
  `tools/agent_bridge_tools.py`, routes in harness_routes.py, action
  handler on channel `harness.agent.request`):
  * `agent_checkin` — presence heartbeat (`harness_agent_sessions`);
    response piggybacks answered requests since last check-in.
  * `ask_user` — kinds approve (Approve/Deny buttons) / question / review
    (Reply… button) / notify (fire-and-forget). Publishes through
    `publish_and_dispatch` (WS + web push), importance 3.
  * `check_reply` — poll for the user's answer.
  * UI: `GET /api/harness/agents` (active agents + pending counts),
    `POST /api/harness/agent/reply` (free text), notifications.js branch:
    Reply… prompts BEFORE any POST (cancel leaves the banner + request
    pending — learned live: prompting after the action POST strands the
    request because post_action marks read → banner clears everywhere).
  * Verified live: checkin → ask(approve) → REAL notification button click
    → check_reply shows approve; review → free-text reply → agent picked
    it up at next checkin; agents list shows the session.
  * Gotchas: aiosqlite `cursor.rowcount` lied (0 for a landed UPDATE) —
    answer_request verifies by read-back. Windows curl mangles em dashes
    in -d bodies → "Invalid JSON body" (test-side only).
- Doctor manifest grown (agent_checkin/ask_user/check_reply/sandbox_shell;
  pack_search deliberately unlisted — packs aren't default-installed).
- **Open**: browser-refresh-dependent UI verification of the Reply prompt
  (request agrq_a97850bf89fb left pending for Matt); an "active agents"
  panel in the UI proper (endpoint exists); web-push (service worker)
  action path for reply untested — sw caps at 2 actions, fine for
  approve/deny, reply needs the tab.

## Done: Claude-side wiring (items 2-5) — 2026-07-19, verified live

- **atp_discover / atp_call** (`~/.augmentum/atp-mcp-bridge.py`): two
  synthetic meta-tools appended to the MCP tools/list (38 tools now: 36
  curated + 2 meta). discover calls GET /v1/tools/discover?q=, call
  routes through POST /v1/tools/call. The long tail of ~70 chat-exposed
  registry tools is now reachable without bloating the curated manifest.
  Verified: discover "image" → 9 matches (no schemas, >5); discover
  "ebook" → 2 matches (schemas included); atp_call on a discover result
  routed correctly to the real tool.
- **CC hooks** (`~/.augmentum/bridge-hooks.py` + settings.json hooks):
  SessionStart → agent_checkin (title from first prompt, stashes
  agent_id); Stop → agent_checkin(status=done) + ask_user(kind=review),
  cleans up session file; PreToolUse (matcher: Bash|Edit|Write|...) →
  ask_user(kind=approve), polls check_reply every 10s with 5-min
  timeout. One approval sets auto_approve for the rest of the session
  (prevents notification spam). Session state at
  ~/.augmentum/agent-sessions/{session_id}.json. Verified: SessionStart
  created session file with agent_id, Stop cleaned it up, PreToolUse
  routed a permission gate through the bridge (agrq_5ef12c849397).
- **CLAUDE.md tool guide** (scripts/claude-aug/CLAUDE.md, synced to
  ~/.augmentum/claude-config/): new sections — Agent bridge protocol,
  Sandbox (persistent Docker workspaces), Docs & reference (pack_search),
  Research delegation (flow_deep_research + task_status), Long-tail tools
  (atp_discover + atp_call).
- **adversarial-verify skill**: added reproduction step preferring
  sandbox_shell over python_exec for bugs/builds/installs (real container
  beats model recall). Synced to installed location.
- **Web-push caveat** (recorded below): service workers cap at 2 actions —
  Approve/Deny fit. Free-text reply ("Reply…" button) requires the
  Augmentum tab open. This is a browser constraint, not fixable. The
  active-agents panel (/api/harness/agents) is the natural mitigation:
  open the tab on any device, see pending asks, reply inline.
- **n8n restart loop**: known WARN on doctor (unrelated pre-existing;
  the staging container keeps restarting — resource or config issue,
  not bridge-related).

## Design principles established (unchanged)

- Curate tool exposure (~25-30 real schemas max); long tail via meta-tools.
- Verification delegated to deterministic tools, not model confidence.
- Health-gated capability: a tool that isn't there beats a tool that lies.
- Everything replicable via the installer — no hand-crafted user state.

## Web-push limitation (design constraint)

Chrome's service-worker notification API caps at 2 action buttons
(Notification.maxActions). The agent bridge uses these for:
- kind=approve → [Approve, Deny]
- kind=question/review → [Reply…]

For approve/deny, both buttons fit and work from any device (phone,
watch, etc.). For reply, the "Reply…" button triggers a `prompt()`
dialog in the open tab — push-triggered notifications can't prompt
for text (service workers have no DOM). Mitigation planned: the
active-agents panel provides an inline reply field in the UI proper.

## Capability wave 3 — one-call sign-in + recipes (DONE, live-verified)

Driven by a session audit of the proxied deepseek-v4-pro worker runs:
the model's #1 tool error was Edit-before-Read (10/34), and reviewing any
signed-in UI surface cost ~5-6 browser calls, most of them login boilerplate
repeated every session. Two additions close both.

- **`browser_ensure_auth`** (`augmentum/tools/browser_tools.py`) — mints a
  session server-side from the caller's OWN identity (the harness is already
  API-key-authed) and injects it as the `augmentum_session` httpOnly cookie
  via agent-browser `cookies set`. No login form, no password in transcript.
  Idempotent: reads the existing cookie, `validate_token`s it, no-ops if it
  matches the user. GOTCHA (fixed): `cookies set` only persists once a page
  is open — the tool opens `/ui/` FIRST, then sets, then reloads. Verified:
  1st call mints, 2nd is a no-op, and the cookie authenticates real API calls
  (`/api/auth/me`, `/api/config/tools`, `/api/playlists` all 200). The 401s
  that appear in a screenshot's network-failure diagnostics right after auth
  are STALE (buffered from the pre-auth first load) — not a real failure.
- **`atp_recipe`** (`recipe_tool.py` + `recipe_store.py`, migration 317
  `atp_recipes`) — named, per-user macros: save an ordered list of ATP tool
  steps with `{{placeholder}}` args, replay in ONE call. Steps run through
  the SAME registry + `_inject_user_context` gate the `/call` route uses, so
  a recipe reaches only tools its owner could already call, as their own
  user; steps can't call `atp_recipe` (no recursion); max 12 steps. Verified:
  save→list→run (`{{n}}=7` → `49`), and `review_gallery` =
  [ensure_auth, screenshot] collapses the whole gallery-review flow to one
  call returning a screenshot artifact URL.
- Both added to `doctor.py` EXPECTED_ATP_TOOLS and documented in CLAUDE.md
  (Browser + new Recipes section). 38 tools now live in `/v1/tools/list`.
- **CLAUDE.md: read-before-Edit rule** added (hard rule) — the harness
  tracks read-state at the API layer, so a hook can't satisfy it for the
  model; the behavioral rule is the fix. Kills the top proxied-model error.

## Capability wave 4 — self-minting workflow memory (DONE, live-verified)

The Hermes/AWM/Voyager pattern: agents that save what worked and improve it
over time, no fine-tuning. Researched three reference systems (Hermes Agent's
auto-written Markdown skills + FTS `when_to_use` retrieval; ICML-2025 Agent
Workflow Memory's trajectory-induced workflows; Voyager's ever-growing skill
library + self-repair loop). Key axis: executable macro vs. SOFT procedural
memory. `atp_recipe` (wave 3) is the executable tier; this adds the soft tier.

- **`workflow`** (`workflow_tool.py` + `workflow_store.py`, migration 318
  `atp_workflows` + FTS5). A workflow = `when_to_use` trigger + numbered
  steps + description — natural-language GUIDANCE the model reads and adapts,
  not a script. Actions: save (mint/refine, version bump), search (FTS on
  trigger), list, get, delete, record_outcome (times_used/succeeded stats).
- **Auto-injection is the payoff**: `harness.py::_build_workflow_briefing`
  FTS-matches the incoming ask against `when_to_use` and injects the top-1
  workflow into the harness briefing (reusing `inject_harness_context`), so
  recall costs ZERO per-turn tool calls — the "don't plan from scratch" win.
  Subtractive: top-1 only, trigger must match. Gated by
  `harness_workflow_inject_enabled` (default on).
- **Governance (Matt's call): auto-mint + easy prune.** The model writes to
  its OWN per-user + `harness:project` scope directly (no staging gate —
  procedural how-to is lower-risk than facts-about-the-user); list/get/delete
  are the prune surface. Reuses the harness memory scope isolation.
- Verified live: mint v1 -> refine v2 (stats preserved) -> `search` matched a
  DIFFERENT phrasing ("show me the pictures gallery screenshot" hit a workflow
  triggered on "review the image gallery") -> record_outcome -> and a real
  harness `/v1/chat/completions` request logged `harness_context_injected
  workflow=True` for project=augmentum while an unrelated project (phonetics)
  injected NO workflow (isolation holds). 39 tools now live.
- Recipe (executable, wave 3) vs workflow (soft, wave 4) documented in
  CLAUDE.md so the model picks the right tier.

Open follow-ons (not built): a curator pass that flags/rewrites workflows
with low times_succeeded/times_used; version history + rollback (currently a
version counter only); optional "promote to shared scope" gate if a workflow
proves broadly useful. All were deferred by the auto-mint/easy-prune choice.

## Next up (agreed direction)

1. ~~Browser tools via ATP~~ DONE (above).
2. **`atp_discover`/`atp_call` meta-tools in the bridge**: expose the
   ~70 non-whitelisted registry tools with zero per-turn context cost
   (deferred tool loading; keeps small-model tool selection sharp).
3. **Bridge in standard claude's shared config** (safe: MCP is client-side
   HTTP, doesn't touch model routing; keep built-in WebSearch there).
4. Maybe: `--think` wrapper flag exploiting tier-aware thinking routing.

## Design principles established

- Curate tool exposure (~25-30 real schemas max); long tail via meta-tools.
- Verification delegated to deterministic tools, not model confidence.
- Health-gated capability: a tool that isn't there beats a tool that lies.
- Everything replicable via the installer — no hand-crafted user state.
