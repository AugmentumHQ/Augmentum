# Augmentum Security Model

The public threat model + posture lives in [`SECURITY.md`](../SECURITY.md)
at the repo root (deployment tiers, supported versions, reporting). This
doc is the **contributor-facing deep dive** — the patterns you have to
follow when extending the substrate.

If you're touching auth, multi-tenant scoping, fabric, community install,
SafeHttpClient, or any code that reads user-controlled input, read this
end to end before opening a PR.

## What changed in 2026

Augmentum is no longer the "single-user local proxy" the original
security doc described. The current posture:

- **Multi-tenant** — 139 user-scoped tables, Argon2id passwords, opaque
  session tokens, fail-closed ASGI middleware. Default install still
  binds `127.0.0.1`, but the auth substrate is real and exercised.
- **Federated** — paired-peer fabric (default OFF) routes 6 modalities
  to other Augmentum boxes via Ed25519-signed envelopes.
- **Cross-device** — cast tokens bridge browser ↔ TV ↔ phone to one
  user session.
- **Community-extensible** — community marketplace deep-links (auth-exempt
  GET preview + auth-gated POST install) pull configuration artifacts
  from a trusted GitHub origin allowlist.

Three new substrates with three distinct trust models. Mix them up and
you get cross-tenant leaks or peer-budget theft.

## Threat boundaries (current)

```
┌──────────────────────────────────────────────────────────────────────┐
│  User's machine                                                       │
│  ┌────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │  Browser (UI)           │  │  Augmentum container                  │ │
│  │  localhost:6443 (HTTPS) │──▶  ┌─────────────────────────────────┐ │ │
│  │  localhost:6100 (HTTP)  │  │  │  AuthMiddleware (fail-closed)    │ │ │
│  └────────────────────────┘  │  │  ↓ Argon2id + opaque tokens      │ │ │
│                               │  │  ↓ scope["user"] attachment       │ │ │
│  ┌────────────────────────┐  │  │  ↓ user_id scope on every CRUD    │ │ │
│  │  SillyTavern / Cursor  │──▶  └─────────────────────────────────┘ │ │
│  │  Claude Desktop (MCP)   │  │  ┌─────────────────────────────────┐ │ │
│  └────────────────────────┘  │  │  Sandboxed Python executor       │ │ │
│                               │  │  Cap-dropped, ro-fs, 512MB cap    │ │ │
│  ┌────────────────────────┐  │  └─────────────────────────────────┘ │ │
│  │  Phone / TV (cast)      │──▶  ┌─────────────────────────────────┐ │ │
│  │  cast tokens (30m TTL)  │  │  │  SQLite + WAL (139 user-scoped) │ │ │
│  └────────────────────────┘  │  └─────────────────────────────────┘ │ │
└──────────────────────────────────────────────────────────────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
        ┌─────────────────┐    ┌──────────────────────┐    ┌──────────────┐
        │  Internet        │    │  Paired fabric peer  │    │  Cloud LLM    │
        │  SafeHttpClient  │    │  Ed25519 envelopes   │    │  per-user key │
        │  blocks RFC1918  │    │  per-peer svc user   │    │  Fernet-at-rest│
        └─────────────────┘    └──────────────────────┘    └──────────────┘
```

## Trust zones

| Zone | What lives there | Trust given |
|---|---|---|
| Same user, same browser session | The UI talking to the proxy | Full |
| Same user, different surface (phone, TV, MCP client) | Cast tokens / API keys / MCP bearer | Full per-user |
| Same user, different device (cast couch co-op guest) | Named-guest profile under one host_user_id | Limited to host's session, isolated saves |
| Paired peer (fabric) | Another Augmentum box you've signed off | 6 modality endpoints only; path allowlist via `peer_middleware.py:95` |
| Anonymous browser hitting a public endpoint | `/`, `/login`, `/status`, `/community-install`, fabric pair, cast pair | Per-route minimum needed |
| Public LAN if `AUGMENTUM_BIND_HOST=0.0.0.0` | Anyone on your WiFi | Same as browser — auth IS the trust boundary |
| Internet (port-forwarded) | Tier C — not recommended without expert hardening | See `SECURITY.md` Tier C |

## Auth substrate

Centerpiece: `augmentum/auth/middleware.py::AuthMiddleware` — raw ASGI
(not BaseHTTPMiddleware, because it has to handle WebSocket upgrades).

**Public-path allowlist** at `middleware.py:19-58` (`_PUBLIC_PATHS`) and
`:53-130` (`_PUBLIC_PREFIXES`). Adding a route here MUST come with a
written justification in the diff because it's a fail-closed bypass.

Current exemptions and why:

| Path | Reason |
|---|---|
| `/`, `/ui`, `/api/version`, `/favicon.ico` | Static / unauthenticated entry |
| `/api/auth/login`, `/setup`, `/status` | Before-auth endpoints by definition |
| `/api/fabric/pair`, `/api/fabric/hello`, `/api/fabric/connect` | Peer-to-peer — Ed25519 envelope IS the auth |
| `/api/cast/pair/{start,poll,qr,establish-session}` | TV pages without session cookies |
| `/api/cast/blob/`, `/api/cast/render-output/` | Tokenized resource fetches (token is the credential) |
| `/api/cast/guest/{identify,claim,forget-device}` | Couch-co-op guests; invite token is the credential |
| `/api/cast/stream-auth/redeem` | Cookie-less rendering container redeems a one-shot token |
| `/community-install` | Cross-origin nav from augmentumhq.com; handler does its own `_resolve_user` |

**Self-resolved auth** is required for any public-path route that wants
to know the user's identity (because middleware skipped the attachment).
See [Pattern 27 in `patterns.md`](patterns.md#pattern-27-auth-middleware-exemption--self-resolve)
and `community_routes.py::_resolve_user`.

**Admin-gate prefixes** at `middleware.py:134` (`_ADMIN_PREFIXES`) —
authenticated but not admin returns 403. Use this for shared-infra
mutations (provider config, knowledge pack install, community Power
install, etc).

## Multi-tenant invariant (the #1 thing to not break)

Every CRUD function on a user-scoped table accepts `*, user_id: str = ""`
and appends `AND user_id = ?` to queries when set. The list of user-scoped
tables is auto-generated in `CLAUDE.md` from migration `user_id` columns;
`audit.py`'s `doc_facts` checker keeps it honest. Currently 139 tables.

```python
# CORRECT — every CRUD function follows this pattern:
async def get_item(self, item_id: str, *, user_id: str = ""):
    query = "SELECT * FROM items WHERE id = ?"
    params = [item_id]
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    ...

# WRONG — leaks cross-tenant:
async def get_item(self, item_id: str):
    return await db.execute("SELECT * FROM items WHERE id = ?", (item_id,))
```

**Cache keys** for stateful handlers must be `(user_id, session_id)`,
never bare `session_id`. See `handler_factory.py:66::resolve_session_keys`.

**New user-scoped tables** declare the FK explicitly:
```sql
user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE
```
ON DELETE CASCADE ensures `SessionManager.delete_user` cleans up cleanly;
the runtime-discovery sweep in `session_manager.py:433` is the backup.

## SSRF substrate

Every server-side outbound URL the user or a community artifact
influences MUST go through `augmentum/utils/safe_http.py::SafeHttpClient`.

```python
from augmentum.utils.safe_http import SafeHttpClient, SafeHttpError

client = SafeHttpClient(max_response_size=64 * 1024)
try:
    text, meta = await client.fetch(url, timeout=10.0)
except SafeHttpError as exc:
    return _error("Couldn't fetch", str(exc))
```

`SafeHttpClient` blocks:
- Non-http(s) schemes
- DNS-resolved IPs in private (RFC1918), loopback, link-local, multicast ranges
- Response bodies over `max_response_size` (both Content-Length pre-check and stream-check)
- DNS rebinding via pinned-IP transport (`_PinnedTransport`)

**Exemptions** (admin-configured destinations, treated like env vars):
- LLM provider base URLs (`audio_routes.py`, `openai_routes.py`, `provider_routes.py`)
- Media server URLs (Audiobookshelf, Emby, Jellyfin, Komga, Suwayomi)
- Coder workspace git clone/push (runs inside container, network-isolated)

All exemptions are documented in
`.claude/skills/augmentum-dev/references/security_exceptions.json` with
explicit IDs the SSRF scanner checks.

**Adding a community-style URL surface** also requires a trusted-origin
allowlist (double-gate). See
[Pattern 26 in `patterns.md`](patterns.md#pattern-26-safehttpclient--trusted-origin-allowlist).

## Fabric trust model

Default OFF (`settings.fabric_enabled` in `config.py:154`). Identity
isn't even generated until you flip it on (`fabric/lifespan.py:43`).

**Pairing** — `POST /api/fabric/pair-with-remote` (admin) initiates an
exchange of signed `PairRequest` envelopes (`peer_auth.py:74-126`).
Envelopes carry `(sender_node_id, hostname, pubkey_b64, fingerprint_hint,
role, timestamp)` with a 300s replay window. Both sides verify, persist
to `fabric_nodes`, echo identity. **Out-of-band fingerprint
confirmation** before pairing is the operator's responsibility — there's
no central PKI.

**Per-peer service users** — `SessionManager.get_or_create_fabric_peer_user`
(`session_manager.py:278`) mints `id = "fabric:<short-node-id>"` lazily
on first signed request. Data isolation moves from per-user to per-peer
on the receiving side — `X-Fabric-User-Id` is informational only.

**Path allowlist** — `peer_middleware.py:95` allows ONLY:
- `/v1/*` (LLM, embeddings, images — for peer dispatch)
- `/api/fabric/*` (control plane)

A signed peer cannot mint `sk-aug-*` API keys, edit settings, install
community items, or hit any other route. Cert pinning at the TLS layer
is **a Phase-1+ follow-up** flagged at `client.py:80` — currently TLS
uses `verify=False` and the trust IS the signed envelope.

**Cloud LLM providers are NOT advertised** over fabric
(`fabric/extractors.py:42-67`) so a peer can't spend another peer's
API budget.

## Cast token model

Cast tokens (`devices/cast_tokens.py`):
- **In-RAM only** (no DB) — restart invalidates everything
- **30-minute TTL**
- **IP-bound** (token + remote IP must match)
- **Single-session revocable** (one mint, one revoke point)
- **32-hex secret**

Used by `/api/media/stream/{file_id}` (`media_routes.py:1758`),
`/api/cast/blob/...`, `/api/cast/render-output/...`. The token IS the
credential — auth middleware exempts these paths.

**Render output store** (`cast/output_store.py`): separate in-RAM bytes
store, TTL 5 minutes, max 256 entries, tokenized. Used for HTML-to-PNG
and VRM frame fetches.

**Couch co-op guest tokens** — the invite token in the request body IS
the credential, scoped to one host's data. Phase 3 device-fingerprint
auto-reconnect uses a localStorage UUID warm-slot reclaim within
`WARM_SLOT_TTL_S = 30.0`.

## Community install trust model

Two routes (`augmentum/proxy/community_routes.py`):

| Route | Auth | Why |
|---|---|---|
| `GET /community-install?manifest_url=...` | **Public** (`_PUBLIC_PATHS`) | Cross-origin nav from augmentumhq.com; handler self-resolves auth and inlines a login form when no session |
| `POST /api/community/install` | **Auth-required** | The actual mutation — auth middleware enforces session |

**Double gate** on URL fetches:
1. Trusted-origin allowlist (`_BUILTIN_TRUSTED_ORIGINS` +
   admin-configurable `community_trusted_origins`)
2. `SafeHttpClient` (SSRF + size cap + rebinding-proof)

**Per-category install** validates the artifact against a category-specific
schema before dispatching:
- `characters`, `reasoning-flows` — per-user (any authenticated user)
- `powers`, `knowledge` — **admin-only** (install-wide impact)

**Audit row** (`community_installs` table, user-scoped, migration 236)
written on every successful install. Failures during audit write must
NOT break the install (`try/except log.warning`).

## API key management

Per-user provider API keys (OpenAI, Anthropic, etc.):
- Stored encrypted at rest with Fernet (AES-128-CBC + HMAC-SHA256) via
  `utils/secrets.py`
- Keyfile auto-generated alongside the DB (`/.secret_key`, gitignored)
- Decrypted only at request time; cleared from scope after dispatch
- Never exposed in GET responses — `to_safe_dict()` redacts any field
  with `"key"`, `"secret"`, `"token"` in the name

Augmentum's own API keys (`sk-aug-*`):
- Hashed before storage (never plaintext in `augmentum_api_keys`)
- One-time display on creation
- Per-user scoped — calling with another user's `sk-aug-*` returns 401

## Sandboxed Python executor

Runs in a separate Docker container — not the augmentum container:
- **Read-only filesystem** (only `/tmp` writable, bind-mounted noexec)
- **Capabilities dropped** (`CapDrop=ALL`)
- **PID limit** (64) prevents fork bombs
- **Memory cap** (512 MB)
- **Separate network** (`augmentum-executor-net`, no access to host services)
- **No-new-privileges**
- **Timeout** per execution (configurable, default 30s)

## Backup posture

`state/backup.py` writes backups via `VACUUM INTO` (not `shutil.copy*` —
a copy captures a torn WAL state).

- File permissions: `chmod 0600`
- Directory: `chmod 0700`
- Inside container only — bind-mount hosts must lock down the host path

**Backups contain password hashes, encrypted API keys, message contents,
memories.** Anyone who can read backups + the keyfile decrypts everything.

## Prompt-injection defense

External content reaching the LLM (memory recall, document chunks,
knowledge passages, web search results, fetched URL bodies, future
email bodies, MCP tool outputs) is **wrapped at the recall/retrieval
boundary** with markers that tell the model the content is *data*,
not *instructions*. Live since Phase 1.1 of the 2026-Q3 build plan;
see `augmentum/security/untrusted.py`.

```
<<<UNTRUSTED:memory/active>>>
... recall content ...
<<<END_UNTRUSTED:memory/active>>>
```

The policy preamble explaining the marker convention is added to the
system message exactly once per turn by
`ensure_policy_in_system(request)` — idempotent across the multiple
subsystems (memory, knowledge, future inbox) that may contribute
untrusted content in the same request. Policy is prepended to the
system message FIRST so it precedes any character/persona prompt by
position (persona cannot override policy through ordering).

**Wrapped surfaces** (Phase 1.1 — landed):

| Surface | Wrap site | Label |
|---|---|---|
| Memory recall | `memory/integration.py::recall_and_inject` | `memory/active` |
| Document RAG chunks | `memory/integration.py::recall_and_inject` | `documents/rag` |
| Knowledge pack passages | `knowledge/injection.py::_prepend_to_system` | `knowledge/pack` |
| Web search (SearXNG) results | `tools/web_search.py::execute` | `web/search` |
| Fetched URL body content | `tools/web_fetch.py::execute` | `web/fetch` |

**Marker-forging defense**: literal `<<<` sequences inside untrusted
content are defanged with a zero-width-space split (`<<​<`). An
attacker quoting a fake closing marker in a fetched page or knowledge
passage cannot reconstruct a clean trigraph — the model never sees an
ambiguous boundary.

**Label sanitization**: only `[a-zA-Z0-9_./-]` survive into the
marker. Attacker-controlled label paths cannot smuggle marker syntax
or control sequences.

**Pattern for new sites**: any new prompt-construction site that
embeds external/user content MUST route through
`augmentum.security.untrusted.wrap_untrusted(label, content)` at the
boundary. Adding a new untrusted surface = pick a stable label
(`<subsystem>/<source>` convention) + wrap at the recall point + the
policy is already in place via the per-turn helper. The test surface
`tests/test_security_untrusted.py` is the contract; extend it when
you add a surface.

## Reserved username defense

Closed since Phase 1.4 of the 2026-Q3 build plan. The centralised list
lives in `augmentum/auth/models.py::RESERVED_USERNAMES` (exact match)
and `RESERVED_USERNAME_PREFIXES` (prefix match — currently
`fabric_peer_`, `fabric:`, `usr_`). The helper
`is_reserved_username(name)` is the single decision point.

Enforced at:

* `augmentum/auth/session_manager.py::create_user` — backstop, raises
  `ValueError` if a route handler forgot to validate up-front.
* `augmentum/auth/session_manager.py::create_first_admin` — same.
* `augmentum/proxy/auth_routes.py::auth_setup` — returns 400 with
  user-facing error.
* `augmentum/proxy/auth_routes.py::auth_register` — returns 400.

The list reserves:

* Role / system names: `system`, `root`, `superuser`, `admin`.
* Service identities: `api`, `service`, `daemon`, `bot`.
* Loopback / elevation tokens (defensive — close the door before any
  future internal-tool elevation pattern is introduced): `internal`,
  `internal-tool`, `internal_tool`, `internaltool`.
* Persona / brand names: `augmentum`, `becca`.
* Anonymous / placeholder: `anonymous`, `guest`, `nobody`, `unknown`.
* Test / demo: `demo`, `test`.
* Fabric-provisioned prefixes (a real account squatting one of these
  would block legitimate peer auth via UNIQUE collision in
  `SessionManager.get_or_create_fabric_peer_user`).

Pinned by `tests/test_security_reserved_usernames.py` — adding or
removing a load-bearing entry should fail the registry-shape tests.

## Known gaps

Defended-but-acknowledged shortfalls. These are tracked separately
from "what we intentionally allow" above — those are deliberate
trade-offs, these are work-in-progress.

| Gap | Risk | Mitigation status |
|---|---|---|
| MCP tool outputs not yet wrapped at source | A compromised or misconfigured external MCP server could return prompt-injection in its tool results | **Open** — wrapping at the MCP dispatch boundary in `augmentum/mcp/` is the next Phase 1.1 follow-up |
| Inbox (email body) content has no wrap path yet | Email is the highest-volume untrusted-content surface in real-world use | **Reserved** — Phase 3.4 (Inbox mode) will wrap email bodies with label `email/body` when the mode lands |
| User-typed notes when injected into prompts have no centralized wrap | A note written by an attacker who briefly had access could persist as a prompt-injection vector | **Open** — wrap at notes-injection boundary when notes start being recalled into prompts |
| Tool-only turns (no memory recall, no knowledge pack) may lack the policy preamble | Markers are still placed but the policy explaining them is absent — model defense is weakened on those turns | **Open** — `ensure_policy_in_system` should also fire at the tool dispatch entry point; the wrap markers alone are still defensive but weaker without the policy framing |
| Pre-existing reserved-name users not retroactively audited | An account named `admin` etc. created before Phase 1.4 landed remains valid; the substrate now refuses to create them, but old installs may already have one | **Open** — operators on multi-user installs should grep `SELECT username FROM users WHERE username IN (...)` against the live DB; rename if found. Augmentum is currently solo-dogfooded so the in-the-wild risk is minimal. |
| No atomic-write helper for the few JSON sidecar files Augmentum still writes (cookbook state, settings backup exports, etc.) | A kill -9 mid-write corrupts the file | **Reserved** — Phase 1.6 will add `augmentum/utils/atomic_io.py` and audit the remaining JSON write sites |
| Job system has no enforced "user always hears back on terminal state" contract | A timed-out or crashed background job can leave the user wondering whether work happened | **Reserved** — Phase 1.5 will add `augmentum/jobs/monitor.py` with explicit terminal-state guarantees |
| Token budget per turn not metered | Becca's context construction grows as the action catalog grows; eventually system + memory + persona + observation + knowledge could eat the budget before the user request lands | **Reserved** — Phase 1.6 will add `augmentum/context/budget.py` BudgetTracker with per-contributor caps and overrun logging |
| Fabric peer cert pinning | Peer pairing currently relies on SSH-style fingerprint pinning over self-signed TLS — Ed25519 envelope IS the auth | **Phase 1+** follow-up, flagged at `fabric/client.py:80`. Documented above as intentional for now. |
| Token scopes are coarse | Companion/mobile tokens carry either `chat` or `admin` scope with no per-capability granularity | **Backlog** — finer scopes when the surface area grows |
| `/api/version` allowlisted in `auth/middleware.py:25` but no handler is registered | Operational inconsistency — `/api/version` returns 404 today. Not a leak (no leak surface = no leak), but the allowlist entry implies a handler exists. | **Open** — either register a handler (current persistence_degraded + app version surface) or remove the allowlist entry. |
| Legacy substring-based redactions (`/api/config/tools` line 928) use `"token" in key.lower()` etc. | False positives on keys like `voice_smart_turn` (none currently — but the pattern is fragile), false negatives on suffix-only keys like `_credential` or `_passwd`. Auth-gated so impact is limited. | **Open** — migrate to `augmentum.security.scrub.is_secret_key` for uniform behavior with the rest of the security stack. |

When fixing a known gap, update this table in the same commit (move
from Open to Reserved when scope is settled, remove when the gap is
closed and tests prove the closure).

## What we intentionally allow (and why)

These are NOT vulnerabilities — they are architectural requirements:

| "Finding" | Why it's intentional | Would break if "fixed" |
|---|---|---|
| CORS `*` origin | Multi-frontend support (Open WebUI, SillyTavern, Cursor, Claude Desktop via MCP) | All external frontends would fail to connect |
| CSP `unsafe-inline` + `unsafe-eval` | Notes editor (Milkdown/CM6), dynamic theming, syntax highlighting, mermaid diagrams | Notes editor, code blocks, themes, diagrams all break |
| `data:` in img-src | Character card avatars stored as base64 data URIs | All character avatars disappear |
| HTTP via 6100 alongside HTTPS via 6443 | Local Docker services + frontends that don't trust self-signed certs | Some local clients can't connect |
| Large request bodies (50MB) | Vision image attachments, character card imports with embedded portraits | Image uploads + card imports break |
| Raw HTML in article reader | Trafilatura output rendered in Browse tab; CSP blocks inline scripts as defense-in-depth | Browse reader shows no content |
| Fabric trust without TLS cert pinning | Ed25519 envelope IS the auth (SSH-style fingerprint pinning); cert pinning Phase-1+ follow-up | Pair attempts fail on every self-signed deploy |
| Public path exemptions for cast pair / render-output / blob | Receiver devices have no session cookie; the token IS the credential | Cast surfaces stop working |
| Coder routes accept clone URLs without SSRF check | Operations run inside Docker workspace network, isolated from host | Coder clone/push break |
| Provider URLs not SSRF-checked | Admin-configured persistent destinations, treated like env vars | Every provider you've configured stops working |

All of these have explicit entries in
`.claude/skills/augmentum-dev/references/security_exceptions.json` with
the reasoning visible to future auditors.

## Patterns you must follow

Every contributor-authored MR touching auth/data/URLs should be checked
against the relevant `patterns.md` entries:

- [Pattern 16 — User-Scoped CRUD](patterns.md#pattern-16-user-scoped-crud-explicit) — every store function
- [Pattern 22 — Surface Exposure declaration](patterns.md#pattern-22-unified-primitive-layer--tool-surface-declaration) — every tool
- [Pattern 23 — Fabric-Aware Backend Resolution](patterns.md#pattern-23-fabric-aware-backend-resolution) — every LLM dispatch
- [Pattern 26 — SafeHttpClient + Trusted Origin Allowlist](patterns.md#pattern-26-safehttpclient--trusted-origin-allowlist) — every outbound URL fetch
- [Pattern 27 — Auth Middleware Exemption + Self-Resolve](patterns.md#pattern-27-auth-middleware-exemption--self-resolve) — every new public path
- [Pattern 28 — Community Install Dispatcher](patterns.md#pattern-28-community-install-dispatcher) — every new community category

## Reporting issues

For security issues, follow `SECURITY.md` (private vulnerability
reporting via GitHub or email). DO NOT open public issues for security
bugs.

For audit script false positives, edit
`.claude/skills/augmentum-dev/references/security_exceptions.json`
with a documented reason — the next audit run will pick it up. Never
suppress real findings; for accepted rolling counts, bump the audit
baseline instead.
