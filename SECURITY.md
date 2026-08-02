# Security Policy

Thanks for taking security seriously. Augmentum is a self-hosted AI platform that runs on user-owned hardware, holds personal data, and exposes a network surface — so we want this document to be honest about what we protect, what we don't (yet), and how to report issues.

## Reporting a vulnerability

**Please don't open public GitHub issues for security bugs.** Use either:

- **GitHub Private Vulnerability Reporting** (preferred): https://github.com/AugmentumHQ/Augmentum/security/advisories/new
- **Email**: `augmentumhq@gmail.com`

When reporting, include:
- A description of the issue
- Steps to reproduce, ideally with a minimal proof of concept
- Impact assessment (what an attacker could do)
- Augmentum version (commit hash or tag)
- Your deployment context (localhost, LAN, internet-exposed, behind reverse proxy, etc.)

**Response timeline:**
- Initial acknowledgement within 72 hours
- Triage + severity assessment within 7 days
- Patch + disclosure within 90 days for confirmed issues, or sooner for actively exploited vulnerabilities

We commit to crediting reporters by name (or anonymously, on request) in release notes.

## Supported versions

Augmentum is pre-1.0; only the latest release on `main` receives security fixes. Once we tag stable releases, this section will list which versions remain supported.

## Threat model

The threat model depends on how you deploy Augmentum. Pick the row that matches yours.

### Tier A: Localhost only (default install)

**Configuration:** `AUGMENTUM_BIND_HOST` unset or `127.0.0.1`. The container's published ports are reachable only from the host machine. Other devices on your LAN cannot connect.

**In scope:**
- Other user accounts on the same machine that can reach loopback
- Malware running as your user that can talk to localhost services
- Browser-based attacks: malicious websites in your browser executing requests against `localhost:6100`

**Out of scope:**
- Network attackers (they cannot reach the service)
- Physical access (out of scope for any software)

**This is the recommended configuration for most users.**

### Tier B: LAN-exposed (`AUGMENTUM_BIND_HOST=0.0.0.0`)

**Configuration:** Augmentum is published on every host interface. Anyone on your local network can reach it.

**Adds to threat model:**
- Other devices on your WiFi (your phone is fine, but so is anyone else connected)
- Wired LAN neighbours, IoT devices, guests on a shared network
- During first install: a "first-user-wins" race where someone could claim the admin slot before you do — register your account immediately on install

**Mitigations Augmentum provides:**
- Argon2id password hashing
- Rate-limited login (token bucket per IP and username)
- Opaque session tokens, not JWTs
- Multi-tenant data isolation (every user-scoped table requires `user_id`)
- Sandboxed Python executor (read-only FS, dropped capabilities, memory cap)
- HTTPS via Caddy when `COMPOSE_PROFILES=https`

**You are responsible for:**
- Trusting your LAN (don't expose on coworking / hostel / public WiFi)
- Registering the admin account before any other device connects
- Keeping your machine's OS and Docker patched

### Tier C: Internet-exposed (port-forwarded, public reverse proxy)

**Not recommended without expert hardening.** Augmentum is not yet audited at the level appropriate for direct internet exposure.

If you must:
- Always front with HTTPS (Caddy/Traefik/nginx with valid cert)
- Restrict by IP (allowlist your home IP, geo-block) or use a VPN/Tailscale gateway
- Subscribe to release notes and patch promptly
- Treat your install as "I am running my own production service" — backup, monitor, alert
- Consider running auth-only WAF rules on the reverse proxy

## What Augmentum protects

| Surface | Mechanism |
|---|---|
| Authentication | Argon2id passwords, opaque session tokens, fail-closed middleware (no auth = 503/WS-close) |
| Login brute-force | Per-IP and per-username rate limiting |
| Multi-tenant data | `user_id` scoping enforced across 160+ user-scoped tables, both at the route layer and at the storage layer (e.g. coder profile CRUD); cross-tenant leak audits closed in 2026-04 |
| CSRF | Global Origin/Referer middleware on all state-changing routes (`_CSRFOriginMiddleware` in `proxy/server.py`). Bearer-token API requests and the unauthenticated setup/login/status routes are the only exemptions |
| SSRF | All outbound URL fetches go through `SafeHttpClient` (`utils/safe_http.py`), which rejects loopback / RFC1918 / link-local / multicast targets. Admin-configured destinations (LLM providers, media servers) are exempt by design |
| API keys at rest | Encrypted with Fernet (AES-128-CBC + HMAC-SHA256) via `utils/secrets.py`. Keyfile auto-generated alongside the database |
| Backups | `state/backup.py` writes via `VACUUM INTO`, then `chmod 0600` on the file and `chmod 0700` on `/data/backups/`. Bind-mount hosts must still restrict the underlying host path |
| Log exposure | `RichTracebackFormatter` is configured with `show_locals=False` so request bodies don't leak into 500 stack traces. Admin UI can change the runtime log level without a restart |
| Python execution | Containerized executor: read-only FS, dropped capabilities, 64-PID limit, 512 MB memory cap, separate network |
| Coder workspace container | All dangerous Linux capabilities dropped, `no-new-privileges:true`, per-workspace isolation. Outbound network egress is still open — see Known Limitations |
| Docker socket | Augmentum container does **not** mount the raw Docker socket. All Docker API calls flow through `tecnativa/docker-socket-proxy` with an explicit allowlist |
| Local network exposure | Default install binds to `127.0.0.1`; LAN/WAN exposure is opt-in via `AUGMENTUM_BIND_HOST` |
| WebSocket flooding | Per-frame size limit on WS upgrades |
| Privacy | No telemetry, no analytics, no phone-home. Local-first. Data leaves the machine only when the user configures a cloud provider (LLM, image gen, STT) |

## Known limitations (we're working on these)

These are documented gaps with planned fixes in flight. See the project's
hardening roadmap at the project's internal hardening roadmap
for the full tier list and target releases.

- **CSRF cookie hardening.** Origin/Referer validation ships globally via
  middleware (see the protections table above), but session cookies do not
  yet carry an explicit `SameSite=Strict` directive — browsers fall back to
  their defaults, which is fine in practice but not what we want to rely on.
  Tier-1.
- **SSRF allowlist for admin-configured destinations.** `SafeHttpClient`
  blocks the obvious internal-network targets, but URLs the admin types in
  (provider base URLs, media servers, knowledge-pack sources) are trusted
  by design. A first-party domain allowlist with per-route exceptions would
  be stronger. Tier-2.
- **Prompt injection → tool abuse.** An LLM that processes attacker-
  controlled text (a malicious document, a hostile web page) can be coaxed
  into using tools (`web_fetch`, `file_ops`, `python_exec`) in ways the
  user didn't intend. The current containment is structural — the Python
  executor is sandboxed, web tools return sanitized plain text, file ops
  are scoped — but there is no content-source-aware gating on the tool
  call itself. Mitigation is in design.
- **Coder workspace egress.** Workspace containers drop capabilities and
  set `no-new-privileges`, but `NetworkMode=bridge` leaves outbound
  internet access wide open. Egress filtering and a seccomp profile are
  Tier-2 work.
- **Long-lived stream auth revocation lag.** SSE and WebSocket endpoints
  (bug-finder events, coder run streams, voice fanout) validate auth +
  ownership at connect time and then run a bounded generator. If an
  admin disables an account mid-stream, the user's existing connection
  finishes the current run rather than being torn down immediately.
  Streams are user-scoped (no cross-tenant leak) and bounded (typically
  seconds to minutes), so the practical exposure is "one more run's
  worth of output after revocation." Per-frame re-validation on long
  streams is on the post-1.0 list.
- **Log content at `INFO` level.** Even with `show_locals=False`, INFO logs
  can still carry user content via explicit `log.info(...)` calls that
  include message previews or transcripts. Default Docker log retention
  is local; we still recommend `LOG_LEVEL=WARNING` for production-style
  installs.
- **Backup-file host permissions.** Inside the container, backups are
  `chmod 0600` and the directory is `0700`. If you bind-mount `/data` to
  a host path, the host directory is yours to lock down — these are full
  database snapshots with password hashes, API keys (still encrypted at
  rest, but a leaked keyfile + leaked DB = decryptable), and message
  contents.
- **Dependency supply chain.** `pyproject.toml` pins versions but not
  hashes; Dockerfiles use floating tags (`python:3.11-slim-bookworm`,
  `ubuntu:24.04`) rather than `@sha256:` digests. Vendoring + digest
  pinning is on the pre-launch roadmap. Tier-1.

## Privacy posture

- **No telemetry.** Augmentum makes no outbound calls except to the providers you explicitly configure.
- **No analytics.** No tracking pixels, no error reporting service, no usage data shipped anywhere.
- **No update checks** that phone home.
- **All conversation data, memories, character cards, documents, and dream content** stay on disk in your `/data` volume. They never leave unless you explicitly use a cloud LLM/image/STT provider.
- **When you DO use a cloud provider** (OpenAI, Anthropic, Stability, Deepgram, etc.), Augmentum sends only what's needed for the request — but that data is then subject to the provider's retention policy. Each provider's policy is documented at the provider's own site; we recommend reviewing before you configure cloud keys.

A more detailed privacy/data-flow map per feature lives at [`docs/PRIVACY.md`](docs/PRIVACY.md).

## Out-of-scope

The following are not Augmentum security issues, even if they cause problems:

- Bugs in upstream models or providers (OpenAI, Anthropic, etc.)
- Browser quirks or extensions interfering with the UI
- User-misconfigured deployments (e.g., binding to public IP without TLS)
- Issues in third-party Docker images we orchestrate (Ollama, Speaches, Chatterbox) unless caused by our integration
- Resource exhaustion attacks where the attacker is already authenticated as an admin
- Content output by LLMs (these are language models, they say wrong things)

Bugs that do qualify, even if outside the typical pattern:
- Authentication or authorization bypass
- Cross-tenant data leakage
- Remote code execution or container escape
- Sensitive data logged in plaintext
- TLS misconfiguration shipped by default
- Privilege escalation between user roles

## A request

If you're going to test Augmentum's security:
- Test on your own install. Don't probe other people's deployments.
- Don't run automated scans against any AugmentumHQ-hosted infrastructure (we don't run any yet, but as a future-proofing note).
- If you find something interesting and want to verify with us before publishing, the email above is open.

Thanks for keeping people's data safe.
