# Media-server integration docs (local corpus)

Authoritative, **locally-pulled** API documentation for every media server
Augmentum can one-click provision (Discover → Media servers). We integrate
**by reading these files**, not by web search or model memory — so every
endpoint, payload field, and port we depend on is verifiable against the
real spec in-repo.

When adding or deepening an integration:

1. Read the relevant file here (OpenAPI JSON, GraphQL/config docs, or the
   vendored server source) — not the web, not training data.
2. Cross-check our provider code (`augmentum/media/providers/<name>.py`),
   the catalog entry (`augmentum/providers/catalog.json`), and the
   first-run bootstrap against it.
3. If you bump the pinned image, re-pull the corresponding spec (commands
   in each **Source** section below) and re-verify.

## What's here

| Service | File(s) | Source | Pulled | Spec version |
|---|---|---|---|---|
| **Jellyfin** | `jellyfin/openapi.json` (2.0 MB) | `https://api.jellyfin.org/openapi/jellyfin-openapi-stable.json` | 2026-06-18 | OpenAPI "stable" |
| **Komga** | `komga/openapi.json` (172 KB) | `https://demo.komga.org/v3/api-docs` (live springdoc) | 2026-06-18 | 1.24.4 |
| **Audiobookshelf** | `audiobookshelf/api-*.md` (Slate docs) + `server-source-{Server,ApiRouter}.js` | `audiobookshelf/audiobookshelf-api-docs` + `advplyr/audiobookshelf` (server source) | 2026-06-18 | master |
| **Suwayomi** | `suwayomi/configuring-suwayomi-server.md` (24 KB) | `Suwayomi-Server` wiki: *Configuring Suwayomi‑Server* | 2026-06-18 | wiki master |

### Per-service refresh commands

```bash
base="docs/integrations/media-servers"
# Jellyfin
curl -sL "https://api.jellyfin.org/openapi/jellyfin-openapi-stable.json" -o "$base/jellyfin/openapi.json"
# Komga (any reachable instance's springdoc works; demo is convenient)
curl -sL "https://demo.komga.org/v3/api-docs" -o "$base/komga/openapi.json"
# Suwayomi config reference (wiki)
curl -sL "https://raw.githubusercontent.com/wiki/Suwayomi/Suwayomi-Server/Configuring-Suwayomi%E2%80%90Server.md" -o "$base/suwayomi/configuring-suwayomi-server.md"
# Audiobookshelf docs (Slate includes) + authoritative server source
for inc in server libraries items me podcasts schemas users sessions search; do
  curl -sL "https://raw.githubusercontent.com/audiobookshelf/audiobookshelf-api-docs/main/source/includes/_${inc}.md" -o "$base/audiobookshelf/api-${inc}.md"; done
curl -sL "https://raw.githubusercontent.com/advplyr/audiobookshelf/master/server/Server.js" -o "$base/audiobookshelf/server-source-Server.js"
curl -sL "https://raw.githubusercontent.com/advplyr/audiobookshelf/master/server/routers/ApiRouter.js" -o "$base/audiobookshelf/server-source-ApiRouter.js"
```

> **Notes on coverage gaps.** Suwayomi's GraphQL API has no committed SDL —
> it's generated at runtime by graphql-kotlin and introspectable from a
> running instance at `/api/graphql`. Pull it with an introspection query
> once an instance is up; the config-reference wiki (provisioning, auth,
> `webUISubpath`, ports) is what we needed for the install path. Komga's
> Spring Boot **actuator** endpoints aren't part of the springdoc OpenAPI
> (only `/actuator/info` shows) — that's expected, not missing.

## Verification log (2026-06-18 — all four checked against the files above)

Provisioning + auto-connect integration verified endpoint-by-endpoint:

- **Jellyfin** ✅ — `/Startup/Configuration`, `/Startup/User` (GET+POST),
  `/Startup/RemoteAccess`, `/Startup/Complete`, `/System/Info/Public`
  (`StartupWizardCompleted`), `/Users/AuthenticateByName` all present.
  DTO fields match: `StartupUserDto{Name,Password}`,
  `StartupConfigurationDto{UICulture,MetadataCountryCode,PreferredMetadataLanguage}`,
  `StartupRemoteAccessDto{EnableRemoteAccess,EnableAutomaticPortMapping}`,
  and login uses `AuthenticateUserByName{Username,Pw}` (note: **`Pw`**, not
  `Password` — the existing provider already does this correctly).
- **Komga** ✅ (with a fix) — `GET/POST /api/v1/claim` with
  `X-Komga-Email`/`X-Komga-Password` header params confirmed.
  **BUG CAUGHT + FIXED:** the spec (1.24.4) has **no `/api/v1/users*`
  paths** — they moved to `/api/v2`. `KomgaProvider.login`/`verify_token`
  validated against `/api/v1/users/me`, which **404s on the `:latest` image
  we provision** → login would have failed. Now tries `/api/v2/users/me`
  first, falls back to `/api/v1/users/me` for older servers (`_ME_PATHS`).
  Minor (non-blocking): `ping` calls `/api/v2/actuator/info` but the
  actuator lives at `/actuator/info`; the `/api/v1/claim` fallback masks
  it, left as-is.
- **Audiobookshelf** ✅ — `POST /init` body
  `{"newRoot":{"username","password"}}` matches the docs exactly
  (`server-source-Server.js` line 433: `req.body.newRoot`,
  `newRoot.username`, hashed `newRoot.password`); `GET /status` →
  `{"isInit":...}`; `POST /login` → `user.token`. **Port:** the official
  `ghcr.io/advplyr/audiobookshelf` image sets `ENV PORT=80` / `EXPOSE 80`,
  so the catalog's `internal_port: 80` is correct (not the conventional
  host port 13378).
- **Suwayomi** ✅ — `server.authMode = basic_auth` with
  `server.authUsername`/`server.authPassword` (env `AUTH_MODE`/
  `AUTH_USERNAME`/`AUTH_PASSWORD` per the image startup script);
  `server.port = 4567`; `server.webUISubpath` exists (the basis for the
  isolate-next console proxy).
