# Emby, Jellyfin, and Plex API Research

Generated on 2026-04-22 from official vendor sources.

## Generated Trees

- [emby-api-tree.txt](./emby-api-tree.txt): 535 operations across 70 service groups.
- [jellyfin-api-tree.txt](./jellyfin-api-tree.txt): 388 operations across 61 tag groups.
- [plex-api-tree.txt](./plex-api-tree.txt): 250 operations across 29 tag groups.
- [generate_media_server_api_research.py](../../scripts/generate_media_server_api_research.py): reproducible generator used for the three trees above.

## What Matters For Augmentum

- Emby and Jellyfin are the closest fit to the current `MediaProvider` abstraction in `augmentum/media/providers/base.py`. Their login, library-view, item-query, image, playback-info, and session-progress flows line up closely.
- Plex now has an official developer surface at `developer.plex.tv/pms/`. Its modern docs push clients toward `/media/providers` as well as the older `/library/sections/*` surface.
- Plex defaults to XML unless you explicitly send `Accept: application/json`. Emby documents both JSON and XML. Jellyfin's published OpenAPI is JSON-first.
- Stable client identity matters on all three. Persist a device/client identifier and send it on every request.
- Zero-auth probing is easier on Emby and Jellyfin because both publish public system-info endpoints. Plex detection is better handled via GDM or after the user supplies a base URL and token.

## Ports and Discovery

| Server | Default HTTP | Default HTTPS | Discovery / LAN ports | Notes |
| --- | --- | --- | --- | --- |
| Emby | `8096/TCP` | `8920/TCP` | `7359/UDP` | Emby documents UDP `7359` for server discovery by client apps. |
| Jellyfin | `8096/TCP` | `8920/TCP` | `7359/UDP` | Jellyfin documents a broadcast to `7359/UDP` returning server name, IP, and ID. |
| Plex | `32400/TCP` | Configurable secure mode on PMS | `1900/UDP`, `5353/UDP`, `8324/TCP`, `32410/UDP`, `32412/UDP`, `32413/UDP`, `32414/UDP`, `32469/TCP` | GDM discovery uses the `32410/12/13/14` UDP ports. Plex explicitly warns against exposing the extra LAN-only ports to the WAN. |

## Auth and Request Conventions

| Server | Base URL shape | Login / token acquisition | Per-request auth | Important calling notes |
| --- | --- | --- | --- | --- |
| Emby | `http[s]://host:port/emby/{apipath}` | `GET /Users/Public`, then `POST /Users/AuthenticateByName` | `Authorization: Emby ...` or `X-Emby-Authorization`, then `X-Emby-Token`; server API keys can also be sent as `api_key` query parameter | Supports both JSON and XML. API keys are the better fit for server-to-server integrations. |
| Jellyfin | Server base URL plus published OpenAPI paths such as `/System/Info/Public` and `/Users/AuthenticateByName` | `GET /Users/Public`, then `POST /Users/AuthenticateByName` | Official SDK builds `Authorization: MediaBrowser Client="...", Device="...", DeviceId="...", Version="...", Token="..."` | Root-path usage is an inference from the published OpenAPI paths. Use a stable `DeviceId`. |
| Plex | PMS root, e.g. `http://host:32400/` or the `plex.direct` URL shape in the official developer docs | Token is obtained from Plex account auth; older support docs show temporary token discovery via Plex Web App | Send `X-Plex-Token`; also send `X-Plex-Client-Identifier` and `X-Plex-Product` at minimum | Send `Accept: application/json` if you want JSON. The docs also expose `X-Plex-Pms-Api-Version`. Prefer headers over query-string tokens except when you have to hand a raw URL to a player/browser. |

## Recommended Calling Practices

- Normalize and store the user-entered base URL without a trailing slash.
- Emby requires the `/emby` prefix. Jellyfin does not expose that prefix in its published stable OpenAPI.
- Prefer HTTPS when the server has it enabled, but do not assume the HTTPS port is active by default on Emby or Jellyfin.
- Persist one stable `device_id` per Augmentum installation and reuse it across sessions.
- Prefer header tokens over query tokens for normal API traffic.
- For Plex, always request JSON explicitly and follow returned `key` paths instead of hardcoding only one traversal style.
- For Emby and Jellyfin, build one shared code path where possible and branch only on the places where paths or auth header names differ.

## Canonical Integration Traces

### Emby

1. Probe server: `GET /emby/System/Info/Public`
2. Enumerate visible users if needed: `GET /emby/Users/Public`
3. Authenticate: `POST /emby/Users/AuthenticateByName`
4. Fetch top-level views: `GET /emby/Users/{UserId}/Views`
5. Enumerate catalog: `GET /emby/Users/{UserId}/Items` and `GET /emby/Users/{UserId}/Items/Resume`
6. Fetch item detail: `GET /emby/Users/{UserId}/Items/{Id}`
7. Fetch art: `GET /emby/Items/{Id}/Images/{Type}`
8. Resolve stream/transcode options: `GET` or `POST /emby/Items/{Id}/PlaybackInfo`
9. Play: `GET /emby/Audio/{Id}/stream` or the relevant video/HLS endpoint family
10. Report progress: `POST /emby/Sessions/Playing`, `POST /emby/Sessions/Playing/Progress`, `POST /emby/Sessions/Playing/Stopped`
11. Persist watched/favorite state when needed: `POST /emby/Users/{UserId}/Items/{ItemId}/UserData`

### Jellyfin

1. Optional LAN discovery: UDP broadcast to `7359`
2. Probe server: `GET /System/Info/Public`
3. Enumerate visible users if needed: `GET /Users/Public`
4. Authenticate: `POST /Users/AuthenticateByName`
5. Fetch top-level views: `GET /UserViews` or `GET /Items/Root`
6. Enumerate catalog: `GET /Items`, `GET /Items/Latest`, `GET /UserItems/Resume`
7. Fetch item detail: `GET /Items/{itemId}`
8. Fetch art: `GET /Items/{itemId}/Images/{imageType}`
9. Resolve stream/transcode options: `GET` or `POST /Items/{itemId}/PlaybackInfo`
10. Play: `GET /Audio/{itemId}/stream`, `GET /Videos/{itemId}/stream`, or the HLS families
11. Report progress: `POST /Sessions/Playing`, `POST /Sessions/Playing/Progress`, `POST /Sessions/Playing/Stopped`
12. Persist watched/favorite/progress state when needed: `GET` or `POST /UserItems/{itemId}/UserData`

### Plex

1. Optional LAN discovery: GDM on UDP `32410/32412/32413/32414`
2. Probe server capabilities: `GET /` with `Accept: application/json`
3. Enumerate providers: `GET /media/providers`
4. Fall back to classic libraries when needed: `GET /library/sections/all`
5. Enumerate section contents: `GET /library/sections/{sectionId}/all`
6. Fetch item detail: `GET /library/metadata/{ids}`
7. Inspect current playback: `GET /status/sessions`
8. Update timeline/progress: `POST /:/timeline`
9. Mark watched / unwatched: `PUT /:/scrobble`, `PUT /:/unscrobble`
10. Resolve playback decision and transcode path: `GET /{transcodeType}/:/transcode/universal/decision`

## Mapping To Augmentum's Current Provider Interface

| Augmentum method | Emby | Jellyfin | Plex |
| --- | --- | --- | --- |
| `ping()` | `GET /emby/System/Info/Public` | `GET /System/Info/Public` | `GET /` with token and JSON `Accept` header |
| `login()` | `POST /emby/Users/AuthenticateByName` | `POST /Users/AuthenticateByName` | External Plex auth/token flow, then PMS verification via `GET /` |
| `fetch_catalog()` | `GET /emby/Users/{UserId}/Items` | `GET /Items` | `GET /media/providers`, `GET /library/sections/all`, then section traversal |
| `build_cover_url()` | `/emby/Items/{Id}/Images/{Type}` | `/Items/{itemId}/Images/{imageType}` | Returned art/thumb keys plus PMS base URL |
| `build_stream_url()` | `/emby/Audio/*`, `/emby/Videos/*`, playback-info-assisted | `/Audio/*`, `/Videos/*`, playback-info-assisted | Direct-part URLs or transcode decision/start flow |
| `fetch_progress()` | Session and user-data endpoints | Session and user-data endpoints | `/status/sessions` plus timeline history |
| `push_progress()` | `/Sessions/Playing/*` | `/Sessions/Playing/*` | `/:/timeline`, optionally scrobble/unscrobble |

## Source Notes

- Emby and Jellyfin remain structurally similar because Jellyfin descends from an earlier Emby release. The overlap is visible in path names, playback APIs, and session reporting endpoints.
- Plex is the least drop-in with the current Augmentum adapter shape. Its official modern direction is provider-driven and more feature-gated than the Emby/Jellyfin style.
- If the first expansion target is "books, audiobooks, comics, then broader media later", Jellyfin is likely the cheapest second implementation after Emby, while Plex will need its own adapter assumptions around auth, content negotiation, and key-following.

## Official Sources

- Emby REST API overview: <https://dev.emby.media/doc/restapi/index.html>
- Emby API key auth: <https://dev.emby.media/doc/restapi/API-Key-Authentication.html>
- Emby user auth: <https://dev.emby.media/doc/restapi/User-Authentication.html>
- Emby REST reference index: <https://dev.emby.media/reference/RestAPI.html>
- Emby connectivity and ports: <https://support.emby.media/support/articles/Connectivity.html>
- Jellyfin networking and ports: <https://jellyfin.org/docs/general/post-install/networking/>
- Jellyfin stable OpenAPI index: <https://api.jellyfin.org/openapi/>
- Jellyfin stable OpenAPI JSON: <https://api.jellyfin.org/openapi/jellyfin-openapi-stable.json>
- Jellyfin auth guide: <https://kotlin-sdk.jellyfin.org/guide/authentication.html>
- Jellyfin auth header implementation: <https://github.com/jellyfin/jellyfin-sdk-typescript/blob/efc9447da4d274de96c4dd52908f7d204174a92b/src/utils/authentication.ts>
- Plex official PMS developer docs: <https://developer.plex.tv/pms/>
- Plex ports and firewall guidance: <https://support.plex.tv/articles/201543147-what-network-ports-do-i-need-to-allow-through-my-firewall/>
- Plex token guidance: <https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/>
- Plex network settings and secure-connections guidance: <https://support.plex.tv/articles/200430283-network/>
