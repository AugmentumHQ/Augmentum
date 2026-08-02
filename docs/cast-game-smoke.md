# Cast-Game live smoke checklist

End-to-end verification for the browser-cast pipeline. Run on a real
stack with at least one paired receiver (`/ui/cast-pair/` from the
TV browser) and one phone-as-controller (`/ui/cast-control/`).

Substrate covered by automated tests (`tests/test_cast_input_registry.py`
+ `tests/test_cast_input_bridge_routes.py::TestBrowserCastWS`
+ `tests/test_cast_games_registry.py`
+ `tests/test_cast_games_routes.py`
+ `tests/test_cast_games_proxy.py`
+ `tests/test_cast_games_proxy_routes.py`
+ `tests/test_cast_games_telemetry.py`
+ `tests/test_cast_input_adapters_js.mjs`); this checklist exercises
the visual + browser-side handoffs the unit tests can't reach.

As of 2026-06-05, the cast button calls
`POST /api/cast/games/{id}/classify` first to pick a strategy +
adapter chain per-(user, title), so cross-origin games now reach
controller input via Strategy 2 (origin proxy). See spec
`docs/superpowers/specs/2026-06-04-universal-cast-pipeline-design.md`
for the architecture.

As of 2026-06-23 (Phase 4 telemetry half), the cast surface emits an
`augmentum.input_telemetry` tick every ~5s; the server's
`TelemetryDemoter` watches for a cheap-`shim` cast that has input
flowing (`frames_received`) but an `unreachable_targets > 0`
cross-origin iframe, and stamps the title's profile `failed_at` so the
NEXT cast auto-promotes shim → proxy. Section 5 below verifies this on
a real TV + controller. (The proactive Playwright probe — first-cast
classification without the round-trip — is the remaining Phase 4 piece
and is not yet built.)

## Prerequisites

1. Stack up: `./start.sh -d` (or `start.bat -d`).
2. TV browser at `https://<host>:6443/ui/cast-pair/` paired and showing
   the receiver home (`cast-home`). (The installed Android TV APK is
   already paired — just bring it back to the home surface.)
3. Phone/host signed in, at `/ui/library/`.
4. A second tab or second device acting as cast-controller phone at
   `/ui/cast-control/` is OPTIONAL for sections 1–4 — the library cast
   button does the send directly from the host browser. Section 5
   (telemetry demotion) REQUIRES a real controller pushing input.

> Dev loop: server-side changes here load on `./start.sh restart
> augmentum` (no rebuild). UI changes (the loader, cast-launch.js) need
> a hard refresh of the cast surface on the TV.

## Four game flavors

For each, the loop is:

1. Find the game in library2.
2. Click the cast (📺) icon in the detail-pane secondary actions.
3. Pick the receiver from the picker.
4. Expect: TV shows the game kiosk-style (no Augmentum chrome), toast
   on phone "Now playing on <Receiver>".
5. From a paired phone-as-controller, press a button: the game on
   the TV reacts.

### 1. Browser-runnable emulator ROM (NES / SNES / GBA / GB / Genesis)

- Pin a small ROM (smallest you have — Mario, Sonic, etc.) on library2.
- Cast it.
- TV shows EmulatorJS fullscreen, no settings/save/load chrome, no AI
  panel. Game boots.
- Phone gamepad button → emulator reacts (A button = pressing the
  ROM's A-equivalent).

### 2. Streamed game (Dolphin / PCSX2 / future AGSP profile)

- Find a streaming-profile-backed title (any ROM with system_id whose
  `streaming_profile` is set — GameCube / Wii today).
- Cast it.
- TV shows the Selkies WebRTC viewer fullscreen — no chrome, no
  Settings, no Close. (Game-stream container starts server-side as
  usual; the receiver is just rendering the stream URL.)
- Phone gamepad input flows into the container's UInput pad via the
  EXISTING container path (not via this new browser-cast path).
- Note: this flavor exercises the kiosk-mode flag in `stream-stage.js`;
  input is unchanged from the regular streamed-cast flow.

### 3. js13k game (Space Huggers, etc.)

- Browse js13k via the library2 Games tab → pin one.
- Cast it from the detail-pane.
- TV shows the game's iframe fullscreen. The classifier picks
  Strategy 2 (origin proxy) — the game is fetched through
  `/api/cast/game-proxy/<token>/...` so it's same-origin to the
  receiver, and the universal-input-adapter loader is injected at
  the top of `<head>` automatically.
- Phone gamepad button → game reacts. For games that listen for
  keyboard events instead of Gamepad API, pin an `input_chain`
  override via:
  `PUT /api/cast/games/{title_id}/profile {"input_chain":["keyboard"]}`
  (a UI for this is in spec but not yet built — Phase 4 of the
  universal cast pipeline spec). Default chain is `['gamepad_api']`;
  the keyboard adapter is opt-in per game.

### 4. Marketplace / curated web game (2048, A Dark Room, etc.)

- Same as #3. Marketplace entries on vendor domains ride Strategy 2.
- Patatap, Bemuse, HexGL — verify they LOAD on the TV and react to
  gamepad input (Patatap/HexGL via gamepad_api; A Dark Room via the
  `keyboard` adapter override).
- Service workers are intentionally disabled in proxied games (the
  loader's boot script overrides `navigator.serviceWorker.register`)
  so games that hard-depend on SW caching may degrade — track via
  the `cast_proxy_unrewritten_cross_origin_url` log entry.

### 5. Telemetry-driven auto-demotion (Phase 4, server side)

This proves the loop end-to-end: a cross-origin game cast on the cheap
shim strategy detects that input can't reach it and auto-promotes to the
proxy on the next cast — no manual override.

Pick a game whose embed is **cross-origin** to the server (a js13k /
GitHub-Pages / marketplace-vendor URL — NOT a same-origin emulator ROM).

1. **Force the shim** so we can watch the demotion (otherwise the
   classifier may already pick proxy). Either delete any saved profile:
   `DELETE /api/cast/games/{title_id}/profile`, or pin shim explicitly:
   `PUT /api/cast/games/{title_id}/profile {"strategy":"shim"}`.
2. Cast the game from `/ui/library/`. The TV loads it in an iframe; the
   game is on its own origin, so the gamepad shim can't reach it.
3. On the TV receiver console, confirm the surface is on shim and sees an
   unreachable iframe:
   `__augCastAdapter.context()` → `{strategy: 'shim', title_id: '…'}`
   `__augCastAdapter.targets()` → `{reachable: 0, unreachable: 1}`
4. **Push your controller for ~20 seconds.** The game does NOT react
   (expected on shim + cross-origin). The loader emits telemetry every
   ~5s with `frames_received` climbing and `unreachable_targets: 1`.
5. Watch the server log for `cast_telemetry_demotion_recorded`
   (`from_strategy=shim`). The profile's `failed_at` is now stamped.
6. **Re-cast the same game.** This time the classifier promotes to proxy:
   the surface URL is `/api/cast/game-proxy/<token>/…`,
   `__augCastAdapter.context().strategy` → `'proxy'`, and
   `__augCastAdapter.targets()` → `{reachable: 1, unreachable: 0}`
   (the game is now same-origin to the receiver).
7. Push the controller again — **the game reacts.** Loop proven.

If it doesn't demote: confirm `frames_received` is actually climbing
(controller really connected via `/ui/cast-control/`), that
`unreachable_targets` is `1` (truly cross-origin embed), and that the
server booted the demoter (`cast_profile_registry_initialized`).

## What to look for on failure

| Symptom | Likely cause |
|---|---|
| Picker says "No TVs connected" | Receiver dropped its WS. Refresh `/ui/cast-receiver/` on the TV. |
| Toast says "Cast failed: HTTP 404" | Receiver registered but WS closed mid-send. Same fix. |
| TV shows the play URL but iframe is blank | CSP frame-src doesn't whitelist the embed_url's origin. Check browser DevTools on the TV. |
| TV shows iframe but no input (cross-origin) | Expected on the shim strategy. `__augCastAdapter.targets()` shows `{unreachable: 1}`. Push the controller ~20s and re-cast — the telemetry demoter promotes to proxy (section 5). Or force it now: `PUT /api/cast/games/{id}/profile {"strategy":"proxy"}`. |
| TV shows iframe but no input (same-origin) | The active adapter doesn't match what the game listens for. `__augCastAdapter.active()` shows the chain; override via `PUT /api/cast/games/{id}/profile {"input_chain":["keyboard"]}` (or `touch`/`pointer`). |
| Cross-origin game shows blank iframe | Check browser DevTools for CSP frame-src failures. The classifier may have fallen back to Strategy 1 (shim) instead of Strategy 2 (proxy) — confirm the proxy session_store was wired (server log `cast_profile_registry_initialized` + the proxy fetcher init). |
| Proxied game's service worker errors in console | Expected — the loader's boot script intentionally disables `navigator.serviceWorker.register` (suppressed with a warning). Games that hard-depend on SW caching for assets may degrade; track which games this affects in the quirks table follow-up. |
| TV shows "Loading…" forever | Backend `/api/titles/{id}` 4xx — check server logs. |

## Manual cleanup

The cast survives both phone reload and the host browser closing — the
receiver keeps the iframe up until you re-cast (replaces) or close
the receiver tab.
