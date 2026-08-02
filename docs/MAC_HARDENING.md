# Mac Hardening Punch List

Audit conducted 2026-06-04. Captures every macOS support failure point that
surfaced from a read-only sweep of the codebase (no live Mac available to
runtime-test against).

**Headline assessment:** architecturally sound — no "doesn't fundamentally
work on Mac" issues. Failure mode is *silent degradation*, not crashes. The
Docker-on-Colima CPU path is essentially clean today; the hardening
targets close the gaps for (a) GPU-mode-on-Colima silent fallback and
(b) running natively (outside Docker) on Apple Silicon.

Items are ordered by what bites a Mac user first. Check off as we land
each fix.

---

## P0 — bites on day 1, no error message to lead them to the fix

- [x] **P0-1 · GPU passthrough silently fails under Colima** *(landed — `hardware.py` emits `gpu_requested_but_unavailable` warning with a hint when `AUGMENTUM_VARIANT=gpu` is set but neither CUDA nor Apple Silicon Metal is detected)*

- [x] **P0-2 · `stat -c%s` is GNU-only** *(landed — Dockerfile:98, Dockerfile.gpu:136, Dockerfile.gpu:168 all switched to `wc -c < file`)*
  - Files: `Dockerfile:98`, `Dockerfile.gpu:136`, `Dockerfile.gpu:168`
  - Concern: BSD `stat` (macOS) uses `-f %z`. The `2>/dev/null || echo 0`
    fallback masks the failure as "0 bytes", which silently passes the
    "remove if truncated" check. Broken downloads slip through on any
    macOS-based build path (Colima native, future).
  - Fix: replace `stat -c%s "$file"` with `wc -c < "$file"`. POSIX,
    works everywhere.

---

## P1 — bites them if they step outside the Docker happy path (native run, Apple Silicon)

- [x] **P1-3 · `/usr/local/bin/llama-server` hardcoded** *(landed — `server.py` engine v2 auto-detection now falls back to `shutil.which("llama-server")` when the configured path is missing; same fallback wired into `ClassifierSibling.start()`. The dataclass defaults stay as `/usr/local/bin/llama-server` for the Docker container case; PATH lookup kicks in only when the default doesn't resolve.)*

- [x] **P1-4 · `/home/augmentum/.{kokoro,dtln}` defaults** *(landed — new shared helper `augmentum/utils/paths.py::resolve_model_dir(name)` resolves to the Docker path if present, else `~/Library/Caches/augmentum/<name>` on macOS, `%LOCALAPPDATA%\augmentum\<name>` on Windows, `~/.cache/augmentum/<name>` on Linux native. `kokoro_tts.py` and `denoiser.py` now use it.)*

---

## P2 — felt over weeks, not day 1

- [x] **P2-5 · No `torch.backends.mps` check** *(landed — `hardware.py` checks `torch.backends.mps.is_available()` after CUDA, returns `device="mps"` with a unified-memory VRAM estimate (~60% of system RAM). `HardwareProfile.device` doc updated to include "mps".)*

- [x] **P2-6 · No multi-arch Docker declarations** *(landed — README "Cloned-repo install" gained an Apple Silicon block documenting both the `colima start --arch x86_64 --vm-type vz --vz-rosetta` native path and the `DOCKER_DEFAULT_PLATFORM=linux/amd64` per-command fallback. Verified CI `build-images.yml` publishes amd64-only manifests today, so the QEMU/Rosetta path is the actual user experience until multi-arch buildx lands.)*
  - Files: `Dockerfile`, `Dockerfile.gpu`, `Dockerfile.llama-server`,
    `Dockerfile.workspace`
  - Long-term follow-up still open: multi-arch buildx in CI
    (`docker/build-push-action` with `platforms: linux/amd64,linux/arm64`)
    so Apple Silicon users get native images without the platform-pin
    dance. Tracked separately.

---

## P3 — discovery-phase, mostly self-resolving

- [x] **P3-7 · Native dependency wheel availability** *(landed — `CONTRIBUTING.md` "Setup" gained a "Tested platforms" block: x86_64 Linux (CI) + x86_64 Windows (maintainer) + arm64 macOS (best-effort, no live Mac), Xcode CLT prerequisite for the source-compile fallback, and the CPU/MPS-instead-of-CUDA acceleration note for Apple Silicon.)*

---

## Verified clean (no action needed)

These categories were swept and came back empty — keep an eye on regressions:

- **mDNS / discovery** — `augmentum/cast/mdns.py`, `augmentum/fabric/discovery.py`,
  `augmentum/devices/discovery/subnet_sweep.py`. TCP unicast deliberately
  chosen over multicast for Colima/Docker Desktop compatibility.
- **Audio backends** — `torchaudio`, `librosa`, `soundfile`, ONNX Runtime
  used throughout; no `/dev/snd`, ALSA, or PulseAudio hardcoding.
- **Networking** — no `/sys/class/net`, no `eth0`/`wlan0` NIC-name
  hardcoding. Socket UDP probe in `mdns.py` is cross-platform.
- **File watching** — `watchdog` library used (handles FSEvents on macOS
  transparently). No direct `inotify` calls.
- **No systemd / SELinux / setcap dependencies.**
- **Permissions** — all `chown`/`chmod`/`useradd` calls are inside
  Docker (always Linux). Capabilities (`IPC_LOCK` etc.) declared at
  compose level — Linux Docker only, irrelevant to macOS host.
- **Web Push setup docs** — `docs/web-push-setup.md` uses the correct
  macOS `security add-trusted-cert` command.

---

## How to use this list

- Knock off P0 first — those are the silent-failure points that turn into
  bug reports we can't reproduce.
- P1 unblocks the "native Mac" install path that some power users will
  want for Apple Silicon performance.
- P2 is the path to actually accelerating on Mac hardware.
- P3 is documentation, not code.

When you fix one, tick the box and reference the commit in this file.

---

## Verification (2026-06-04)

After the P0+P1+P2-5 batch landed:

- All 6 touched Python files parse cleanly (AST check).
- New helper `augmentum/utils/paths.py` smoke-tested: resolves
  `/home/augmentum/.kokoro` when the Docker path exists, else falls
  through to the OS-appropriate user cache (verified `LOCALAPPDATA`
  branch on a Windows host).
- `audit.py --quiet` shows 7 regressions — every one traces to files
  this pass didn't touch (migrations 244/245, `ui/scripts/notifications/
  push-prompt.js`, `ui/scripts/settings.js`, `augmentum/connect/
  contacts.py:116`, pre-existing orphaned endpoints, environmental
  `registry.check_failed` from missing `pydantic` in the host shell).
  None of the Mac hardening edits introduced an audit finding.

Remaining open items: **multi-arch CI buildx** (P2-6 long-term tail —
publish `linux/arm64` images alongside `linux/amd64` so Apple Silicon
runs natively instead of via Rosetta/QEMU). Not blocking; docs cover
the workaround.

**Runtime test on real Mac hardware still pending** — the maintainer
doesn't have a Mac to test against. First Mac user to install this
should hit the warmer code paths; any actual breakage will surface as a
loud `gpu_requested_but_unavailable` warning (good — that's the design)
or as a model-load failure pointing at the new user-cache path (also
good — clear root cause). The substrate is in place for them to fix
issues incrementally.

---

## Second-pass findings (2026-06-05) — web client + Safari/iOS

After the first pass closed, a deeper sweep covered the **web client**
(Safari iOS, iPadOS, macOS) and a second look at backend platform
assumptions. Methodology: two parallel read-only agent audits across 11
backend and 9 frontend categories, cross-verified against the live
codebase.

### Landed this pass

- [x] **P1-8 · `stat -c %Y` in `augmentum/coder/tools.py`** *(landed —
  both call sites at `tools.py:307` and `tools.py:654` switched to
  `stat -c %Y '<path>' || stat -f %m '<path>'` OR-chain so BSD/macOS
  coreutils path stays correct if a native-Mac coder workspace ever
  lands. Today the probe always runs in a Linux container so this is
  defensive; the cost is a few bytes of bash.)*

- [x] **P2-9 · `navigator.sendBeacon` lacks keepalive-fetch fallback in
  cast surfaces** *(landed — `cast-audio/cast-audio.js`,
  `cast-comic/cast-comic.js`, `cast-video/cast-video.js` pagehide
  handlers now capture `sendBeacon`'s return value and fall through
  to a `fetch(url, { keepalive: true })` when it returns false or
  throws. Safari iOS low-power mode refuses the queue without throwing
  — those code paths previously lost the user's last play position.
  `chat/sessions.js:737` already had this guard with a 60KB size cap;
  cast progress payloads are ~80 bytes so no size cap needed.)*

- [x] **P2-10 · `100vh` without `100dvh` fallback on cast surfaces**
  *(landed — `cast-livetv.css` (html/body + `.cl-video`),
  `cast-home.css` (`.home`), `cast-control.css` (body),
  `cast-stage.css` (body), `surface-receiver.css` (`.surface-shell`,
  `.reader-stage`) all gained the `height: 100vh; height: 100dvh;`
  pair already used in `styles.css:146-147`. iOS Safari URL-bar
  collapse no longer overflows these layouts.)*

- [x] **P2-11 · No `env(safe-area-inset-*)` on cast surfaces**
  *(landed — main app already had 60+ safe-area uses in
  `ui/styles/*.css`; the gap was in cast-* receiver/controller
  surfaces. `cast-control.css` `.now-playing` / `.scroll-pad` /
  `.toast` and `cast-livetv.css` `.cl-hud` and `cast-comic.css` `.hud`
  all now pad bottom/top/left/right inset so the iPhone notch +
  home-indicator don't overlap pinned UI when the phone is used as
  the controller surface.)*

- [x] **P3-12 · Input `font-size < 16px` triggers iOS auto-zoom**
  *(landed — `voice_mic_tester.html` and `avatar-testbench.html`
  gained `@media (pointer: coarse) { input, textarea, select {
  font-size: 16px !important; } }` so loading the diagnostics on an
  iPad doesn't trap the user in a zoomed-in viewport. Desktop debug
  layout stays at the original tight font-size.)*

### Verified already implemented (audit false positives)

These items the second-pass audit flagged as gaps; verifying against
the live code showed the defense is already in place:

- **Voice WS keepalive for iOS backgrounding** — `voice.js:258` has
  `_heartbeatTimer` periodic ping; `voice.js:922` wires
  `visibilitychange` to `_onVisibilityChange` which detects a
  half-open WS on tab regain; `voice.js:2202-2218` runs a 3-attempt
  reconnect with backoff `[3000, 6000, 12000]ms`.
- **PWA install hint on Safari** — `settings.js:1612-1618` detects when
  `beforeinstallprompt` didn't fire and switches the button to a
  "Show install steps" mode. The expanded instructions
  (`settings.js:1855-1858`) include the explicit Safari iOS/iPadOS
  "Share → Add to Home Screen" line.
- **`navigator.sendBeacon` 64KB cap on chat sessions sync** —
  `chat/sessions.js:737` already drops oversize sessions and warns
  rather than silently failing; `coder.js:1450-1465` already has the
  sendBeacon + keepalive-fetch fallback pattern.

### Verified clean (no action needed)

- **PWA meta tags** — `apple-touch-icon`, `apple-mobile-web-app-capable`,
  `apple-mobile-web-app-status-bar-style`, `viewport-fit=cover` all
  present in `ui/index.html`.
- **Web Push** — VAPID `applicationServerKey` passed correctly;
  Safari 16.4+ supported.
- **WebRTC `getUserMedia` constraints** — no `exact: true` clauses
  that Safari rejects (`voice.js:660-690`, `becca-ptt.js:100-110`).
- **Video playback** — `<video>` elements use `playsinline` and the
  ones that autoplay set `muted` (`cast-video/index.html:22-28`,
  `avatar-testbench.html:112`).
- **WebGL/Three.js, BroadcastChannel, OffscreenCanvas** — clean usage
  or appropriate guards.
- **Touch/pointer** — drag-drop uses pointer events, `-webkit-tap-
  highlight-color` removed on cast surfaces.
- **Backend file watching, audio backends, networking, encoding** —
  all cross-platform library APIs (watchdog, torchaudio, librosa,
  zeroconf, psutil). No `/dev/snd`, ALSA, `pactl`, `eth0`, or
  `inotify`-direct calls.
- **Setup scripts** — `setup.sh`, `start.sh`, `install/install-mac.sh`
  all bash 3.2-compatible (macOS default).
- **`socket.gethostname()` for mDNS** — `mdns.py:69` strips the
  `.local` suffix correctly.
- **Coder container** — probes `host.docker.internal` before
  `172.17.0.1`; both work on Colima.

### Verification (2026-06-05)

After this second-pass batch landed:

- 1 touched Python file parses cleanly (`augmentum/coder/tools.py`).
- 3 JS files (`cast-audio.js`, `cast-comic.js`, `cast-video.js`) and 6
  CSS files (`cast-livetv`, `cast-home`, `cast-control`, `cast-stage`,
  `cast-comic`, `surface-receiver`) plus 2 HTML files
  (`voice_mic_tester`, `avatar-testbench`) edited with surgical scope —
  only the audited concerns, no surrounding refactors.
- `audit.py --quiet` against the live working tree shows 88.5/100 with
  1 regression (`registry.check_failed: 0 -> 1`), which is the
  environmental signal from `pydantic` not being installed in the host
  shell the audit shells out to — not a code regression. Score is
  effectively flat versus the 88.6 from the first hardening pass.
  (An earlier run during this pass briefly showed 78.5 with 10
  regressions; that was a stale-baseline read caught mid-bump, not a
  real regression — re-running cleanly reproduces 88.5.)
