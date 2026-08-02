/**
 * cast-receiver.js — agnostic surface shell.
 *
 * The receiver loads URLs into slot containers and routes messages
 * between iframes and the WS. It does NOT know what the modality is
 * — surface_kind drives the chosen render path, but adding a new
 * kind requires no code change (unknown kinds fall through to
 * iframe-loading, which works for any /ui/<surface>/ route).
 *
 * See docs/superpowers/specs/2026-05-20-cast-surface-protocol.md for
 * the architectural rationale.
 */

const stage = document.getElementById('stage');
const shell = document.querySelector('[data-cr-shell]');
const shellInner = document.querySelector('[data-cr-shell-inner]');
const statusEl = document.getElementById('status');

// The server tells us our own registration_id via a CMD_IDENTITY
// right after WS accept. We stash it so iframed surfaces (notably
// cast-home, for tile-tap-to-cast) can call /api/cast/send back to
// this receiver without a server lookup. Re-issued on every WS
// reconnect because the registration_id is connection-scoped.
let _myReceiverId = '';

// slot_name → HTMLElement
const slots = Object.fromEntries(
  Array.from(document.querySelectorAll('[data-cr-slot]')).map(el => [el.dataset.crSlot, el])
);

// surface_id → { surface_id, surface_kind, slot, el, listeners: [{evt,fn}] }
const surfaces = new Map();

let ws = null;
let reconnectTimer = null;
let pollTimer = null;

// Terminal close codes — receiving these from the server means
// re-pairing won't help (the next attempt will fail the exact same
// way). We must STOP the auto-pair loop, otherwise the receiver
// spins forever burning pair codes + spamming server logs.
//   4003 — receiver revoked (set by _bind_trusted; see CMD_REVOKED)
//   1008 — auth required / policy violation (guest cookie on cast path)
const TERMINAL_CLOSE_CODES = new Set([4003, 1008]);
// Auth-reject close codes — the cookie/session we tried is gone or
// expired, so a silent cookie reconnect can't recover. THIS is the only
// path that should fall back to QR re-pairing.
//   4001 — Unauthorized (no valid session cookie / pair token)
const AUTH_REPAIR_CLOSE_CODES = new Set([4001]);
// Set true when the server tells us not to retry (CMD_REVOKED arrived,
// or a terminal close code was observed). Cleared only by a page reload.
let _terminallyStopped = false;

const RECONNECT_DELAY_MS = 2000;
// A home TV is a passive, always-on cast target: the phone controller
// and the companion can push a surface to it at any time, so we keep the
// WS alive across server restarts + network blips by reconnecting with
// the persistent session cookie — indefinitely, capped backoff. Only an
// auth-reject (cookie expired) drops us back to QR pairing.
const MAX_RECONNECT_DELAY_MS = 30000;
let _reconnectAttempts = 0;
// A WS that the server rejects BEFORE accepting the handshake (no valid
// session cookie) surfaces in the browser as a bare 1006 abnormal-closure —
// the server's 4001/1008 close codes can't be delivered pre-accept, so the
// AUTH_REPAIR_CLOSE_CODES path above never fires and the TV would loop
// "reconnecting" forever. We instead detect it structurally: a cookie-only
// connect that closes without ever reaching 'open' is an auth/handshake
// failure. Tolerate a few (transient blip while Wi-Fi comes up at boot),
// then fall back to QR re-pairing.
const MAX_HANDSHAKE_FAILURES_BEFORE_REPAIR = 3;
let _consecutiveHandshakeFailures = 0;
let _wsOpenedThisAttempt = false;
const POLL_INTERVAL_MS = 1800;
const STATUS_HIDE_MS = 2500;


/* ── Status / shell ──────────────────────────────────────────── */


function setStatus(text, opts = {}) {
  if (!statusEl) return;
  statusEl.textContent = text || '';
  statusEl.classList.toggle('visible', !!text);
  if (opts.autoHide) {
    clearTimeout(setStatus._t);
    setStatus._t = setTimeout(() => statusEl.classList.remove('visible'), STATUS_HIDE_MS);
  }
}


function setShellContent(html) {
  shellInner.innerHTML = html;
  shell.classList.remove('hidden');
  console.log('[cast-receiver] shell shown (setShellContent)');
}


function showShellPlaceholder(text, isError = false) {
  shellInner.innerHTML = `<div id="placeholder"${isError ? ' class="err"' : ''}>${text}</div>`;
  shell.classList.remove('hidden');
  console.log('[cast-receiver] shell shown (placeholder)', text);
}


function hideShell() {
  shell.classList.add('hidden');
  console.log('[cast-receiver] shell hidden');
}


function send(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  try { ws.send(JSON.stringify(obj)); return true; }
  catch { return false; }
}


/* ── Surface mount / unmount ─────────────────────────────────── */


function _findSurfaceInSlot(slot) {
  for (const s of surfaces.values()) {
    if (s.slot === slot) return s;
  }
  return null;
}


function _mountElementForKind(kind, url) {
  // Native-render shortcut for the obvious atomic kinds; iframe for
  // everything else (the forward-compat catch-all).
  switch (kind) {
    case 'media.image': {
      const img = document.createElement('img');
      img.alt = '';
      img.src = url;
      return img;
    }
    case 'media.video': {
      const v = document.createElement('video');
      v.src = url;
      v.controls = true;
      v.autoplay = true;
      v.playsInline = true;
      return v;
    }
    case 'media.audio': {
      const a = document.createElement('audio');
      a.src = url;
      a.controls = true;
      a.autoplay = true;
      return a;
    }
    default: {
      // html.generic + every unknown future kind. The /ui/<surface>/
      // route at `url` is responsible for its own rendering.
      const iframe = document.createElement('iframe');
      iframe.src = url;
      iframe.allow = 'autoplay; fullscreen; encrypted-media; gamepad; xr-spatial-tracking';
      return iframe;
    }
  }
}


function _attachMediaListeners(surfaceId, el, kind) {
  if (kind !== 'media.video' && kind !== 'media.audio') return;
  let lastProgress = -10;
  const onPlaying = () => {
    send({
      type: 'event', event: 'surface_state',
      data: { surface_id: surfaceId, state: { paused: false } },
    });
    _pushMediaState({
      state: 'playing',
      position_ms: Math.round((Number(el.currentTime) || 0) * 1000),
      speed: Number(el.playbackRate) || 1,
    });
  };
  const onPause = () => {
    send({
      type: 'event', event: 'surface_state',
      data: { surface_id: surfaceId, state: { paused: true } },
    });
    _pushMediaState({
      state: 'paused',
      position_ms: Math.round((Number(el.currentTime) || 0) * 1000),
      speed: 0,
    });
  };
  const onTimeUpdate = () => {
    if (Math.floor(el.currentTime) >= lastProgress + 5) {
      lastProgress = Math.floor(el.currentTime);
      send({
        type: 'event', event: 'surface_state',
        data: {
          surface_id: surfaceId,
          state: {
            position_s: Number(el.currentTime) || 0,
            duration_s: Number(el.duration) || 0,
          },
        },
      });
      // Also re-publish to the MediaSession so the system seek bar
      // tracks reality. We only push every 5s so this is cheap.
      _pushMediaState({
        state: el.paused ? 'paused' : 'playing',
        position_ms: Math.round((Number(el.currentTime) || 0) * 1000),
        speed: el.paused ? 0 : (Number(el.playbackRate) || 1),
      });
    }
  };
  const onEnded = () => {
    closeSurface(surfaceId, 'ended');
  };
  const onError = () => send({
    type: 'event', event: 'surface_state',
    data: { surface_id: surfaceId, state: { error: el.error ? String(el.error.code) : 'unknown' } },
  });
  el.addEventListener('playing', onPlaying);
  el.addEventListener('pause', onPause);
  el.addEventListener('timeupdate', onTimeUpdate);
  el.addEventListener('ended', onEnded);
  el.addEventListener('error', onError);
  return [
    { evt: 'playing', fn: onPlaying },
    { evt: 'pause', fn: onPause },
    { evt: 'timeupdate', fn: onTimeUpdate },
    { evt: 'ended', fn: onEnded },
    { evt: 'error', fn: onError },
  ];
}


function mountSurface({ surface_id, surface_kind, surface_url, slot, state }) {
  if (!slots[slot]) {
    send({
      type: 'event', event: 'surface_state',
      data: { surface_id, state: { error: `unknown_slot:${slot}` } },
    });
    return;
  }

  // Replace any existing surface in the slot (single-occupant rule).
  const existing = _findSurfaceInSlot(slot);
  if (existing) closeSurface(existing.surface_id, 'replaced');

  const el = _mountElementForKind(surface_kind, surface_url);
  slots[slot].appendChild(el);

  const listeners = _attachMediaListeners(surface_id, el, surface_kind) || [];

  surfaces.set(surface_id, {
    surface_id,
    surface_kind,
    surface_url,
    slot,
    el,
    listeners,
    state: state || {},
  });

  // For iframe surfaces, push the init payload (surface_id + initial
  // state) once the iframe is ready. Without this the surface code
  // doesn't know its identity, which means it can't emit events
  // back keyed to the right surface_id.
  if (el.tagName === 'IFRAME') {
    const initPayload = {
      type: 'augmentum.surface_init',
      surface_id,
      kind: surface_kind,
      slot,
      state: state || {},
      // Pass-through so interactive surfaces (e.g. cast-home tile
      // taps) can POST /api/cast/send back to this receiver without
      // a server lookup. Empty until the server-side identity
      // handshake completes — that races against iframe load, so
      // _rebroadcastIdentityToMain() also re-pushes on identity
      // arrival.
      receiver_id: _myReceiverId,
    };
    el.addEventListener('load', () => {
      try { el.contentWindow?.postMessage(initPayload, '*'); } catch {}
      // Push the receiver's prefs bag once the iframe is up. cast-home
      // listens for ``augmentum.prefs`` and uses it to filter rails;
      // any other iframe surface just ignores the unknown message.
      _postPrefsTo(el);
    });
  }

  send({
    type: 'event', event: 'surface_opened',
    data: { surface_id, kind: surface_kind, slot },
  });

  // Push initial playback metadata to the MediaSession when the
  // surface_open args carry media info. Server-side ``args.state``
  // typically includes title / subtitle / cover_url / duration_ms
  // for media surfaces. For non-media surfaces (cast-home, comic
  // reader, browse panel) we just leave the bridge empty.
  if (slot === 'main') {
    const initState = state || {};
    const isMediaSurface =
      surface_kind === 'media.video'
      || surface_kind === 'media.audio'
      || (surface_kind === 'html.generic' && (initState.title || initState.media));
    if (isMediaSurface) {
      _activeMediaSurfaceId = surface_id;
      const media = initState.media || initState;
      _pushMediaMeta({
        title: media.title || '',
        subtitle: media.subtitle || '',
        artist: media.artist || media.author || '',
        cover_url: media.cover_url || media.poster_url || '',
        duration_ms: typeof media.duration_ms === 'number'
          ? media.duration_ms
          : (typeof media.duration_s === 'number'
              ? Math.round(media.duration_s * 1000)
              : 0),
      });
      // Default initial state — surfaces tend to autoplay so we
      // optimistically mark playing; a pause patch will correct it.
      _pushMediaState({ state: 'playing', position_ms: 0, speed: 1 });
    } else {
      _activeMediaSurfaceId = '';
      _clearMedia();
    }
  }

  // Hide the shell once main slot is occupied.
  if (slot === 'main') hideShell();
}


function closeSurface(surfaceId, reason = 'cmd') {
  const s = surfaces.get(surfaceId);
  if (!s) return;
  const wasMainSlot = s.slot === 'main';
  try {
    if (typeof s.el.pause === 'function') s.el.pause();
  } catch {}
  for (const { evt, fn } of s.listeners) {
    try { s.el.removeEventListener(evt, fn); } catch {}
  }
  s.el.remove();
  surfaces.delete(surfaceId);
  // Drop the MediaSession's now-playing card if the surface we're
  // closing was the active media owner. Skips when a non-media
  // surface closes (it never claimed the bridge in the first place).
  if (_activeMediaSurfaceId && _activeMediaSurfaceId === surfaceId) {
    _activeMediaSurfaceId = '';
    _clearMedia();
  }
  send({
    type: 'event', event: 'surface_closed',
    data: { surface_id: surfaceId, reason },
  });
  // Restore cast-home as the idle surface when the main slot empties.
  // Skipped for:
  //   - 'replaced'    : mountSurface is about to put a new surface in
  //                     the slot; remounting cast-home here would flash
  //                     it for one frame between back-to-back casts.
  //   - 'disconnected': WS drop path tears down every surface; the
  //                     reconnect handler shows the pairing shell and
  //                     re-mounts cast-home itself on the next open.
  // Without this restore, the TV showed a black void after a movie /
  // show ended until the user explicitly cast something else — the
  // single-occupant slot rule destroyed cast-home on the way in and
  // nothing brought it back.
  if (
    wasMainSlot
    && reason !== 'replaced'
    && reason !== 'disconnected'
    && ws?.readyState === WebSocket.OPEN
  ) {
    mountSurface({
      surface_id: `cast_home_${Date.now()}`,
      surface_kind: 'html.generic',
      surface_url: '/ui/cast-home/',
      slot: 'main',
      state: {},
    });
  }
}


function applyPatch(surfaceId, patch) {
  const s = surfaces.get(surfaceId);
  if (!s) return;
  const el = s.el;
  if (s.surface_kind === 'media.video' || s.surface_kind === 'media.audio') {
    if (typeof patch.paused === 'boolean') {
      try { patch.paused ? el.pause() : el.play(); } catch {}
    }
    if (typeof patch.position_s === 'number' && 'currentTime' in el) {
      el.currentTime = patch.position_s;
    }
    if (typeof patch.volume === 'number' && 'volume' in el) {
      el.volume = Math.max(0, Math.min(1, patch.volume));
    }
    if (typeof patch.muted === 'boolean') {
      el.muted = patch.muted;
    }
    return;
  }
  // For iframe surfaces, forward the patch via postMessage so the
  // surface code (own JS) can apply whatever it wants.
  if (el.tagName === 'IFRAME') {
    try {
      el.contentWindow?.postMessage(
        { type: 'augmentum.surface_state', surface_id: surfaceId, patch },
        '*',
      );
    } catch {}
    // Mirror controller-driven transport changes into the MediaSession
    // so the system now-playing card reflects them too. We can't read
    // the iframe's actual time from across the origin, so the seek
    // bar tracks "what the controller asked for" rather than the
    // surface's confirmation — close enough for the system glance.
    if (_activeMediaSurfaceId === surfaceId) {
      const update = {};
      if (typeof patch.paused === 'boolean') {
        update.state = patch.paused ? 'paused' : 'playing';
        update.speed = patch.paused ? 0 : 1;
      }
      if (typeof patch.position_s === 'number') {
        update.position_ms = Math.round(patch.position_s * 1000);
      }
      if (Object.keys(update).length) _pushMediaState(update);
    }
  }
}


function focusSlot(slot) {
  // Phase A: minimal — just toggle visibility/z-index of overlay.
  // Most useful when promoting pip↔main; deferred until needed.
  if (!slots[slot]) return;
  // No-op placeholder. Future: rotate z-indexes so {slot} reaches
  // the top of the visible stack.
}


/* ── Iframe postMessage relay (surface_event from iframes) ──── */


window.addEventListener('message', (ev) => {
  // Match the source frame to a known surface.
  for (const s of surfaces.values()) {
    if (s.el.tagName === 'IFRAME' && s.el.contentWindow === ev.source) {
      const payload = ev.data;
      if (!payload || typeof payload !== 'object') return;
      // Iframes opt into the surface_state echo path by tagging their
      // postMessage with ``type: 'augmentum.surface_state'``. We
      // unwrap that into the same wire shape the native media-shortcut
      // path uses (cast-receiver.js:132+) so the server's per-surface
      // state bag merges from both sources without branching.
      if (payload.type === 'augmentum.surface_state'
          && payload.state && typeof payload.state === 'object') {
        send({
          type: 'event', event: 'surface_state',
          data: { surface_id: s.surface_id, state: payload.state },
        });
        return;
      }
      // Everything else is forwarded as a generic surface_event.
      send({
        type: 'event', event: 'surface_event',
        data: { surface_id: s.surface_id, ...payload },
      });
      return;
    }
  }
});


/* ── Per-receiver display preferences ─────────────────────────────
 *
 * Fetched once on boot from /api/cast/receiver-self/prefs?device_id=…
 * and cached for the lifetime of this page. cast-home consumes the
 * bag via postMessage on mount so it can hide rails / skip the
 * backdrop cycle / etc. without re-fetching on its own.
 *
 * Updates from the controller (PUT to trusted-receivers/{id}/prefs)
 * don't push to us in real time today — the user has to reload the
 * receiver to see new prefs. That's intentional for the lean MVP;
 * a future iteration can add a cast-event for "prefs_changed". */
let _receiverPrefs = null;
// Captured on the same fetch as the prefs so cast-home can pass it
// when calling /api/cast/library/home — the server filters the rail
// list by this receiver's rails_visible bag before any data fetch.
// Empty until /api/cast/receiver-self/prefs returns a trust binding;
// unpaired receivers stay empty and the home call falls through to
// "show all rails" on the server side.
let _receiverTrustedId = "";

async function _fetchReceiverPrefs(deviceId) {
  try {
    const qs = deviceId ? `?device_id=${encodeURIComponent(deviceId)}` : '';
    const r = await fetch(`/api/cast/receiver-self/prefs${qs}`, {
      credentials: 'same-origin',
      cache: 'no-store',
    });
    if (!r.ok) return null;
    const body = await r.json();
    if (typeof body.trusted_id === 'string') {
      _receiverTrustedId = body.trusted_id;
    }
    return body.prefs || null;
  } catch {
    return null;
  }
}

function _postPrefsTo(iframe) {
  if (!iframe || !_receiverPrefs) return;
  try {
    iframe.contentWindow?.postMessage(
      {
        type: 'augmentum.prefs',
        payload: _receiverPrefs,
        // trusted_id rides with prefs so cast-home can pass it on
        // /api/cast/library/home calls — the server filters rails by
        // this receiver's rails_visible bag before fetching data.
        trusted_id: _receiverTrustedId || '',
      },
      '*',
    );
  } catch { /* swallow */ }
}

/** Re-push the identity to whatever's mounted in main. Called on
 *  CMD_IDENTITY arrival in case cast-home was already mounted (and
 *  saw an empty receiver_id in its surface_init payload because the
 *  identity cmd hadn't landed yet). cast-home listens for both
 *  ``augmentum.surface_init`` and the dedicated ``augmentum.identity``
 *  message so it can hydrate either way. */
function _rebroadcastIdentityToMain() {
  const main = _findSurfaceInSlot('main');
  if (!main || main.el?.tagName !== 'IFRAME') return;
  try {
    main.el.contentWindow?.postMessage(
      { type: 'augmentum.identity', receiver_id: _myReceiverId },
      '*',
    );
  } catch { /* swallow */ }
}


/* ── System-level (Android TV bridge) ─────────────────────────── */
//
// The bundled Android TV APK injects ``window.AugmentumTV`` with
// real system-volume control. Browser receivers don't have it, so
// we feature-detect and silently no-op for those — the controller
// gets ``supported: false`` back and hides the TV-master slider.

function _bridge() {
  return typeof window.AugmentumTV !== 'undefined' ? window.AugmentumTV : null;
}

/* Best-effort network-identity sniff for WoL auto-fill. Returns
 * {mac, ip} — both '' when unavailable. Browsers can't expose MAC,
 * so only the Android-TV bridge ever fills these. Server-side store
 * normalises + tolerates absence; callers below never branch on this. */
function _networkInfo() {
  const b = _bridge();
  if (!b) return { mac: '', ip: '' };
  let mac = '';
  let ip = '';
  // Newer bridges expose a single accessor; older ones split it. Try
  // both shapes; swallow + ignore method-absent failures so a
  // partial-implementation bridge doesn't crash the ready event.
  try {
    if (typeof b.getNetworkInfo === 'function') {
      const raw = b.getNetworkInfo();
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
      if (parsed && typeof parsed === 'object') {
        mac = String(parsed.mac_address || parsed.mac || '');
        ip = String(parsed.local_ip || parsed.ip || '');
      }
    }
  } catch {}
  if (!mac && typeof b.getMacAddress === 'function') {
    try { mac = String(b.getMacAddress() || ''); } catch {}
  }
  if (!ip && typeof b.getLocalIp === 'function') {
    try { ip = String(b.getLocalIp() || ''); } catch {}
  }
  return { mac, ip };
}

function _emitSystemVolumeState() {
  const b = _bridge();
  if (!b) {
    send({
      type: 'event', event: 'system_volume_state',
      data: { supported: false, volume: 0, muted: false },
    });
    return;
  }
  let volume = 0;
  let muted = false;
  try { volume = Number(b.getSystemVolume()) || 0; } catch {}
  try { muted = !!b.isSystemMuted(); } catch {}
  send({
    type: 'event', event: 'system_volume_state',
    data: { supported: true, volume, muted },
  });
}

/* ── MediaSession (Android TV remote / Assistant / system tray) ────
 *
 * The APK's PlaybackService holds a MediaSessionCompat. We push title
 * + state into it whenever a surface mounts / a playback event fires,
 * and consume remote-key events from it via the ``__augmentumOnRemote``
 * global which the bridge invokes when the user hits play/pause/skip
 * on the TV remote.
 *
 * No-ops cleanly in non-APK contexts: ``_bridge()`` returns null and
 * the helpers bail. Browser receivers ignore the whole pathway. */

function _pushMediaMeta(meta) {
  const b = _bridge();
  if (!b || typeof b.setNowPlaying !== 'function') return;
  try { b.setNowPlaying(JSON.stringify(meta || {})); } catch {}
}

function _pushMediaState(state) {
  const b = _bridge();
  if (!b || typeof b.setPlaybackState !== 'function') return;
  try { b.setPlaybackState(JSON.stringify(state || {})); } catch {}
}

function _clearMedia() {
  const b = _bridge();
  if (!b || typeof b.clearNowPlaying !== 'function') return;
  try { b.clearNowPlaying(); } catch {}
}

/** Active media surface tracker. Whichever surface is currently the
 *  authoritative playback source (main-slot media surface) gets its
 *  events relayed to the bridge. Cleared on close. */
let _activeMediaSurfaceId = '';

/** Remote-key consumer. Invoked by AugmentumTvBridge → MediaSession
 *  callbacks when the user hits a transport key on the TV remote /
 *  asks Google Assistant to pause / etc. Translates the action to a
 *  patch and dispatches it through applyPatch to whatever's mounted. */
window.__augmentumOnRemote = function _augmentumOnRemote(action, value) {
  const sid = _activeMediaSurfaceId;
  if (!sid) return;
  let patch = null;
  switch (action) {
    case 'play':    patch = { paused: false }; break;
    case 'pause':   patch = { paused: true }; break;
    case 'stop':    closeSurface(sid, 'remote_stop'); return;
    case 'ffwd':    patch = { seek_delta_s: 30 }; break;
    case 'rewind':  patch = { seek_delta_s: -15 }; break;
    case 'seek_ms': patch = { position_s: Number(value) / 1000 }; break;
    // ``next`` / ``previous`` aren't part of the patch schema today;
    // forward as raw postMessage so the surface (if it knows what
    // those mean) can act on them, otherwise harmless.
    case 'next':
    case 'previous': {
      const s = surfaces.get(sid);
      if (s?.el?.tagName === 'IFRAME') {
        try {
          s.el.contentWindow?.postMessage(
            { type: 'augmentum.remote', action }, '*');
        } catch {}
      }
      return;
    }
    default: return;
  }
  applyPatch(sid, patch);
};


function applySystemVolume(args) {
  const b = _bridge();
  if (!b) {
    // Echo back ``supported: false`` so the controller can hide the
    // slider immediately instead of waiting for a polling cycle.
    _emitSystemVolumeState();
    return;
  }
  try {
    if (typeof args.muted === 'boolean') {
      b.setSystemMuted(args.muted);
    }
    if (typeof args.volume === 'number') {
      b.setSystemVolume(args.volume);
    } else if (typeof args.delta === 'number' && args.delta !== 0) {
      b.adjustVolume(args.delta | 0);
    }
  } catch (err) {
    console.warn('[cast-receiver] system volume bridge call threw', err);
  }
  // Echo the snapped state so the controller mirrors reality —
  // Android quantises to integer step counts, so 0.37 might land at
  // 0.40 / 0.33 / whatever depending on the device's step count.
  _emitSystemVolumeState();
}


/* ── Command dispatch ─────────────────────────────────────────── */


function handleCmd(msg) {
  const cmd = String(msg.cmd || '');
  const args = msg.args || {};
  switch (cmd) {
    // ── Identity handshake (server → receiver, once per connect) ─
    case 'identity':
      _myReceiverId = String(args.registration_id || '');
      console.log('[cast-receiver] identity received', _myReceiverId);
      // If cast-home is already mounted (typical for a reconnect),
      // re-push surface_init so it picks up the fresh receiver id.
      _rebroadcastIdentityToMain();
      break;

    // ── Terminal "you've been revoked" signal ────────────────────
    // Server is about to close us with 4003. Stop the re-pair loop
    // and tell the user how to recover — they need to restore the
    // device from Settings → TVs on a logged-in browser.
    case 'revoked': {
      const trustedId = String(args.trusted_id || '');
      const reason = String(args.reason || 'revoked');
      console.warn(
        '[cast-receiver] revoked by server',
        { trusted_id: trustedId, reason },
      );
      _stopTerminally(_buildRevokedMessage(trustedId, reason));
      break;
    }

    // ── System-level (device-wide, not surface-scoped) ───────
    case 'system_volume':
      applySystemVolume(args);
      break;

    // ── Follow-mode navigation ───────────────────────────────
    // Forwards verbatim to whatever's mounted in main (cast-home
    // when no media is casting). The cast-home iframe listens for
    // ``augmentum.nav`` postMessages and re-renders. When a real
    // surface is mounted (cast-video etc.), it's an unknown message
    // and silently ignored — exactly what we want.
    case 'nav': {
      const mainSurface = _findSurfaceInSlot('main');
      if (!mainSurface) {
        console.warn('[cast-receiver] nav: no surface mounted in main slot', args);
        break;
      }
      if (mainSurface.el?.tagName !== 'IFRAME') {
        console.warn('[cast-receiver] nav: main surface is not an iframe',
          { tag: mainSurface.el?.tagName, kind: mainSurface.surface_kind });
        break;
      }
      console.log('[cast-receiver] nav forwarding to main iframe',
        { kind: mainSurface.surface_kind, url: mainSurface.surface_url, args });
      try {
        mainSurface.el.contentWindow?.postMessage(
          { type: 'augmentum.nav', payload: args },
          '*',
        );
      } catch (err) {
        console.warn('[cast-receiver] nav postMessage threw', err);
      }
      break;
    }

    // ── Live preferences update ──────────────────────────────
    // Server pushes this after the controller PUTs new prefs. We
    // cache the bag and forward it into whatever's mounted in main
    // (typically cast-home) so the rail filter takes effect without
    // a page reload.
    case 'prefs_changed': {
      const next = (args && typeof args === 'object' && args.prefs)
        ? args.prefs : null;
      if (next) {
        _receiverPrefs = next;
        const main = _findSurfaceInSlot('main');
        if (main?.el?.tagName === 'IFRAME') _postPrefsTo(main.el);
      }
      break;
    }

    // ── Couch co-op invite QR overlay ────────────────────────
    // Host's phone tapped "+ Players" on an active game. We render
    // a fullscreen QR card over whatever's playing in main; the game
    // keeps streaming behind it (semi-transparent overlay slot).
    // Guests scan the QR with their phones, land on cast-guest-join,
    // and their controllers become P2/P3/P4.
    case 'show_invite_qr': {
      _showInviteQrOverlay({
        token: String(args.token || ''),
        qrUrl: String(args.qr_url || ''),
        joinUrl: String(args.join_url || ''),
        slotsRemaining: Number(args.slots_remaining) || 0,
        slotsTotal: Number(args.slots_total) || 0,
        expiresAt: Number(args.expires_at) || 0,
      });
      break;
    }
    case 'hide_invite_qr': {
      _hideInviteQrOverlay(String(args.reason || ''));
      break;
    }
    case 'invite_slot_update': {
      _renderPlayersStrip(args);
      break;
    }

    // ── Browser-cast gamepad input ──────────────────────────
    // Server fanouts this for every gamepad frame on the phone WS
    // bound to this receiver. We forward unchanged to the main slot's
    // iframe, where the universal-input-adapter loader (loaded by
    // /ui/play/ and /ui/play-web/) dispatches it to whichever
    // adapter chain is active — gamepad_api shim by default, or any
    // combination of keyboard / touch / pointer per-game profile.
    // 60 Hz hot path — keep cheap.
    case 'input_gamepad': {
      const main = _findSurfaceInSlot('main');
      if (main?.el?.tagName === 'IFRAME') {
        try {
          main.el.contentWindow?.postMessage(
            {
              kind: 'augmentum.cast_input',
              slot: Number(args.slot) | 0,
              pad_index: Number(args.pad_index) | 0,
              buttons: args.buttons || [],
              axes: args.axes || [],
            },
            '*',
          );
        } catch (_) { /* iframe gone — silent drop */ }
      }
      break;
    }

    // ── Library refresh nudge ───────────────────────────────
    // Server fanouts this after a /api/media/progress write (debounced
    // server-side at 30s/user) so the Continue rail on the idle screen
    // reorders without waiting for its 5-minute poll. Forwarded to the
    // main iframe; cast-home listens and calls refresh(). Surfaces
    // other than cast-home will silently ignore the unknown message.
    case 'library_invalidate': {
      const main = _findSurfaceInSlot('main');
      if (main?.el?.tagName === 'IFRAME') {
        try {
          main.el.contentWindow?.postMessage(
            { type: 'augmentum.library_invalidate' },
            '*',
          );
        } catch (err) {
          console.warn('[cast-receiver] library_invalidate forward threw', err);
        }
      }
      break;
    }

    // ── Surface verbs (new protocol) ─────────────────────────
    case 'surface_open':
      mountSurface({
        surface_id: String(args.surface_id || ''),
        surface_kind: String(args.surface_kind || 'html.generic'),
        surface_url: String(args.surface_url || ''),
        slot: String(args.slot || 'main'),
        state: args.state || {},
      });
      break;
    case 'surface_close':
      closeSurface(String(args.surface_id || ''), 'cmd');
      break;
    case 'surface_focus':
      focusSlot(String(args.slot || 'main'));
      break;
    case 'surface_state':
      applyPatch(String(args.surface_id || ''), args.patch || {});
      break;

    // ── Legacy cmds (backward compat — handled directly by the
    //    receiver until servers fully migrate to surface_open). ──
    case 'play': {
      const url = String(args.url || '');
      const looksVideo = (args.kind === 'video') ||
        /\.(mp4|webm|mkv|m4v|mov)(\?|$)/i.test(url);
      mountSurface({
        surface_id: `lg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        surface_kind: looksVideo ? 'media.video' : 'media.audio',
        surface_url: url,
        slot: 'main',
        state: args,
      });
      break;
    }
    case 'show_image':
      mountSurface({
        surface_id: `lg_${Date.now()}_img`,
        surface_kind: 'media.image',
        surface_url: String(args.url || ''),
        slot: 'main',
        state: args,
      });
      break;
    case 'show_html':
    case 'show_artifact':
      mountSurface({
        surface_id: `lg_${Date.now()}_html`,
        surface_kind: 'html.generic',
        surface_url: String(args.url || ''),
        slot: 'main',
        state: args,
      });
      break;
    case 'pause':
    case 'resume':
    case 'seek':
    case 'volume':
    case 'stop': {
      const main = _findSurfaceInSlot('main');
      if (!main) break;
      if (cmd === 'stop') {
        closeSurface(main.surface_id, 'cmd');
        break;
      }
      const patch = {};
      if (cmd === 'pause') patch.paused = true;
      if (cmd === 'resume') patch.paused = false;
      if (cmd === 'seek') patch.position_s = Number(args.position_s) || 0;
      if (cmd === 'volume') patch.volume = Number(args.level);
      applyPatch(main.surface_id, patch);
      break;
    }
    default:
      // Forward-compat: unknown cmds are dropped silently. Future
      // server versions may emit cmds this receiver doesn't yet
      // understand; better than rejecting the WS.
      break;
  }
  if (msg.id) {
    send({ type: 'event', event: 'ack', id: msg.id, data: {} });
  }
}


/* ── Couch co-op invite QR overlay ──────────────────────────── */


// Token of the currently-displayed invite; null when no overlay.
let _activeInviteToken = null;
let _inviteCountdownTimer = null;

function _showInviteQrOverlay({
  token, qrUrl, joinUrl, slotsRemaining, slotsTotal, expiresAt,
}) {
  const slot = document.querySelector('[data-cr-slot="overlay"]');
  if (!slot) return;
  _activeInviteToken = token;

  // Re-render in place if the overlay is already up — slot counter
  // updates as guests claim. Cheaper than tearing down and rebuilding.
  const existing = slot.querySelector('.cr-invite-card');
  if (existing) {
    const counter = existing.querySelector('.cr-invite-slots');
    if (counter) {
      counter.textContent = _inviteSlotLabel(slotsRemaining, slotsTotal);
    }
    return;
  }

  slot.innerHTML = '';
  const card = document.createElement('div');
  card.className = 'cr-invite-card';
  card.innerHTML = `
    <div class="cr-invite-title">Invite players</div>
    <div class="cr-invite-sub">
      Scan with a phone — your controller joins as the next player.
    </div>
    <div class="cr-invite-qr">
      <img alt="Join QR code" src="${_attr(qrUrl)}">
    </div>
    <div class="cr-invite-url">${_attr(joinUrl)}</div>
    <div class="cr-invite-slots">${_inviteSlotLabel(slotsRemaining, slotsTotal)}</div>
    <div class="cr-invite-countdown" data-cr-countdown></div>
  `;
  slot.appendChild(card);

  // Tick the countdown locally — server isn't going to spam us with
  // updates, and a fresh expires_at is enough to render the time-left.
  if (_inviteCountdownTimer) clearInterval(_inviteCountdownTimer);
  const tick = () => {
    const el = card.querySelector('[data-cr-countdown]');
    if (!el) return;
    const secs = Math.max(0, Math.round(expiresAt - Date.now() / 1000));
    el.textContent = secs > 0 ? `Expires in ${secs}s` : 'Expired';
    if (secs <= 0) _hideInviteQrOverlay('expired');
  };
  tick();
  _inviteCountdownTimer = setInterval(tick, 1000);
}

function _hideInviteQrOverlay(reason) {
  _activeInviteToken = null;
  if (_inviteCountdownTimer) {
    clearInterval(_inviteCountdownTimer);
    _inviteCountdownTimer = null;
  }
  const slot = document.querySelector('[data-cr-slot="overlay"]');
  if (slot) slot.innerHTML = '';
  if (reason) console.log('[cast-receiver] invite QR hidden', reason);
}

/* ── Player roster strip ────────────────────────────────────────────
 * Renders the bottom-of-screen chip strip showing who's playing. The
 * host is implicit P1 (their input goes through the same WS but isn't
 * tracked in CastInputRegistry); guests fill P2..P4 as they claim.
 */
function _renderPlayersStrip(args) {
  const strip = document.querySelector('[data-cr-players-strip]');
  if (!strip) return;
  const players = Array.isArray(args?.players) ? args.players : [];
  if (players.length === 0) {
    strip.hidden = true;
    strip.innerHTML = '';
    return;
  }
  strip.hidden = false;
  // P1 chip always-on while ANY guest has claimed — the host is
  // implicit. We could plumb the host's display name through but
  // for v1 "Host" is enough.
  const hostChip = `
    <div class="cr-player-chip" data-slot="host">
      <span class="cr-player-dot" style="background:#888"></span>
      <span class="cr-player-name">Host</span>
    </div>
  `;
  const guestChips = players.map((p) => `
    <div class="cr-player-chip" data-slot="${Number(p.slot)}">
      <span class="cr-player-num">P${Number(p.slot) + 1}</span>
      <span class="cr-player-dot" style="background:${_attr(p.color || _autoColor(p.slot))}"></span>
      <span class="cr-player-name">${_attr(p.name || 'Guest')}</span>
    </div>
  `).join('');
  strip.innerHTML = `${hostChip}${guestChips}`;
}

function _autoColor(slot) {
  // Stable per-slot colour for unnamed/anonymous guests.
  const palette = ['#4ade80', '#60a5fa', '#fbbf24', '#f472b6'];
  return palette[(Number(slot) || 0) % palette.length];
}

function _inviteSlotLabel(remaining, total) {
  if (total <= 0) return '';
  if (remaining <= 0) return 'All slots taken';
  if (remaining === 1) return `${remaining} of ${total} slot open`;
  return `${remaining} of ${total} slots open`;
}

function _attr(s) {
  return String(s || '')
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}


/* ── Pair bootstrap (unchanged from previous version) ──────── */


function renderPairPanel({ pair_code, pair_url, qr_url, expires_in }) {
  setShellContent(`
    <div class="cr-pair">
      <div class="cr-pair-title">Pair this receiver</div>
      <div class="cr-pair-sub">Scan with your phone, or open this URL on a logged-in device.</div>
      <div class="cr-pair-qr"><img src="${qr_url}" alt="Pair QR code" /></div>
      <div class="cr-pair-code">${pair_code}</div>
      <div class="cr-pair-url">${pair_url}</div>
      <div class="cr-pair-expires">expires in <span data-cr-expires>${expires_in}</span>s</div>
    </div>
  `);
  const expiresEl = shell.querySelector('[data-cr-expires]');
  let remaining = Number(expires_in) || 0;
  clearInterval(renderPairPanel._t);
  renderPairPanel._t = setInterval(() => {
    remaining -= 1;
    if (expiresEl) expiresEl.textContent = String(Math.max(0, remaining));
    if (remaining <= 0) clearInterval(renderPairPanel._t);
  }, 1000);
}


async function startPair() {
  setStatus('pairing');
  let body;
  try {
    const r = await fetch('/api/cast/pair/start', { method: 'POST' });
    if (!r.ok) throw new Error(`pair/start returned ${r.status}`);
    body = await r.json();
  } catch (err) {
    showShellPlaceholder(`Pair start failed: ${err.message}`, true);
    reconnectTimer = setTimeout(startPair, RECONNECT_DELAY_MS);
    return;
  }
  renderPairPanel(body);
  startPolling(body.pair_code, body.poll_path);
}


function startPolling(pairCode, pollPath) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    let body;
    try {
      const r = await fetch(pollPath);
      if (r.status === 404) {
        clearInterval(pollTimer);
        showShellPlaceholder('Pair expired — restart pairing', true);
        reconnectTimer = setTimeout(startPair, RECONNECT_DELAY_MS);
        return;
      }
      if (!r.ok) return;
      body = await r.json();
    } catch {
      return;
    }
    if (body.state === 'approved' && body.ws_token) {
      clearInterval(pollTimer);
      // Establish the HTTP session cookie BEFORE opening the WS so
      // every subsequent iframe / surface load carries auth. Without
      // this, the WebView has only the WS-scoped wsp_* and any /api
      // call from a surface 401s. See /api/cast/pair/establish-session.
      establishSessionThenConnect(body.ws_token);
    } else if (body.state === 'expired') {
      clearInterval(pollTimer);
      showShellPlaceholder('Pair expired — restart pairing', true);
      reconnectTimer = setTimeout(startPair, RECONNECT_DELAY_MS);
    }
  }, POLL_INTERVAL_MS);
}


/* ── WebSocket ───────────────────────────────────────────────── */


async function establishSessionThenConnect(wsToken) {
  setStatus('signing in');
  let setOk = false;
  try {
    const r = await fetch('/api/cast/pair/establish-session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      // device_id lets the server key this session to the physical TV so a
      // re-pair REPLACES this device's previous auth_sessions row instead of
      // stacking a new one per pairing (device rows are exempt from the
      // browser LRU pool and capped per source instead).
      body: JSON.stringify({
        ws_token: wsToken,
        device_id: _resolveDeviceIdentity().device_id || '',
      }),
    });
    setOk = r.ok || r.status === 204;
    if (!setOk) {
      console.warn('[cast-receiver] establish-session', r.status, await r.text().catch(() => ''));
    }
  } catch (err) {
    console.warn('[cast-receiver] establish-session threw', err);
  }

  // Verify the cookie actually reached the cookie jar — on Android
  // WebView there have been cases where Set-Cookie from fetch lands
  // late or not at all. If auth_status doesn't see us as logged in,
  // surface that clearly rather than silently opening the WS and
  // letting iframe surfaces 401 mysteriously.
  let authenticated = false;
  try {
    const r = await fetch('/api/auth/status', { credentials: 'include', cache: 'no-store' });
    if (r.ok) {
      const body = await r.json();
      authenticated = !!body.authenticated;
    }
  } catch (err) {
    console.warn('[cast-receiver] auth status check threw', err);
  }

  if (!authenticated) {
    showShellPlaceholder(
      setOk
        ? 'Session set but cookie not visible — iframe surfaces may fail. Reload from the TV to retry.'
        : 'Session bootstrap failed — iframe surfaces will not load.',
      true,
    );
  } else {
    console.log('[cast-receiver] session established + verified');
  }
  connectWS(wsToken);
}


// connectWS(wsToken) — open the receiver WS. With a one-time wsp_* token
// (right after QR pairing) OR with NO token, in which case the same-origin
// session cookie authenticates us (middleware cookie-fallback for
// /api/cast/receiver/). Passing null is the steady-state reconnect path:
// a paired TV holds a long-lived cookie, so it re-attaches silently after
// a server restart without burning a new pair code.
function connectWS(wsToken) {
  clearTimeout(reconnectTimer);
  const proto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
  const base = `${proto}//${location.host}/api/cast/receiver/ws`;
  const url = wsToken
    ? `${base}?token=${encodeURIComponent(wsToken)}`
    : base;
  setStatus('connecting');

  _wsOpenedThisAttempt = false;
  ws = new WebSocket(url);

  ws.addEventListener('open', () => {
    // Live again — clear the backoff so the next drop reconnects fast.
    _wsOpenedThisAttempt = true;
    _reconnectAttempts = 0;
    _consecutiveHandshakeFailures = 0;
    // Auto-mount the cast-home idle surface so the TV shows
    // recently-played + clock + ambient art instead of a "Ready"
    // placeholder. A real surface_open from /api/cast/send (or
    // /api/cast/render-stream/start) replaces it via the
    // single-occupant slot rule (which DESTROYS the cast-home
    // iframe — it's not backgrounded). When the playing surface
    // later closes, closeSurface re-mounts cast-home as the idle
    // fallback so the TV returns here instead of going black.
    mountSurface({
      surface_id: `cast_home_${Date.now()}`,
      surface_kind: 'html.generic',
      surface_url: '/ui/cast-home/',
      slot: 'main',
      state: {},
    });
    setStatus('ready', { autoHide: true });
    // Announce initial system-volume capability + state so the
    // controller's TV-master slider can populate (or hide itself
    // when no bridge is present).
    _emitSystemVolumeState();
    const identity = _resolveDeviceIdentity();
    // Fetch this receiver's prefs in parallel with the ready event.
    // The cast-home iframe may already be mounted; on completion we
    // push the bag in so it can re-render with rails filtered.
    _fetchReceiverPrefs(identity.device_id).then((prefs) => {
      if (!prefs) return;
      _receiverPrefs = prefs;
      // Re-push to whatever's mounted in main now (cast-home, most
      // likely). Iframe surfaces that don't care just ignore it.
      const main = _findSurfaceInSlot('main');
      if (main?.el?.tagName === 'IFRAME') _postPrefsTo(main.el);
    });
    // Wake-on-LAN auto-fill: the Android TV bundle's AugmentumTV bridge
    // exposes the device's primary network info (MAC + LAN IP) so the
    // server can store it on the trusted_receivers row and the user can
    // wake this TV later from cast-control without manually typing the
    // MAC. Browser-tab receivers don't have the bridge (and browsers
    // never expose MAC for privacy reasons) — they leave the fields
    // empty and the user enters the MAC by hand.
    const net = _networkInfo();
    send({
      type: 'event', event: 'ready',
      data: {
        platform: identity.platform,
        device_id: identity.device_id,
        user_agent: navigator.userAgent.slice(0, 200),
        screen: { w: window.screen.width, h: window.screen.height, dpr: window.devicePixelRatio || 1 },
        input_devices: ['touch'],
        hw_accel: _detectHwAccel(),
        surface_capabilities: {
          'html.generic':       { schema_version: 1 },
          'media.image':        { schema_version: 1 },
          'media.video':        { schema_version: 1 },
          'media.audio':        { schema_version: 1 },
        },
        label: identity.label,
        mac_address: net.mac,
        local_ip: net.ip,
      },
    });
  });

  ws.addEventListener('message', (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg && msg.type === 'cmd') handleCmd(msg);
  });

  ws.addEventListener('close', (ev) => {
    const code = ev && typeof ev.code === 'number' ? ev.code : 0;
    const reason = ev && ev.reason ? String(ev.reason) : '';
    console.log('[cast-receiver] ws closed', { code, reason });
    setStatus('reconnecting');
    // Tear down active surfaces so a stale frame can't linger across the
    // gap; the cookie reconnect re-mounts cast-home on open.
    for (const sid of Array.from(surfaces.keys())) closeSurface(sid, 'disconnected');

    // If we already received CMD_REVOKED, _terminallyStopped is set and
    // the shell shows the revoke explanation — don't clobber it.
    if (_terminallyStopped) return;

    // Terminal close codes mean re-pairing won't help. Show the actual
    // close reason instead of looping silently.
    if (TERMINAL_CLOSE_CODES.has(code)) {
      const msg = code === 4003
        ? _buildRevokedMessage('', 'revoked')
        : `Receiver auth failed (close ${code}${reason ? ` — ${reason}` : ''}). Reload the page to retry.`;
      _stopTerminally(msg);
      return;
    }

    // Auth-reject: the session cookie is gone/expired, so a silent
    // reconnect can't recover — fall back to QR re-pairing.
    if (AUTH_REPAIR_CLOSE_CODES.has(code)) {
      showShellPlaceholder('Session expired — re-pair from your phone…', true);
      reconnectTimer = setTimeout(startPair, RECONNECT_DELAY_MS);
      return;
    }

    // Handshake/auth failure OR server-unreachable: a cookie-only reconnect
    // (wsToken == null) that closed WITHOUT ever reaching 'open' surfaces as a
    // bare 1006 — the browser can't tell "server rejected my cookie pre-accept"
    // (auth dead → re-pair) from "server is down/restarting" (cookie fine →
    // keep waiting). Counting never-opened closes as auth failures was the
    // bug: a server restart takes longer than a few retries, so a valid HOME
    // TV re-paired needlessly on every restart. Disambiguate with an HTTP
    // probe — HTTP delivers real status codes where a pre-accept WS can't.
    if (!wsToken && !_wsOpenedThisAttempt) {
      _consecutiveHandshakeFailures += 1;
      if (_consecutiveHandshakeFailures >= MAX_HANDSHAKE_FAILURES_BEFORE_REPAIR) {
        _consecutiveHandshakeFailures = 0;
        _probeAuthThenReconnectOrRepair(reason);
        return;
      }
    }

    // Any other drop (server restart, network blip) → reconnect silently
    // with the persistent cookie, backing off but retrying indefinitely so
    // the TV stays a live cast target for the controller + companion.
    _reconnectSilently(reason);
  });

  ws.addEventListener('error', (ev) => {
    // The browser never tells us anything useful in 'error' (security
    // boundary); we just log presence so timing is visible in the
    // console next to the close event.
    console.warn('[cast-receiver] ws error event', ev && ev.type);
  });
}

// Silently reconnect with the persistent cookie, backing off but retrying
// indefinitely so the TV rides out a server restart / network blip and stays
// a live cast target for the controller + companion.
function _reconnectSilently(reason) {
  _reconnectAttempts += 1;
  const delay = Math.min(
    MAX_RECONNECT_DELAY_MS,
    RECONNECT_DELAY_MS * 2 ** Math.min(_reconnectAttempts - 1, 4),
  );
  showShellPlaceholder(`Reconnecting${reason ? ` (${reason})` : ''}…`, true);
  reconnectTimer = setTimeout(() => connectWS(null), delay);
}

// Disambiguate a run of never-opened cookie reconnects: only an actual auth
// rejection should drop us to QR re-pairing. HTTP carries real status codes
// where a pre-accept WS close can't, so probe /api/auth/me:
//   401/403 → the session cookie is genuinely dead → re-pair (unavoidable).
//   200     → cookie valid + server up → the WS drop wasn't auth → keep
//             reconnecting (a home TV must not re-pair on a transient blip).
//   network error → server still unreachable (restarting) → keep waiting;
//             re-pairing wouldn't reach the server anyway.
// This is the fix for "every server restart forces a re-pair even on home":
// a restart takes longer than the handshake-failure window, and the old code
// read that as an auth failure.
async function _probeAuthThenReconnectOrRepair(reason) {
  let status = 0;
  try {
    const r = await fetch('/api/auth/me', { credentials: 'same-origin', cache: 'no-store' });
    status = r.status;
  } catch {
    status = 0;  // server unreachable → NOT an auth failure
  }
  if (status === 401 || status === 403) {
    showShellPlaceholder('Not signed in on this TV — re-pair from your phone…', true);
    reconnectTimer = setTimeout(startPair, RECONNECT_DELAY_MS);
    return;
  }
  _reconnectSilently(reason);
}


/* ── Terminal-error handling ──────────────────────────────────────
 *
 * When the server tells us to stop (CMD_REVOKED or a terminal close
 * code), we tear down timers + freeze the receiver page on a clear
 * placeholder. Looping silently was the original bug: the server kept
 * killing every reconnect with 4003, the receiver kept re-pairing,
 * and the user saw a flicker between QR and disconnect with no
 * explanation of why. */

function _stopTerminally(message) {
  _terminallyStopped = true;
  clearTimeout(reconnectTimer);
  clearInterval(pollTimer);
  reconnectTimer = null;
  pollTimer = null;
  showShellPlaceholder(message, true);
  setStatus('');
}

function _buildRevokedMessage(trustedId, reason) {
  // trusted_id is shown to help the user find the right row in the
  // Settings → TVs list; harmless to leak — it's already in their own
  // account's UI. reason is currently always "revoked" or "unknown"
  // (the upstream fall-through case when get_by_device returns null).
  const tail = reason === 'revoked'
    ? 'Open Settings → TVs on a logged-in browser, find this device under Revoked, and tap Restore. Then reload this page.'
    : 'The server rejected this device. Open Settings → TVs on a logged-in browser to inspect — then reload this page.';
  const idLine = trustedId ? ` (device id ${trustedId})` : '';
  return `This receiver was revoked${idLine}. ${tail}`;
}


function _detectHwAccel() {
  const accel = [];
  try {
    const canvas = document.createElement('canvas');
    if (canvas.getContext('webgl2')) accel.push('webgl2');
    else if (canvas.getContext('webgl')) accel.push('webgl');
  } catch {}
  return accel;
}


function _deriveLabel() {
  const ua = navigator.userAgent.toLowerCase();
  if (ua.includes('augmentumtvreceiver')) return 'Android TV';
  if (ua.includes('googletv') || ua.includes('android tv')) return 'Google TV';
  if (ua.includes('android')) return 'Android Browser';
  if (ua.includes('tv') || ua.includes('webos') || ua.includes('tizen')) return 'Smart TV';
  return 'Browser Receiver';
}


/**
 * _resolveDeviceIdentity — pick the stable identity for this receiver.
 *
 * Three sources, in priority order:
 *
 *   1. URL query — the native Android TV shell passes
 *      ``?device_id=<uuid>&platform=android-tv&label=...``.
 *      That UUID is generated once on first launch and pinned in
 *      SharedPreferences, so it survives reboots + reinstalls of the
 *      receiver page itself (the WebView).
 *
 *   2. localStorage — for browser-tab receivers we still want a soft
 *      "remember me" so a user who opens the cast-receiver page on
 *      the same browser gets the same trusted row, not a new one
 *      every refresh. Cleared by Clear Site Data; that's fine because
 *      browser tabs are inherently more ephemeral than dedicated
 *      receiver hardware.
 *
 *   3. None — for receivers that decline storage (incognito etc).
 *      The server treats empty device_id as ephemeral and skips the
 *      trusted_receivers upsert (intentional — see receiver_registry).
 */
function _resolveDeviceIdentity() {
  const params = new URLSearchParams(location.search);
  const urlDeviceId = (params.get('device_id') || '').trim();
  const urlPlatform = (params.get('platform') || '').trim();
  const urlLabel = (params.get('label') || '').trim();

  if (urlDeviceId) {
    return {
      device_id: urlDeviceId,
      platform: urlPlatform || 'android-tv',
      label: urlLabel || _deriveLabel(),
    };
  }

  const LS_KEY = 'augmentum.cast.device_id';
  let stored = '';
  try {
    stored = localStorage.getItem(LS_KEY) || '';
    if (!stored) {
      stored = _generateUuid();
      localStorage.setItem(LS_KEY, stored);
    }
  } catch {
    // Storage disabled (cookies off, sandboxed iframe, etc.) — fall
    // back to ephemeral identity. The receiver still works; it just
    // won't get a persistent trusted_receivers row.
    stored = '';
  }

  return {
    device_id: stored,
    platform: 'browser',
    label: _deriveLabel(),
  };
}


function _generateUuid() {
  // Prefer crypto.randomUUID where available (HTTPS or localhost).
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  // Fallback: 16 random bytes formatted as a v4 UUID.
  const buf = new Uint8Array(16);
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(buf);
  } else {
    for (let i = 0; i < 16; i++) buf[i] = Math.floor(Math.random() * 256);
  }
  buf[6] = (buf[6] & 0x0f) | 0x40;
  buf[8] = (buf[8] & 0x3f) | 0x80;
  const h = Array.from(buf, b => b.toString(16).padStart(2, '0'));
  return `${h.slice(0, 4).join('')}-${h.slice(4, 6).join('')}-${h.slice(6, 8).join('')}-${h.slice(8, 10).join('')}-${h.slice(10).join('')}`;
}


// Boot: try a cookie-authenticated reconnect first. A previously-paired
// home TV holds a long-lived session cookie, so it re-attaches silently
// on power-on / app relaunch with no QR. A fresh (or expired) TV gets a
// 4001 close, which the handler turns into QR pairing.
connectWS(null);
