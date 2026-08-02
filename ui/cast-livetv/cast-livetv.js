/**
 * cast-livetv.js — TV Live TV surface.
 *
 * Loaded inside the cast-receiver iframe when the controller casts a
 * Live TV channel. The receiver's iframe carries the user's session
 * cookies, so this surface authenticates the same way the browser
 * does — POST /api/livetv/play to mint a session, hls.js (vendored
 * at /ui/lib/hls.js) for playback, POST /api/livetv/stop on tear-
 * down so the upstream tuner releases promptly.
 *
 * URL params:
 *   ?server_id=…    media-server id (Emby/JF)
 *   ?channel_id=…   channel external id
 *   ?title=…        channel name to surface in the HUD
 *   ?number=…       channel number for the HUD chip
 *
 * Patches accepted via postMessage from cast-receiver (the
 * controller sends them as ``{type:'augmentum.surface_state', patch:
 * {paused, volume, muted}}``). Seek / position are intentionally
 * absent — live TV has no meaningful seek bar from a cable / OTA
 * feed; if/when DVR-buffer windows ship, they'll land as a separate
 * patch shape.
 */

const params = new URLSearchParams(location.search);
const SERVER_ID  = (params.get('server_id')  || '').trim();
const CHANNEL_ID = (params.get('channel_id') || '').trim();
const TITLE      = (params.get('title')      || 'Live TV').trim();
const NUMBER     = (params.get('number')     || '').trim();
const LOGO_URL   = (params.get('logo_url')   || '').trim();
const NOW_TEXT   = (params.get('now')        || '').trim();

const HUD_AUTO_HIDE_MS = 4000;

const $ = (sel) => document.querySelector(sel);
const elVideo    = $('[data-cl-video]');
const elHud      = $('[data-cl-hud]');
const elLogo     = $('[data-cl-logo]');
const elFallback = $('[data-cl-fallback]');
const elNumber   = $('[data-cl-number]');
const elName     = $('[data-cl-name]');
const elNow      = $('[data-cl-now]');
const elStatus   = $('[data-cl-status]');

let _sessionToken = '';
let _hls = null;
let _hudTimer = null;
let _firstFrameShown = false;
let _stopping = false;

// surface_id assigned by cast-receiver via augmentum.surface_init.
// We don't currently echo surface_state events upstream (the
// controller's UI for Live TV is volume + mute + close — no
// transport bar to keep in sync), but the id is captured for
// future use.
let _surfaceId = '';


/* ── HUD helpers ─────────────────────────────────────────────── */

function showHud(ms = HUD_AUTO_HIDE_MS) {
  if (!elHud) return;
  elHud.classList.add('visible');
  clearTimeout(_hudTimer);
  _hudTimer = setTimeout(() => elHud.classList.remove('visible'), ms);
}

function setStatus(msg) {
  if (!elStatus) return;
  elStatus.textContent = msg || '';
  elStatus.classList.toggle('hidden', !msg);
}


/* ── Boot ────────────────────────────────────────────────────── */

function _initHud() {
  elName.textContent = TITLE || 'Live TV';
  elNumber.textContent = NUMBER || '';
  elNumber.style.display = NUMBER ? '' : 'none';
  if (NOW_TEXT) {
    elNow.textContent = NOW_TEXT;
    elNow.hidden = false;
  }
  if (LOGO_URL) {
    elLogo.src = LOGO_URL;
    elLogo.onerror = () => {
      elLogo.hidden = true;
      elFallback.hidden = false;
      elFallback.textContent = _initialsFor(TITLE);
    };
  } else {
    elLogo.hidden = true;
    elFallback.hidden = false;
    elFallback.textContent = _initialsFor(TITLE);
  }
  showHud(6000);  // longer initial reveal — viewer wants confirmation
}

function _initialsFor(name) {
  const cleaned = String(name || '').replace(/[^a-zA-Z0-9 ]/g, ' ').trim();
  if (!cleaned) return '#';
  const parts = cleaned.split(/\s+/);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[1][0]).toUpperCase();
}

async function _startPlayback() {
  if (!SERVER_ID || !CHANNEL_ID) {
    setStatus('Missing server or channel — cannot start playback.');
    return;
  }
  setStatus('Starting Live TV…');
  try {
    const playUrl = `/api/livetv/play/${encodeURIComponent(SERVER_ID)}`
      + `/${encodeURIComponent(CHANNEL_ID)}?title=${encodeURIComponent(TITLE)}`;
    const r = await fetch(playUrl, {
      method: 'POST',
      credentials: 'same-origin',
    });
    if (!r.ok) {
      setStatus(r.status === 502
        ? 'Tuner busy or upstream returned no playable source.'
        : `Could not start playback (HTTP ${r.status}).`);
      return;
    }
    const data = await r.json();
    if (!data?.session_token || !data?.manifest_url) {
      setStatus('Playback response missing session info.');
      return;
    }
    _sessionToken = data.session_token;
    _attachHls(data.manifest_url);
  } catch (err) {
    console.error('[cast-livetv] play failed', err);
    setStatus('Could not start playback.');
  }
}

function _attachHls(manifestUrl) {
  const Hls = window.Hls;
  if (Hls?.isSupported()) {
    _hls = new Hls({
      lowLatencyMode: false,
      enableWorker:   true,
      backBufferLength: 30,
      // Cold-tuner warmup tolerance — Emby's live transcoder blocks
      // the variant playlist (live.m3u8) until the first segment is
      // ready, which can take 20-30s on MPEG-2 / AC-3 source. Default
      // hls.js timeouts (10s manifest/level, 20s fragment) trip
      // levelLoadTimeOut before the upstream produces anything.
      manifestLoadingTimeOut: 30000,
      manifestLoadingMaxRetry: 2,
      levelLoadingTimeOut:    45000,
      levelLoadingMaxRetry:   2,
      fragLoadingTimeOut:     45000,
      fragLoadingMaxRetry:    4,
    });
    _hls.on(Hls.Events.ERROR, (_evt, data) => {
      if (data?.fatal) {
        console.warn('[cast-livetv] fatal hls error', data);
        setStatus(`Stream error: ${data.type || 'unknown'}`);
      }
    });
    _hls.loadSource(manifestUrl);
    _hls.attachMedia(elVideo);
  } else if (elVideo.canPlayType('application/vnd.apple.mpegurl')) {
    // Native HLS (Safari / iOS — receivers running tvOS / iPadOS).
    elVideo.src = manifestUrl;
  } else {
    setStatus('This browser does not support HLS playback.');
  }

  elVideo.addEventListener('playing', () => {
    if (!_firstFrameShown) {
      _firstFrameShown = true;
      setStatus('');
    }
  });
  elVideo.addEventListener('error', () => {
    setStatus(`Video error (code ${elVideo.error?.code || '?'}).`);
  });
}


/* ── Tear-down ───────────────────────────────────────────────── */

async function _teardown() {
  if (_stopping) return;
  _stopping = true;
  try {
    if (_hls) {
      _hls.destroy();
      _hls = null;
    }
  } catch { /* */ }
  try {
    elVideo.pause();
    elVideo.removeAttribute('src');
    elVideo.load();
  } catch { /* */ }
  if (_sessionToken) {
    // keepalive so the request survives the iframe being torn down
    // by the receiver (cast-receiver calls el.remove() on surface
    // close, which detaches our async fetch's window).
    try {
      await fetch(`/api/livetv/stop/${encodeURIComponent(_sessionToken)}`, {
        method: 'POST',
        credentials: 'same-origin',
        keepalive: true,
      });
    } catch { /* server-side idle sweep is the safety net */ }
  }
}

// Receiver tears the iframe down on surface_close. ``pagehide``
// fires reliably in that path (more so than ``unload`` which is
// inconsistent across browsers).
window.addEventListener('pagehide', () => { void _teardown(); });


/* ── Patch handler ───────────────────────────────────────────── */

function handlePatch(patch) {
  if (!patch || typeof patch !== 'object' || !elVideo) return;
  try {
    if (typeof patch.paused === 'boolean') {
      if (patch.paused) elVideo.pause();
      else elVideo.play().catch(() => {});
      showHud();
    }
    if (typeof patch.volume === 'number') {
      elVideo.volume = Math.max(0, Math.min(1, patch.volume));
      showHud();
    }
    if (typeof patch.muted === 'boolean') {
      elVideo.muted = patch.muted;
      showHud();
    }
    // Channel-change patch: receiver-side fast-zap without tearing
    // down the surface. Drops the current session and starts a new
    // one. Optional — controller may also choose to close + reopen
    // the surface, which is simpler.
    if (patch.channel_id && patch.server_id) {
      _changeChannel(patch.server_id, patch.channel_id, patch.title || '');
    }
  } catch (err) {
    console.warn('[cast-livetv] patch apply failed', err);
  }
}

async function _changeChannel(serverId, channelId, title) {
  // Stop the current session up at upstream; then start a new one in
  // place. The video element is reused so the TV doesn't black-flash
  // through a full iframe reload.
  const previousToken = _sessionToken;
  _sessionToken = '';
  _firstFrameShown = false;
  setStatus(`Changing channel to ${title || channelId}…`);

  try {
    if (previousToken) {
      await fetch(`/api/livetv/stop/${encodeURIComponent(previousToken)}`, {
        method: 'POST',
        credentials: 'same-origin',
        keepalive: true,
      });
    }
    if (_hls) {
      _hls.destroy();
      _hls = null;
    }
    const r = await fetch(
      `/api/livetv/play/${encodeURIComponent(serverId)}/${encodeURIComponent(channelId)}`
      + `?title=${encodeURIComponent(title || '')}`,
      { method: 'POST', credentials: 'same-origin' },
    );
    if (!r.ok) {
      setStatus(`Channel change failed (HTTP ${r.status}).`);
      return;
    }
    const data = await r.json();
    _sessionToken = data.session_token;
    elName.textContent = title || channelId;
    showHud();
    _attachHls(data.manifest_url);
  } catch (err) {
    console.error('[cast-livetv] channel change failed', err);
    setStatus('Channel change failed.');
  }
}

window.addEventListener('message', (ev) => {
  const data = ev.data;
  if (!data || typeof data !== 'object') return;
  if (data.type === 'augmentum.surface_init') {
    _surfaceId = data.surface_id || '';
  } else if (data.type === 'augmentum.surface_state') {
    handlePatch(data.patch);
  }
});


/* ── Kick off ────────────────────────────────────────────────── */

_initHud();
void _startPlayback();
