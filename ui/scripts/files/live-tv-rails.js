/**
 * Files panel — Live TV rail browser + HLS player.
 *
 * Renders the YouTube-TV-style horizontal rails returned by
 * ``GET /api/livetv/rails`` when the Live TV chip is active. Each
 * rail scrolls independently; each tile shows the network logo
 * (theme-aware: LogoDark on light themes, LogoLight on dark) +
 * channel number + the EPG ``CurrentProgram`` if one is airing.
 *
 * Mirrors ``comics.js`` 's four-export contract so ``files/index.js``
 * can delegate uniformly:
 *
 *     isLiveTvChipActive()       predicate on currentScope+source
 *     renderLiveTvRails()        main render into ``state.el.grid``
 *     resetLiveTvView()          cleanup on chip leave
 *     initLiveTvListeners()      one-time click/key wiring
 *
 * Tile-click play path:
 *
 *   1. POST /api/livetv/play/{server_id}/{channel_id}
 *      → mint a session token + receive a proxy manifest URL
 *   2. Lazy-load hls.js (vendored under /ui/lib/hls.js)
 *   3. Open a modal player; HLS handles segment fetching through
 *      our proxy (browser never sees the Emby api_key)
 *   4. On close: POST /api/livetv/stop/{token} so the upstream
 *      tuner releases promptly instead of timing out
 */

import { escapeHtml, showToast } from '../app.js';
import { state } from './state.js';

const RAILS_ENDPOINT = '/api/livetv/rails';
// Hard cap so a rail with 200 ESPN-affiliate channels doesn't
// dominate. Users still get "All Channels" for full discovery.
const TILES_PER_RAIL_CAP = 40;
const HLS_SCRIPT_SRC = '/ui/lib/hls.js/hls.min.js';

let _listenersWired = false;
let _renderToken = 0;          // race-guard for overlapping fetches
let _hlsLoadingPromise = null; // single-flight script loader
let _activePlayer = null;      // {sessionToken, hls, video, root}


// ── Public contract ────────────────────────────────────────────────

export function isLiveTvChipActive() {
  return state.currentScope === 'cloud' && state.currentSource === 'live_tv';
}

export function resetLiveTvView() {
  // No drill-down state to clear yet; the rails are flat. Stub kept
  // so the index.js dispatch surface stays symmetric with comics.
}

/**
 * Companion-reachable live TV play — same path as clicking a channel tile
 * in the Files panel. POSTs a play session to the stream proxy, then opens
 * the HLS player overlay. Does NOT require the Files panel or Live TV chip
 * to be active — the overlay appends to document.body.
 *
 * @param {{serverId: string, channelId: string, name?: string}} params
 * @returns {Promise<boolean>} true if playback started, false on error
 */
export async function playLiveTvChannel({ serverId, channelId, name }) {
  if (!channelId || !serverId) {
    console.warn('[live-tv] playLiveTvChannel: missing serverId or channelId');
    return false;
  }
  const label = name || channelId;
  try {
    const playUrl = `/api/livetv/play/${encodeURIComponent(serverId)}`
      + `/${encodeURIComponent(channelId)}?title=${encodeURIComponent(label)}`;
    const r = await fetch(playUrl, {
      method: 'POST',
      credentials: 'same-origin',
    });
    if (!r.ok) {
      const msg = r.status === 502
        ? 'Tuner busy or upstream returned no playable source.'
        : `Could not start playback (HTTP ${r.status}).`;
      showToast(msg, 'error');
      return false;
    }
    const data = await r.json();
    if (!data?.session_token || !data?.manifest_url) {
      showToast('Playback response missing session info.', 'error');
      return false;
    }
    await _openPlayer({
      sessionToken: data.session_token,
      manifestUrl:  data.manifest_url,
      title:        data.title || label,
    });
    return true;
  } catch (err) {
    console.error('[live-tv] playLiveTvChannel failed', err);
    showToast('Could not start playback.', 'error');
    return false;
  }
}

/** Entry point called by loadFiles when the Live TV chip is active. */
export async function renderLiveTvRails() {
  const grid = state.el.grid;
  if (!grid) return;

  grid.className = 'files-grid files-grid-live-tv';
  grid.innerHTML = `<div class="files-live-tv-loading">Loading channels…</div>`;

  // Each invocation bumps the token; a slower in-flight fetch
  // resolving after a newer one won't overwrite the fresher UI.
  const myToken = ++_renderToken;
  let data;
  try {
    const r = await fetch(RAILS_ENDPOINT, { credentials: 'same-origin' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (err) {
    if (myToken !== _renderToken) return;
    grid.innerHTML = `
      <div class="files-empty">
        <span class="files-empty-icon">\u{1F4FA}</span>
        <span class="files-empty-text">Couldn't load Live TV</span>
        <span class="files-empty-hint">${escapeHtml(String(err?.message || err))}</span>
      </div>
    `;
    return;
  }
  if (myToken !== _renderToken) return;

  const rails = Array.isArray(data?.rails) ? data.rails : [];
  if (rails.length === 0) {
    grid.innerHTML = `
      <div class="files-empty">
        <span class="files-empty-icon">\u{1F4FA}</span>
        <span class="files-empty-text">No Live TV channels yet</span>
        <span class="files-empty-hint">
          Connect an Emby or Jellyfin server with Live TV configured
          (HDHomeRun, M3U source, or external guide). The Files panel
          will surface a categorized channel browser here.
        </span>
      </div>
    `;
    return;
  }

  grid.innerHTML = rails.map(_renderRail).join('');
  // Mount cast-buttons after the rails commit to the DOM. Lazy import
  // inside _mountCastButtons keeps the cast bundle out of the bootup
  // path for users who never open this chip.
  void _mountCastButtons();
}


// ── Render helpers ─────────────────────────────────────────────────

function _renderRail(rail) {
  const tiles = (rail.channels || [])
    .slice(0, TILES_PER_RAIL_CAP)
    .map(_renderTile)
    .join('');
  const overflow = (rail.channels?.length || 0) > TILES_PER_RAIL_CAP
    ? `<span class="files-live-rail-overflow">+${rail.channels.length - TILES_PER_RAIL_CAP} more in All Channels</span>`
    : '';
  return `
    <section class="files-live-rail" data-rail-id="${escapeHtml(rail.id)}" data-rail-kind="${escapeHtml(rail.kind)}">
      <header class="files-live-rail-head">
        <h3 class="files-live-rail-title">${escapeHtml(rail.title)}</h3>
        <span class="files-live-rail-count">${(rail.channels || []).length} channels</span>
      </header>
      <div class="files-live-rail-strip" role="list">
        ${tiles}
        ${overflow}
      </div>
    </section>
  `;
}

function _renderTile(ch) {
  const channelNum = (ch.channel_number || '').trim();
  const program    = ch.current_program;
  const nowName    = program?.name ? String(program.name) : '';

  // Theme-aware logo: dark themes get the LogoLight variant (light-
  // on-dark logo) when upstream advertises one, falling back to the
  // Primary variant; light themes get LogoDark or Primary. The
  // single URL is resolved here at render time — the CSS-attr-swap
  // approach (``content: attr(... url)``) is technically defined
  // but has poor browser support in practice, so we pick once and
  // re-render on theme change via the same chip-render path.
  const theme = (document.documentElement.getAttribute('data-theme') || 'light').toLowerCase();
  const preferred = theme === 'dark' ? 'light' : 'dark';
  const logoUrl =
       (ch[`has_logo_${preferred}`] ? _logoUrlFor(ch, preferred) : '')
    || _logoUrlFor(ch, 'primary');

  // "Live now" bar — present only when EPG gave us a current program.
  // The soft pulse class is added unconditionally on live tiles since
  // every live channel IS broadcasting, even if EPG metadata is missing.
  const nowBar = nowName ? `
    <div class="files-live-tile-now">
      <span class="files-live-tile-now-dot" aria-hidden="true"></span>
      <span class="files-live-tile-now-text" title="${escapeHtml(nowName)}">${escapeHtml(nowName)}</span>
    </div>
  ` : `
    <div class="files-live-tile-now files-live-tile-now-empty">
      <span class="files-live-tile-now-dot" aria-hidden="true"></span>
      <span class="files-live-tile-now-text">Live</span>
    </div>
  `;

  const logoImg = logoUrl ? `
    <img class="files-live-tile-logo"
         src="${escapeHtml(logoUrl)}"
         alt="${escapeHtml(ch.name)}"
         loading="lazy"
         onerror="this.classList.add('files-live-tile-logo-missing')">
  ` : `
    <div class="files-live-tile-logo files-live-tile-logo-fallback">
      ${escapeHtml(_initialsFor(ch.name))}
    </div>
  `;

  // ``role="button"`` + ``tabindex="0"`` preserves keyboard semantics
  // we lost by moving from <button> to <div> (had to drop the button
  // so a nested cast-button doesn't violate "no buttons inside
  // buttons"). The init listener handles Enter / Space.
  return `
    <div class="files-live-tile cast-btn-host"
         role="button"
         tabindex="0"
         data-channel-id="${escapeHtml(ch.external_id)}"
         data-server-id="${escapeHtml(ch.server_id || '')}"
         data-channel-name="${escapeHtml(ch.name)}"
         data-channel-number="${escapeHtml(channelNum)}"
         data-channel-now="${escapeHtml(nowName)}"
         data-logo-url="${escapeHtml(logoUrl)}"
         title="${escapeHtml(ch.name)}${nowName ? ' — ' + escapeHtml(nowName) : ''}">
      <div class="files-live-tile-logo-frame">
        ${logoImg}
        ${channelNum ? `<span class="files-live-tile-number">${escapeHtml(channelNum)}</span>` : ''}
      </div>
      <div class="files-live-tile-name">${escapeHtml(ch.name)}</div>
      ${nowBar}
    </div>
  `;
}


/** Cast-button overlay mounted on every tile after render. Uses
 *  capability ``media.video_play@1`` with a ``livetv:`` content
 *  key so the picker dispatches to the cast-livetv surface rather
 *  than the file-id-driven cast-video surface. */
async function _mountCastButtons() {
  const grid = state.el.grid;
  if (!grid) return;
  const tiles = grid.querySelectorAll('.files-live-tile');
  if (!tiles.length) return;
  // Lazy-load cast-button only when the Live TV chip actually
  // renders rails. Avoids dragging the cast dep into pages that
  // never touch this surface.
  let mountCastButton;
  try {
    ({ mountCastButton } = await import('../cast-button.js'));
  } catch (err) {
    console.warn('[live-tv] cast-button import failed', err);
    return;
  }
  tiles.forEach((tile) => {
    if (tile.querySelector('.cast-btn')) return;  // idempotent
    const channelId = tile.dataset.channelId;
    const serverId  = tile.dataset.serverId;
    const name      = tile.dataset.channelName || '';
    const number    = tile.dataset.channelNumber || '';
    const nowText   = tile.dataset.channelNow || '';
    const logoUrl   = tile.dataset.logoUrl || '';
    if (!channelId || !serverId) return;

    const castBtn = mountCastButton({
      capability: 'media.video_play@1',
      size: 'sm',
      className: 'cast-btn-on-image cast-btn-hover-reveal files-live-tile-cast',
      title: `Cast ${name} to TV`,
      getContent: () => ({
        title:      name,
        // The receiver surface mints its own session; the URL
        // here is informational only (cast-picker uses contentKey
        // + metadata to compose the surface URL).
        contentKey: `livetv:${serverId}:${channelId}`,
        metadata: {
          server_id:  serverId,
          channel_id: channelId,
          number,
          now:        nowText,
          logo_url:   logoUrl,
        },
      }),
    });
    tile.appendChild(castBtn);
  });
}

function _logoUrlFor(ch, variant) {
  // The server_id + channel id pair lets the route proxy through the
  // right Emby/JF without ever exposing the user's media-server token
  // to the browser. ``has_logo_*`` from the rails payload tells us
  // which variants the upstream actually has — calling for a missing
  // variant would 404. Skip the URL entirely so we render the
  // fallback tile rather than a broken-image icon.
  const flagMap = {
    primary: ch.has_logo_primary,
    light:   ch.has_logo_light,
    dark:    ch.has_logo_dark,
  };
  if (!flagMap[variant]) return '';
  if (!ch.server_id || !ch.external_id) return '';
  return `/api/livetv/logo/${encodeURIComponent(ch.server_id)}/`
    + `${encodeURIComponent(ch.external_id)}?variant=${encodeURIComponent(variant)}`;
}

function _initialsFor(name) {
  // Channel-name fallback when no logo is available. Two characters
  // is the sweet spot — "ESPN" → "ES", "Cartoon Network" → "CN".
  const cleaned = String(name || '').replace(/[^a-zA-Z0-9 ]/g, ' ').trim();
  if (!cleaned) return '#';
  const parts = cleaned.split(/\s+/);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[1][0]).toUpperCase();
}


// ── Listeners ──────────────────────────────────────────────────────

export function initLiveTvListeners() {
  if (_listenersWired) return;
  const panel = state.el.panel;
  if (!panel) return;
  _listenersWired = true;

  panel.addEventListener('click', (e) => {
    if (!isLiveTvChipActive()) return;
    const tile = e.target.closest('.files-live-tile');
    if (!tile) return;
    _onTileClick(tile);
  });

  // Keyboard activation parity with the old <button> tile. Enter and
  // Space both fire the click handler so screen-reader / keyboard
  // users can play a channel without grabbing the pointer.
  panel.addEventListener('keydown', (e) => {
    if (!isLiveTvChipActive()) return;
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const tile = e.target.closest('.files-live-tile');
    if (!tile) return;
    e.preventDefault();
    _onTileClick(tile);
  });
}

async function _onTileClick(tile) {
  const channelId = tile.dataset.channelId;
  const serverId  = tile.dataset.serverId;
  const name      = tile.dataset.channelName || 'channel';
  if (!channelId || !serverId) return;

  // Visible busy state on the tile itself — the play call may take
  // a few seconds while Emby spins up the transcoder.
  tile.classList.add('files-live-tile-loading');
  try {
    await playLiveTvChannel({ serverId, channelId, name });
  } finally {
    tile.classList.remove('files-live-tile-loading');
  }
}


// ── Player overlay ─────────────────────────────────────────────────

/** Load hls.js once and cache the promise. Subsequent calls resolve
 *  immediately. ``window.Hls`` is the global the library exposes. */
function _ensureHlsLoaded() {
  if (typeof window.Hls !== 'undefined') return Promise.resolve();
  if (_hlsLoadingPromise) return _hlsLoadingPromise;
  _hlsLoadingPromise = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = HLS_SCRIPT_SRC;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => {
      _hlsLoadingPromise = null;
      reject(new Error('hls.js failed to load'));
    };
    document.head.appendChild(s);
  });
  return _hlsLoadingPromise;
}

async function _openPlayer({ sessionToken, manifestUrl, title }) {
  // Tear down any prior session before starting a new one — both
  // server-side (release upstream tuner) and client-side (free the
  // video element + hls instance). Multiple concurrent live streams
  // would double-bill the tuner allocation on single-tuner setups.
  await _closeActivePlayer();

  const root = document.createElement('div');
  root.className = 'live-tv-player-overlay';
  root.innerHTML = `
    <div class="live-tv-player-frame" role="dialog" aria-label="Live TV">
      <button type="button" class="live-tv-player-close" aria-label="Close">✕</button>
      <div class="live-tv-player-title">${escapeHtml(title || 'Live TV')}</div>
      <video class="live-tv-player-video" playsinline controls autoplay></video>
      <div class="live-tv-player-status"></div>
    </div>
  `;
  document.body.appendChild(root);

  const video    = root.querySelector('video');
  const closeBtn = root.querySelector('.live-tv-player-close');
  const status   = root.querySelector('.live-tv-player-status');

  _activePlayer = { sessionToken, hls: null, video, root };

  const handleClose = () => { void _closeActivePlayer(); };
  closeBtn.addEventListener('click', handleClose);
  root.addEventListener('click', (e) => {
    // Click on the dim backdrop (the overlay root, not the frame)
    // closes — same affordance as the lightbox.
    if (e.target === root) handleClose();
  });
  document.addEventListener('keydown', _onPlayerKey);

  try {
    await _ensureHlsLoaded();
  } catch (err) {
    status.textContent = 'Could not load HLS player library.';
    return;
  }
  if (!_activePlayer || _activePlayer.sessionToken !== sessionToken) {
    // User closed while hls.js was loading. Bail.
    return;
  }

  const Hls = window.Hls;
  if (Hls?.isSupported()) {
    const hls = new Hls({
      // Live windows are short; keep buffer small so seek-to-live
      // re-syncs quickly. Burst-on-drift is the default in hls.js
      // 1.5+ and what we want for live.
      lowLatencyMode: false,
      enableWorker:   true,
      backBufferLength: 30,
      // Cold-tuner warmup tolerance — Emby's live transcoder blocks
      // the variant playlist until the first segment is ready, which
      // on a cold MPEG-2 / AC-3 source takes 20-30s. Default hls.js
      // 10s timeouts trip levelLoadTimeOut before the upstream
      // produces anything. See cast-livetv.js for the matching set.
      manifestLoadingTimeOut: 30000,
      manifestLoadingMaxRetry: 2,
      levelLoadingTimeOut:    45000,
      levelLoadingMaxRetry:   2,
      fragLoadingTimeOut:     45000,
      fragLoadingMaxRetry:    4,
    });
    _activePlayer.hls = hls;
    hls.on(Hls.Events.ERROR, (_evt, data) => {
      if (data?.fatal) {
        console.warn('[live-tv] fatal hls error', data);
        status.textContent = `Stream error: ${data.type || 'unknown'}`;
      }
    });
    hls.loadSource(manifestUrl);
    hls.attachMedia(video);
  } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
    // Safari / iOS — native HLS, no hls.js needed.
    video.src = manifestUrl;
  } else {
    status.textContent = 'This browser does not support HLS playback.';
  }
}

function _onPlayerKey(e) {
  if (e.key === 'Escape') {
    void _closeActivePlayer();
  }
}

async function _closeActivePlayer() {
  const player = _activePlayer;
  _activePlayer = null;
  document.removeEventListener('keydown', _onPlayerKey);
  if (!player) return;

  try {
    if (player.hls) {
      player.hls.destroy();
    }
  } catch (err) {
    console.warn('[live-tv] hls destroy failed', err);
  }
  try {
    if (player.video) {
      player.video.pause();
      player.video.removeAttribute('src');
      player.video.load();
    }
  } catch { /* */ }
  player.root?.remove();

  // Best-effort release of the upstream tuner. We don't block close
  // on the network round-trip — keepalive lets the request finish
  // even if the page is being unloaded.
  try {
    await fetch(`/api/livetv/stop/${encodeURIComponent(player.sessionToken)}`, {
      method: 'POST',
      credentials: 'same-origin',
      keepalive: true,
    });
  } catch { /* ignore — server-side idle sweep is the safety net */ }
}
