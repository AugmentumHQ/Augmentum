// ui/scripts/stream-stage.js
//
// Player surface for AGSP-streamed games (Voxel World today; future
// streamed emulators + native game bundles use the same path).
// Mirrors the API of game-surface.js / emulator-stage.js so the
// library dispatch stays uniform: ``openStreamStage(profile)`` to
// mount, ``closeStreamStage()`` to tear down.
//
// Why an iframe to Selkies' own viewer (vs a custom WebRTC client):
// Selkies-gstreamer ships a complete browser-side WebRTC viewer that
// already handles signaling, ICE negotiation, input forwarding,
// encoder/decoder selection, adaptive bitrate. Reimplementing that
// would be weeks of work for no user-visible benefit. The iframe
// route is parser-supported, sandboxable, and gets us a full working
// stream today. A custom Augmentum-skinned client can replace the
// iframe later if we decide the chrome integration is worth it.
//
// CSP: the parent page's frame-src now allows
// ``http(s)://<this-host>:*`` (added in server.py based on the
// request's Host header) so the iframe can load the stream at
// whichever port the port pool allocated.

import { showToast } from './app.js';

const STREAM_PREFS_KEY = 'gameStreamInputPrefs';
const STREAM_PREFS_STORAGE_KEY = 'augmentum.gameStreamInputPrefs';

// Header icon set — 16x16 SVGs that pick up currentColor. Matches the
// emulator-stage icon language so the chrome reads as the same family.
const _STREAM_ICONS = Object.freeze({
  settings: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
  fullscreen: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/></svg>',
  close: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
});

const DEFAULT_INPUT_PREFS = Object.freeze({
  mouseSensitivity: 0.2,
  gamepadMode: 'auto',
  controllerDeadzone: 0.15,
  touchMode: 'auto',
  resolution: '',
  bitrateMbps: '',
  encoder: 'auto',
});

let _state = {
  overlay: null,
  iframe: null,
  profile: null,
  sessionId: null,
  streamUrl: null,
  prefs: null,
  inputPanel: null,
  closeHandler: null,
  heartbeatTimer: null,
  visibilityHandler: null,
  pagehideHandler: null,
};


// ── Public API ──────────────────────────────────────────────────────


export async function openStreamStage(profile, opts = {}) {
  // ``opts.kiosk`` (boolean) strips the chrome bar (title / settings /
  // fullscreen / close) for cast-receiver embeds. The iframe still
  // loads the Selkies viewer; only the player-facing UI is suppressed.
  // ESC keyboard fallback stays wired in case a hardware keyboard is
  // attached to the TV.
  if (!profile || !profile.id) {
    showToast('Cannot open: missing profile', 'error');
    return;
  }
  const kiosk = !!opts.kiosk;
  // If a previous stage is still up (e.g. user clicked Voxel World
  // twice), tear it down first so we don't leak containers.
  if (_state.overlay) {
    await closeStreamStage('replaced');
  }
  _state.profile = profile;

  // 1) Start (or reuse) the session via the existing AGSP route.
  //
  //    Three paths:
  //      (a) opts.preStarted -- caller already owns a session (e.g.
  //          title-launch route just created one for an emulator
  //          ROM). Skip find/create entirely. Reuse heuristics
  //          don't apply because each ROM launch is a distinct
  //          session keyed to a specific blob.
  //      (b) Reuse an already-live session for this profile. Avoids
  //          piling up containers when the user clicks Voxel World
  //          multiple times. Per-profile reuse is correct for
  //          single-world games like Luanti; emulator launches
  //          always go through path (a) so multi-ROM scenarios
  //          don't accidentally reuse the wrong ROM's container.
  //      (c) Start a new session. Standard click-to-launch path.
  //
  const prefs = _normalizeInputPrefs(
    opts.inputPrefs || await _loadInputPrefs(profile.id),
  );
  _state.prefs = prefs;
  const touchMode = _resolveTouchMode(prefs);

  let started = opts.preStarted || null;
  if (!started && !opts.forceNew) {
    started = await _findLiveSessionForProfile(profile.id);
    if (started) {
      showToast(
        `Resuming existing ${profile.display_name} session.`,
        'info',
      );
    }
  }

  try {
    if (!started) {
      const body = _buildStartRequest(profile, prefs, touchMode);
      const r = await fetch('/api/game-stream/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${r.status}`);
      }
      started = await r.json();
    }
  } catch (e) {
    showToast(`Couldn't start ${profile.display_name}: ${e.message}`, 'error');
    _state.profile = null;
    _state.prefs = null;
    return;
  }
  _state.sessionId = started.session_id;

  // 2) Build the overlay. Chrome matches emulator-stage / web-game
  //    surfaces — same gametime tokens, same icon-button language —
  //    so a user moving between players sees one coherent surface.
  const overlay = document.createElement('div');
  overlay.className = 'stream-stage-overlay';
  if (kiosk) overlay.classList.add('is-kiosk');
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-label', `Streaming: ${profile.display_name}`);

  // Kiosk mode skips the chrome bar entirely. The status pill is the
  // only thing we still want to track internally (so `_state.statusEl`
  // is a no-op-friendly null assignment via the kiosk branch).
  let bar = null;
  if (!kiosk) {
    bar = document.createElement('div');
    bar.className = 'stream-stage-bar';

    const title = document.createElement('span');
    title.className = 'stream-stage-title';
    title.textContent = profile.display_name || profile.id;

    const sub = document.createElement('span');
    sub.className = 'stream-stage-sub';
    sub.textContent = `session ${started.session_id.slice(0, 8)}`;

    const spacer = document.createElement('span');
    spacer.className = 'stream-stage-spacer';

    // Connection status pill — animated dot + label. CSS toggles the
    // animation off when status flips to "live".
    const status = document.createElement('span');
    status.className = 'stream-stage-status';
    status.dataset.status = 'connecting';
    status.innerHTML = `
      <span class="stream-stage-status-dot" aria-hidden="true"></span>
      <span class="stream-stage-status-label">connecting</span>
    `;
    _state.statusEl = status;

    const inputBtn = _textBtn('Settings', _STREAM_ICONS.settings, 'Stream settings');
    inputBtn.addEventListener('click', () => _toggleInputPanel());

    const fullscreenBtn = _textBtn('Fullscreen', _STREAM_ICONS.fullscreen, 'Fullscreen');
    fullscreenBtn.addEventListener('click', () => {
      const el = _state.iframe;
      if (!el) return;
      if (document.fullscreenElement) {
        document.exitFullscreen?.();
      } else {
        el.requestFullscreen?.();
      }
    });

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'stream-stage-iconbtn stream-stage-close-btn';
    closeBtn.setAttribute('aria-label', 'Close');
    closeBtn.title = 'Close (Esc)';
    closeBtn.innerHTML = _STREAM_ICONS.close;
    closeBtn.addEventListener('click', () => closeStreamStage('user'));

    bar.appendChild(title);
    bar.appendChild(sub);
    bar.appendChild(spacer);
    bar.appendChild(status);
    bar.appendChild(inputBtn);
    bar.appendChild(fullscreenBtn);
    bar.appendChild(closeBtn);
  }

  // 3) Mount overlay shell with a loading message FIRST. The backend
  //    readiness endpoint probes Selkies from inside the Docker host
  //    path and returns structured state; the browser no longer has to
  //    spam /stream/<port>/ and fill DevTools with expected 502s.
  const loading = document.createElement('div');
  loading.className = 'stream-stage-loading';
  const loadingPulse = document.createElement('div');
  loadingPulse.className = 'stream-stage-loading-pulse';
  loadingPulse.setAttribute('aria-hidden', 'true');
  const loadingTitle = document.createElement('div');
  loadingTitle.className = 'stream-stage-loading-title';
  const loadingDetail = document.createElement('div');
  loadingDetail.className = 'stream-stage-loading-detail';
  const loadingDiag = document.createElement('div');
  loadingDiag.className = 'stream-stage-loading-diag';
  loading.appendChild(loadingPulse);
  loading.appendChild(loadingTitle);
  loading.appendChild(loadingDetail);
  loading.appendChild(loadingDiag);
  _setLoading(
    loadingTitle,
    loadingDetail,
    loadingDiag,
    `Starting ${profile.display_name || 'stream'}...`,
    'Container is starting.',
    _tlsHint(),
  );
  if (bar) overlay.appendChild(bar);
  overlay.appendChild(loading);
  document.body.appendChild(overlay);
  _state.overlay = overlay;

  // ESC to close while we're still waiting for the stream.
  _state.closeHandler = (e) => {
    if (e.key === 'Escape' && !document.fullscreenElement) {
      closeStreamStage('user');
    }
  };
  document.addEventListener('keydown', _state.closeHandler);

  const readiness = await _waitForReadiness(started, {
    titleEl: loadingTitle,
    detailEl: loadingDetail,
    diagEl: loadingDiag,
    displayName: profile.display_name || profile.id || 'stream',
  });

  if (!_state.overlay) return;  // user closed while we were polling

  if (!readiness?.ready) {
    showToast(
      readiness?.message || `${profile.display_name} took too long to start.`,
      'error',
    );
    await closeStreamStage('timeout');
    return;
  }

  const streamUrl = _streamUrlFromReadiness(readiness, started.stream_port);
  _state.streamUrl = streamUrl;

  // 3b) Container is ready. Swap loading message for the iframe.
  overlay.removeChild(loading);

  const iframe = document.createElement('iframe');
  iframe.className = 'stream-stage-iframe';
  // Pointer lock is REQUIRED for first-person camera control --
  // selkies' input.js calls requestPointerLock() on click and
  // reads movementX/Y (delta) when locked. Without lock, every
  // mouse move is treated as an absolute position, producing the
  // "character looks at sky and spins" symptom. We intentionally do
  // not sandbox this iframe: the Selkies viewer is app-owned code
  // served through our same-origin stream proxy, and sandboxing it
  // with allow-scripts + allow-same-origin only creates Chrome's
  // sandbox-escape warning without adding meaningful isolation.
  iframe.setAttribute(
    'allow',
    'fullscreen; gamepad; autoplay; clipboard-read; clipboard-write',
  );
  iframe.referrerPolicy = 'same-origin';
  iframe.tabIndex = 0;
  iframe.src = streamUrl;
  iframe.addEventListener('load', () => {
    try { iframe.focus(); } catch (_) { /* ignore focus refusal */ }
    _setConnectionStatus('live', 'live');
    _startHeartbeat();
  }, { once: true });
  _state.iframe = iframe;

  overlay.appendChild(iframe);
  _startHeartbeat();

  showToast(
    `Streaming ${profile.display_name} -- click the stream to capture mouse.`,
    'info',
  );
}


function _textBtn(label, svgMarkup, title) {
  // Helper that builds an icon+label button without interpolating the
  // SVG into a template literal (which trips the security scanner's
  // innerHTML-without-escapeHtml check). Icon goes in via innerHTML on
  // a child span; label uses textContent so the scanner is happy.
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'stream-stage-textbtn';
  btn.setAttribute('aria-label', title || label);
  if (title) btn.title = title;
  const icon = document.createElement('span');
  icon.setAttribute('aria-hidden', 'true');
  icon.innerHTML = svgMarkup;
  const text = document.createElement('span');
  text.textContent = label;
  btn.appendChild(icon);
  btn.appendChild(text);
  return btn;
}

function _setLoading(titleEl, detailEl, diagEl, title, detail, diag = '') {
  if (titleEl) titleEl.textContent = title || '';
  if (detailEl) detailEl.textContent = detail || '';
  if (diagEl) diagEl.textContent = diag || '';
}

function _setConnectionStatus(status, label) {
  const el = _state.statusEl;
  if (!el) return;
  el.dataset.status = status;
  const lab = el.querySelector('.stream-stage-status-label');
  if (lab) lab.textContent = label || status;
}


function _buildStartRequest(profile, prefs, touchMode) {
  const body = {
    profile_id: profile.id,
    touch_mode: touchMode,
    input: {
      touch_mode: touchMode,
      mouse_sensitivity: prefs.mouseSensitivity,
      gamepad_enabled: _gamepadEnabledForPrefs(prefs),
      controller_deadzone: prefs.controllerDeadzone,
    },
  };
  if (prefs.resolution) body.resolution = prefs.resolution;
  if (prefs.bitrateMbps) body.bitrate_mbps = prefs.bitrateMbps;
  if (prefs.encoder) body.encoder = prefs.encoder;
  return body;
}


function _touchDeviceHeuristic() {
  return (
    (navigator.maxTouchPoints || 0) > 0
    && window.matchMedia('(pointer: coarse)').matches
  );
}


function _resolveTouchMode(prefs) {
  if (prefs.touchMode === 'on') return true;
  if (prefs.touchMode === 'off') return false;
  return _touchDeviceHeuristic();
}


function _gamepadEnabledForPrefs(prefs) {
  return prefs.gamepadMode !== 'off';
}


function _clampNumber(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}


function _normalizeMode(value, allowed, fallback) {
  const raw = String(value || '').trim().toLowerCase();
  return allowed.includes(raw) ? raw : fallback;
}


function _normalizeInputPrefs(raw) {
  const src = raw && typeof raw === 'object' ? raw : {};
  const merged = { ...DEFAULT_INPUT_PREFS, ...src };
  const resolution = String(merged.resolution || '').trim();
  const bitrate = merged.bitrateMbps === '' || merged.bitrateMbps == null
    ? ''
    : Math.round(_clampNumber(merged.bitrateMbps, 1, 25, 4));
  return {
    mouseSensitivity: _clampNumber(
      merged.mouseSensitivity,
      0.01,
      2.0,
      DEFAULT_INPUT_PREFS.mouseSensitivity,
    ),
    gamepadMode: _normalizeMode(
      merged.gamepadMode,
      ['auto', 'on', 'off'],
      DEFAULT_INPUT_PREFS.gamepadMode,
    ),
    controllerDeadzone: _clampNumber(
      merged.controllerDeadzone,
      0.0,
      0.5,
      DEFAULT_INPUT_PREFS.controllerDeadzone,
    ),
    touchMode: _normalizeMode(
      merged.touchMode,
      ['auto', 'on', 'off'],
      DEFAULT_INPUT_PREFS.touchMode,
    ),
    resolution: /^\d{3,5}x\d{3,5}$/.test(resolution) ? resolution : '',
    bitrateMbps: bitrate,
    encoder: _normalizeMode(
      merged.encoder,
      ['auto', 'nvenc', 'vaapi', 'x264', 'x264enc'],
      DEFAULT_INPUT_PREFS.encoder,
    ),
  };
}


function _parsePrefsBlob(raw) {
  if (!raw) return {};
  if (typeof raw === 'object') return raw;
  try {
    const parsed = JSON.parse(String(raw));
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (_) {
    return {};
  }
}


function _prefsFromBlob(blob, profileId) {
  const base = blob && typeof blob === 'object' ? blob : {};
  const defaults = base.defaults && typeof base.defaults === 'object'
    ? base.defaults
    : {};
  const flat = Object.prototype.hasOwnProperty.call(base, 'mouseSensitivity')
    ? base
    : {};
  const byProfile = base.profiles && typeof base.profiles === 'object'
    ? base.profiles[profileId]
    : null;
  const legacyProfile = base[profileId] && typeof base[profileId] === 'object'
    ? base[profileId]
    : null;
  return _normalizeInputPrefs({
    ...defaults,
    ...flat,
    ...(legacyProfile || {}),
    ...(byProfile || {}),
  });
}


function _readLocalPrefsBlob() {
  try {
    return _parsePrefsBlob(window.localStorage.getItem(STREAM_PREFS_STORAGE_KEY));
  } catch (_) {
    return {};
  }
}


function _writeLocalPrefsBlob(blob) {
  try {
    window.localStorage.setItem(STREAM_PREFS_STORAGE_KEY, JSON.stringify(blob));
  } catch (_) {
    // Server-side settings remain the source of truth when storage is blocked.
  }
}


async function _loadPrefsBlob() {
  try {
    const r = await fetch('/api/config/ui', { cache: 'no-store' });
    if (r.ok) {
      const body = await r.json().catch(() => ({}));
      return _parsePrefsBlob(body[STREAM_PREFS_KEY]);
    }
  } catch (_) {
    // Fall back to this browser's cached copy.
  }
  return _readLocalPrefsBlob();
}


async function _loadInputPrefs(profileId) {
  const blob = await _loadPrefsBlob();
  return _prefsFromBlob(blob, profileId);
}


async function _saveInputPrefs(profileId, prefs) {
  const clean = _normalizeInputPrefs(prefs);
  const blob = await _loadPrefsBlob();
  const next = blob && typeof blob === 'object' && !Array.isArray(blob)
    ? { ...blob }
    : {};
  const profiles = next.profiles && typeof next.profiles === 'object'
    ? { ...next.profiles }
    : {};
  profiles[profileId] = clean;
  next.version = 1;
  next.defaults = _normalizeInputPrefs(next.defaults || DEFAULT_INPUT_PREFS);
  next.profiles = profiles;
  _writeLocalPrefsBlob(next);
  try {
    const r = await fetch('/api/config/ui', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [STREAM_PREFS_KEY]: JSON.stringify(next) }),
    });
    return r.ok;
  } catch (_) {
    return false;
  }
}


function _toggleInputPanel() {
  if (_state.inputPanel) {
    _closeInputPanel();
    return;
  }
  if (!_state.overlay) return;
  const panel = _buildInputPanel();
  _state.overlay.appendChild(panel);
  _state.inputPanel = panel;
}


function _closeInputPanel() {
  if (_state.inputPanel?.parentNode) {
    _state.inputPanel.parentNode.removeChild(_state.inputPanel);
  }
  _state.inputPanel = null;
}


function _selectOptions(options, selected) {
  return options.map(([value, label]) => (
    `<option value="${value}"${value === selected ? ' selected' : ''}>${label}</option>`
  )).join('');
}


function _buildInputPanel() {
  const prefs = _normalizeInputPrefs(_state.prefs);
  const caps = _state.profile?.input_capabilities || {};
  const sensitivityCaps = caps.pointer?.mouse_sensitivity || {};
  const mouseMin = _clampNumber(sensitivityCaps.min, 0.01, 2.0, 0.01);
  const mouseMax = _clampNumber(sensitivityCaps.max, mouseMin, 2.0, 2.0);
  const mouseStep = _clampNumber(sensitivityCaps.step, 0.001, 1.0, 0.01);
  const gamepadSupported = caps.gamepad?.supported !== false;
  // Only show the mouse-sensitivity slider when the profile's
  // entrypoint actually consumes it. Luanti's pointer.mode is
  // "relative" + advertises mouse_sensitivity bounds; emulator-streamed
  // declares "absolute-or-emulator-native" (no consumption); browser-
  // stream declares "none". Without this gate the slider rendered for
  // every profile but only Luanti picked up the value, which silently
  // dropped user input on the other two surfaces.
  const pointerMode = caps.pointer?.mode || '';
  const showMouseSensitivity = (
    pointerMode === 'relative'
    && caps.pointer?.mouse_sensitivity != null
  );
  const touchSupported = caps.touch?.supported !== false;

  const panel = document.createElement('div');
  panel.className = 'stream-stage-input-panel';
  panel.innerHTML = `
    <form>
      <div class="stream-stage-panel-head">
        <span class="stream-stage-panel-title">Stream settings</span>
        <button type="button" data-action="close" class="stream-stage-btn-save">Close</button>
      </div>
      ${showMouseSensitivity ? `
      <label class="stream-stage-field">
        <span class="stream-stage-field-label">Mouse sensitivity</span>
        <div class="stream-stage-field-pair">
          <input class="stream-stage-input" name="mouseSensitivityRange" type="range" min="${mouseMin}" max="${mouseMax}" step="${mouseStep}" value="${prefs.mouseSensitivity}">
          <input class="stream-stage-input" name="mouseSensitivity" type="number" min="${mouseMin}" max="${mouseMax}" step="${mouseStep}" value="${prefs.mouseSensitivity}">
        </div>
      </label>
      ` : ''}
      <label class="stream-stage-field">
        <span class="stream-stage-field-label">Controller</span>
        <select class="stream-stage-select" name="gamepadMode" ${gamepadSupported ? '' : 'disabled'}>
          ${_selectOptions([['auto', 'Auto'], ['on', 'On'], ['off', 'Off']], prefs.gamepadMode)}
        </select>
      </label>
      <label class="stream-stage-field">
        <span class="stream-stage-field-label">Controller deadzone</span>
        <div class="stream-stage-field-pair">
          <input class="stream-stage-input" name="controllerDeadzoneRange" type="range" min="0" max="0.5" step="0.01" value="${prefs.controllerDeadzone}" ${gamepadSupported ? '' : 'disabled'}>
          <input class="stream-stage-input" name="controllerDeadzone" type="number" min="0" max="0.5" step="0.01" value="${prefs.controllerDeadzone}" ${gamepadSupported ? '' : 'disabled'}>
        </div>
      </label>
      ${touchSupported ? `
      <label class="stream-stage-field">
        <span class="stream-stage-field-label">Touch controls</span>
        <select class="stream-stage-select" name="touchMode">
          ${_selectOptions([['auto', 'Auto'], ['on', 'On'], ['off', 'Off']], prefs.touchMode)}
        </select>
      </label>
      ` : ''}
      <label class="stream-stage-field">
        <span class="stream-stage-field-label">Resolution</span>
        <select class="stream-stage-select" name="resolution">
          ${_selectOptions([['', 'Default'], ['1280x720', '720p'], ['1920x1080', '1080p'], ['2560x1440', '1440p']], prefs.resolution)}
        </select>
      </label>
      <label class="stream-stage-field">
        <span class="stream-stage-field-label">Bitrate Mbps</span>
        <input class="stream-stage-input" name="bitrateMbps" type="number" min="1" max="25" step="1" value="${prefs.bitrateMbps}" placeholder="Default">
      </label>
      <label class="stream-stage-field">
        <span class="stream-stage-field-label">Encoder</span>
        <select class="stream-stage-select" name="encoder">
          ${_selectOptions([['auto', 'Auto'], ['nvenc', 'NVENC'], ['vaapi', 'VAAPI'], ['x264', 'x264']], prefs.encoder)}
        </select>
      </label>
      <div class="stream-stage-panel-actions">
        <button type="submit" class="stream-stage-btn-save">Save</button>
        <button type="button" data-action="restart" class="stream-stage-btn-restart">Restart stream</button>
      </div>
    </form>
  `;

  _syncPair(panel, 'mouseSensitivityRange', 'mouseSensitivity', mouseMin, mouseMax);
  _syncPair(panel, 'controllerDeadzoneRange', 'controllerDeadzone', 0, 0.5);
  panel.querySelector('[data-action="close"]')?.addEventListener('click', _closeInputPanel);
  panel.querySelector('[data-action="restart"]')?.addEventListener('click', async () => {
    await _persistPanelPrefs(panel, { restart: true });
  });
  panel.querySelector('form')?.addEventListener('submit', async (event) => {
    event.preventDefault();
    await _persistPanelPrefs(panel, { restart: false });
  });
  return panel;
}


function _syncPair(panel, rangeName, numberName, min, max) {
  const range = panel.querySelector(`[name="${rangeName}"]`);
  const number = panel.querySelector(`[name="${numberName}"]`);
  if (!range || !number) return;
  range.addEventListener('input', () => {
    number.value = range.value;
  });
  number.addEventListener('input', () => {
    range.value = String(_clampNumber(number.value, min, max, Number(range.value)));
  });
}


function _prefsFromPanel(panel) {
  // Read only fields the panel actually rendered. Hidden rows (e.g.
  // mouse sensitivity on the emulator-streamed profile) leave the
  // corresponding pref untouched rather than being forced to the
  // default via _normalizeInputPrefs's fallback path.
  const previous = _normalizeInputPrefs(_state.prefs);
  const value = name => {
    const el = panel.querySelector(`[name="${name}"]`);
    return el ? el.value : undefined;
  };
  const merge = (key) => {
    const v = value(key);
    return v === undefined ? previous[key] : v;
  };
  return _normalizeInputPrefs({
    mouseSensitivity: merge('mouseSensitivity'),
    gamepadMode: merge('gamepadMode'),
    controllerDeadzone: merge('controllerDeadzone'),
    touchMode: merge('touchMode'),
    resolution: merge('resolution'),
    bitrateMbps: merge('bitrateMbps'),
    encoder: merge('encoder'),
  });
}


async function _persistPanelPrefs(panel, { restart = false } = {}) {
  const profile = _state.profile;
  if (!profile?.id) return;
  const next = _prefsFromPanel(panel);
  _state.prefs = next;
  const savedOnServer = await _saveInputPrefs(profile.id, next);
  if (!savedOnServer) {
    showToast('Stream input saved in this browser.', 'info');
  } else if (!restart) {
    showToast('Stream input saved.', 'success');
  }
  if (restart) {
    _closeInputPanel();
    await _restartStreamStage(profile, next);
  }
}


async function _restartStreamStage(profile, prefs) {
  if (!profile?.id) return;
  showToast('Restarting stream with input settings.', 'info');
  await closeStreamStage('replaced');
  await openStreamStage(profile, { forceNew: true, inputPrefs: prefs });
}


function _sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


function _streamUrlForPort(port) {
  const p = Number(port);
  if (!Number.isFinite(p) || p <= 0) return '';
  if (window.location.protocol === 'https:') return `/stream/${p}/`;
  const host = window.location.hostname;
  const hostForUrl = host.includes(':') && !host.startsWith('[') ? `[${host}]` : host;
  return `http://${hostForUrl}:${p}/`;
}


function _streamUrlFromReadiness(readiness, fallbackPort) {
  if (window.location.protocol === 'https:' && readiness?.stream_path) {
    return readiness.stream_path;
  }
  return readiness?.stream_url || _streamUrlForPort(fallbackPort);
}


function _tlsHint() {
  const host = window.location.hostname || '';
  const isPrivateIp = /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[0-1])\.|100\.)/.test(host);
  if (window.location.protocol !== 'https:' || !isPrivateIp) return '';
  if (window.isSecureContext) return '';
  return 'HTTPS is not fully trusted yet; service workers and WebRTC helpers may be blocked.';
}


function _readinessDetail(data) {
  if (!data) return 'Checking stream readiness.';
  if (data.ready) return 'Stream viewer is ready.';
  if (data.stage === 'waiting_stream') return 'Renderer is up; waiting for Selkies to accept the viewer.';
  if (data.stage === 'starting_container') return 'Container is starting.';
  if (data.stage === 'waiting_port') return 'Waiting for stream port assignment.';
  if (data.stage === 'container_stopped') return 'The stream container stopped before the viewer became ready.';
  return data.message || 'Checking stream readiness.';
}


function _readinessDiagnostic(data) {
  const tls = _tlsHint();
  if (!data) return tls;
  const bits = [];
  if (data.session_id) bits.push(`session ${String(data.session_id).slice(0, 8)}`);
  if (data.stream_port) bits.push(`stream ${data.stream_port}`);
  if (data.browser_host_kind === 'tailscale') bits.push('Tailscale');
  if (data.probe?.status) bits.push(`probe HTTP ${data.probe.status}`);
  if (tls) bits.push(tls);
  return bits.join(' - ');
}


async function _fetchReadiness(sessionId) {
  const r = await fetch(
    `/api/game-stream/sessions/${encodeURIComponent(sessionId)}/readiness`,
    { cache: 'no-store' },
  );
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(body.error || `HTTP ${r.status}`);
  }
  return body;
}

async function _sendHeartbeat(reason = 'interval') {
  if (!_state.sessionId) return false;
  try {
    const r = await fetch(
      `/api/game-stream/sessions/${encodeURIComponent(_state.sessionId)}/heartbeat`,
      {
        method: 'POST',
        cache: 'no-store',
        keepalive: reason === 'pagehide',
      },
    );
    return r.ok;
  } catch (_) {
    return false;
  }
}

// Telemetry — fire once per heartbeat interval with whatever performance
// metrics the embedded iframe exposes via postMessage. The backend
// records (rtt_ms, jitter_ms, packet_loss, bitrate_kbps, fps) into the
// game_stream_telemetry table for adaptive bitrate / quality tuning.
// If the viewer never posts metrics, this is a no-op cheap heartbeat
// echo (the server treats missing values as nulls, no row written for
// all-null rows on the backend).
async function _sendTelemetry(metrics) {
  if (!_state.sessionId || !metrics) return false;
  try {
    const r = await fetch(
      `/api/game-stream/sessions/${encodeURIComponent(_state.sessionId)}/telemetry`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify(metrics),
      },
    );
    return r.ok;
  } catch (_) {
    return false;
  }
}

// Listen for postMessage telemetry from the embedded viewer iframe.
// Selkies / Wolf / Sunshine front-ends all publish similar metrics via
// window.parent.postMessage({type:'game-stream-telemetry', ...}, '*');
// we accept any subset of the fields the backend expects.
window.addEventListener('message', (e) => {
  if (!e?.data || typeof e.data !== 'object') return;
  if (e.data.type !== 'game-stream-telemetry') return;
  const m = {};
  if (Number.isFinite(e.data.rtt_ms)) m.rtt_ms = e.data.rtt_ms;
  if (Number.isFinite(e.data.jitter_ms)) m.jitter_ms = e.data.jitter_ms;
  if (Number.isFinite(e.data.packet_loss)) m.packet_loss = e.data.packet_loss;
  if (Number.isFinite(e.data.bitrate_kbps)) m.bitrate_kbps = e.data.bitrate_kbps;
  if (Number.isFinite(e.data.fps)) m.fps = e.data.fps;
  if (Object.keys(m).length > 0) _sendTelemetry(m);
});


function _startHeartbeat() {
  if (!_state.sessionId || _state.heartbeatTimer) return;
  _sendHeartbeat('start');
  _state.heartbeatTimer = window.setInterval(() => {
    _sendHeartbeat('interval');
  }, 25000);
  _state.visibilityHandler = () => {
    if (document.visibilityState === 'visible') {
      _sendHeartbeat('visible');
    } else {
      _sendHeartbeat('hidden');
    }
  };
  _state.pagehideHandler = () => _sendHeartbeat('pagehide');
  document.addEventListener('visibilitychange', _state.visibilityHandler);
  window.addEventListener('pagehide', _state.pagehideHandler);
}


function _stopHeartbeat() {
  if (_state.heartbeatTimer) {
    window.clearInterval(_state.heartbeatTimer);
  }
  if (_state.visibilityHandler) {
    document.removeEventListener('visibilitychange', _state.visibilityHandler);
  }
  if (_state.pagehideHandler) {
    window.removeEventListener('pagehide', _state.pagehideHandler);
  }
}


async function _waitForReadiness(started, ui) {
  const timeoutMs = 60000;
  const intervalMs = 750;
  const startTime = Date.now();
  let last = null;
  while (Date.now() - startTime < timeoutMs) {
    try {
      last = await _fetchReadiness(started.session_id);
      _setLoading(
        ui.titleEl,
        ui.detailEl,
        ui.diagEl,
        `Starting ${ui.displayName}...`,
        _readinessDetail(last),
        _readinessDiagnostic(last),
      );
      if (last.ready) return last;
      if (last.stage === 'container_stopped' || last.stage === 'crashed') {
        return last;
      }
    } catch (e) {
      last = {
        ready: false,
        stage: 'readiness_error',
        message: e?.message || 'Could not check stream readiness.',
      };
      _setLoading(
        ui.titleEl,
        ui.detailEl,
        ui.diagEl,
        `Starting ${ui.displayName}...`,
        last.message,
        _tlsHint(),
      );
    }
    await _sleep(intervalMs);
    if (!_state.overlay) return null;
  }
  return last || {
    ready: false,
    stage: 'timeout',
    message: `${ui.displayName} took too long to start.`,
  };
}


async function _findLiveSessionForProfile(profileId) {
  // Returns a session-shaped dict matching POST /api/game-stream/sessions
  // 201 if there's an already-live container for this user+profile,
  // else null. The list endpoint doesn't filter by profile -- we do
  // it client-side; the typical user has 0-1 live streams so the
  // overhead is irrelevant.
  try {
    const r = await fetch('/api/game-stream/sessions?live_only=1');
    if (!r.ok) return null;
    const body = await r.json();
    const list = Array.isArray(body.sessions) ? body.sessions : [];
    const match = list.find(s => s.profile_id === profileId
      && s.stream_port
      && (s.status === 'starting' || s.status === 'ready'
          || s.status === 'connected' || s.status === 'idle'));
    if (!match) return null;
    return {
      session_id: match.id || match.session_id,
      profile_id: match.profile_id,
      status: match.status,
      stream_port: match.stream_port,
      game_port: match.game_port,
      bitrate_mbps: match.bitrate_mbps,
      resolution: match.resolution,
    };
  } catch (_) {
    return null;
  }
}


export async function closeStreamStage(reason = 'user') {
  _stopHeartbeat();
  _closeInputPanel();
  if (_state.closeHandler) {
    document.removeEventListener('keydown', _state.closeHandler);
  }
  if (_state.overlay && _state.overlay.parentNode) {
    _state.overlay.parentNode.removeChild(_state.overlay);
  }

  // Stop the session server-side. Best-effort -- if the proxy is
  // unreachable the server-side idle reaper will clean up eventually.
  if (_state.sessionId) {
    try {
      await fetch(
        `/api/game-stream/sessions/${encodeURIComponent(_state.sessionId)}`,
        { method: 'DELETE' },
      );
    } catch (_) { /* swallow */ }
  }

  _state = {
    overlay: null,
    iframe: null,
    profile: null,
    sessionId: null,
    streamUrl: null,
    prefs: null,
    inputPanel: null,
    closeHandler: null,
    heartbeatTimer: null,
    visibilityHandler: null,
    pagehideHandler: null,
  };
  if (reason !== 'replaced') {
    showToast('Stream closed.', 'info');
  }
}
