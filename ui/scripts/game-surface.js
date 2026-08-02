/**
 * GameSurface — sandboxed iframe for playing pinned games from the Portal.
 *
 * Mounted inside the Library overlay as a sibling to #workspace. Only one of
 * the two is visible at a time. Supports embed-mode games (iframe with a
 * resolved ``embed_src`` URL) and local-mode games (zip bundles unpacked
 * from the artifact). js13k entries take the local-mode path.
 *
 * Key behaviours:
 *   - Loading state covers the first ~5s of latency with a quiet gradient.
 *   - Header auto-hides after 2.5s of pointer idleness and reappears on
 *     mouse movement within 16px of the top edge.
 *   - Fullscreen uses the native API on the iframe wrapper; Esc returns.
 *   - Close always restores focus to Library.
 *
 * Sandbox + allow attributes are set declaratively on the iframe element
 * in index.html so they can't drift out of sync with programmatic sets.
 */

import { escapeHtml, showToast } from './app.js';
import { ViewStack } from './view-stack.js';
import { composeBundle, installSaveBridge } from './bundle-composer.js';

const IDLE_MS = 2500;
const HOVER_ZONE_PX = 16;

// Touch-first devices get a tap-to-toggle header instead of the mouse-edge
// reveal pattern — mobile has no mouse, so the mousemove listener never
// fires and the header would stay permanently hidden.
const _isTouchPrimary = typeof window !== 'undefined'
  && typeof window.matchMedia === 'function'
  && window.matchMedia('(pointer: coarse)').matches;

// iOS Safari (including iPad in desktop mode) doesn't support the Fullscreen
// API on iframes. Detect up front and hide the button instead of showing a
// control that silently fails. The Mac-with-touch branch catches iPad
// pretending to be a Mac in "Request Desktop Website" mode.
const _isIOS = typeof navigator !== 'undefined' && (
  /iP(hone|od|ad)/.test(navigator.platform || '')
  || (navigator.userAgent?.includes('Mac') && 'ontouchend' in document)
);

let _initialized = false;
let _idleTimer = null;
let _onFullscreenChange = null;
let _onGamepadConnect = null;
let _onGamepadDisconnect = null;
let _onPopState = null;
let _historyEntryPushed = false;
let _closingFromPopState = false;
let _wakeLock = null;
let _onVisibilityChange = null;
let _fitMode = 'stretch';  // 'stretch' | 'fit' — session-scoped, not persisted

// Save state bridge — only wired for local-mode games (srcdoc iframes where
// we control the bundled HTML and can inject the storage shim). Embed-mode
// games run in their own origin and rely on browser storage partitioning
// for per-device persistence.
//
// Lives in ui/scripts/bundle-composer.js so the TV cast surface
// (ui/cast-app/) can use the same bridge. We keep the Library-side pill
// rendering local and pass an onStatus callback through.
let _saveBridge = {
  active: false,
  artifactId: null,
  handle: null,   // returned by installSaveBridge — has uninstall/flush/getInitialSave
};

const _el = {};
let _currentArtifact = null;

function _cache() {
  if (_initialized) return;
  _el.root = document.getElementById('game-surface');
  _el.header = document.getElementById('game-surface-header');
  _el.back = document.getElementById('game-surface-back');
  _el.title = document.getElementById('game-surface-title');
  _el.meta = document.getElementById('game-surface-meta');
  _el.source = document.getElementById('game-surface-source');
  _el.fullscreen = document.getElementById('game-surface-fullscreen');
  _el.iframeWrap = document.getElementById('game-surface-iframe-wrap');
  _el.iframe = document.getElementById('game-surface-iframe');
  _el.loading = document.getElementById('game-surface-loading');
  _el.loadingName = document.getElementById('game-surface-loading-name');
  _el.fallback = document.getElementById('game-surface-fallback');
  _el.fallbackBtn = document.getElementById('game-surface-fallback-btn');
  _el.fit = document.getElementById('game-surface-fit');

  // Promote to body so the three-pane Library can dispatch into us
  // without inheriting display:none from a hidden ancestor. Mirrors
  // workspace.js — same legacy DOM shape, same fix. Idempotent.
  if (_el.root && _el.root.parentElement !== document.body) {
    document.body.appendChild(_el.root);
  }

  _el.back?.addEventListener('click', closeGameSurface);
  _el.fullscreen?.addEventListener('click', _toggleFullscreen);
  _el.fit?.addEventListener('click', _toggleFit);
  _el.fallbackBtn?.addEventListener('click', () => {
    const url = _currentArtifact?.metadata?.source_url || _currentArtifact?.metadata?.embed_url;
    if (url) window.open(url, '_blank', 'noopener,noreferrer');
  });
  _el.iframe?.addEventListener('load', _onIframeLoad);

  _el.root?.addEventListener('mousemove', _onPointerActivity);
  _el.root?.addEventListener('keydown', _onKeyDown);

  // Tap-to-toggle for touch devices. Listens on the surface root so taps
  // anywhere outside the header reveal it; another tap hides it. We can't
  // listen on the iframe itself (cross-origin), but tapping the thin strip
  // around it is enough. Pointer-coarse captures both touch-only devices
  // and pens/styluses; the mouse-only path stays unchanged.
  if (_isTouchPrimary) {
    _el.iframeWrap?.addEventListener('touchend', _onTouchToggle, { passive: true });
  }

  // Hide fullscreen affordance on iOS — requestFullscreen on iframes is a
  // no-op there and having a dead button is worse than having none.
  if (_isIOS && _el.fullscreen) {
    _el.fullscreen.hidden = true;
  }

  _initialized = true;
}

function _resetIdleTimer() {
  _el.root?.removeAttribute('data-idle');
  if (_idleTimer) clearTimeout(_idleTimer);
  _idleTimer = setTimeout(() => {
    if (_el.root && !_el.root.classList.contains('hidden')) {
      _el.root.setAttribute('data-idle', '1');
    }
  }, IDLE_MS);
}

function _onPointerActivity(e) {
  // Only nudge header back if cursor is near the top, or inside the header
  // itself. Avoids flashing the chrome every time the user twitches the mouse
  // deep in the game.
  const rect = _el.root.getBoundingClientRect();
  const nearTop = (e.clientY - rect.top) <= (_el.header?.offsetHeight || 48) + HOVER_ZONE_PX;
  if (nearTop) _resetIdleTimer();
}

function _onTouchToggle() {
  // Toggle: if header is visible, schedule a hide; if hidden, show immediately.
  if (_el.root?.getAttribute('data-idle') === '1') {
    _resetIdleTimer();
  } else {
    if (_idleTimer) clearTimeout(_idleTimer);
    _el.root?.setAttribute('data-idle', '1');
  }
}

function _onKeyDown(e) {
  if (e.key === 'Escape' && !document.fullscreenElement) {
    // Esc-outside-fullscreen closes the surface. Esc-inside-fullscreen is
    // handled by the browser (exits fullscreen without bubbling here).
    closeGameSurface();
  } else if (e.key && e.key.toLowerCase() === 'f' && e.target === _el.root) {
    _toggleFullscreen();
  }
}

async function _toggleFullscreen() {
  if (_isIOS) {
    // Defensive — the button is hidden on iOS but if it somehow fires,
    // fail quietly rather than silently-nothing.
    showToast('Fullscreen isn\'t supported for games in Safari', 'warning');
    return;
  }
  if (document.fullscreenElement) {
    await document.exitFullscreen().catch(() => {});
    return;
  }
  const target = _el.iframeWrap || _el.root;
  try {
    await target.requestFullscreen();
    // Lock to landscape on touch devices once fullscreen is confirmed.
    // The Screen Orientation API requires an active fullscreen context.
    // iOS rejects the call (Apple doesn't permit orientation lock) —
    // the .catch swallows it quietly.
    if (_isTouchPrimary && screen.orientation?.lock) {
      screen.orientation.lock('landscape').catch(() => {});
    }
  } catch (err) {
    console.warn('Fullscreen failed:', err);
    showToast('Fullscreen unavailable for this game', 'warning');
  }
}

// ─── Aspect-ratio toggle ──────────────────────────────────────────────
// Two modes: "stretch" (default, iframe fills container — most games
// want this and handle their own canvas sizing) and "fit" (iframe
// constrained to 16:9 aspect, letter/pillar-boxed inside the container
// — rescues games designed for fixed 16:9 frames from distortion).
// State is session-scoped intentionally; persisting per-game adds a
// settings write on every toggle without meaningful benefit.

function _applyFit() {
  if (!_el.iframeWrap) return;
  _el.iframeWrap.setAttribute('data-fit', _fitMode);
  if (_el.fit) {
    _el.fit.setAttribute(
      'aria-label',
      _fitMode === 'stretch' ? 'Switch to fit-aspect' : 'Switch to stretch',
    );
    _el.fit.textContent = _fitMode === 'stretch' ? '\u2B1C' : '\u2B1B';
    // ⬜ (stretch active) vs ⬛ (fit active) — filled square signals "constrained".
  }
}

function _toggleFit() {
  _fitMode = _fitMode === 'stretch' ? 'fit' : 'stretch';
  _applyFit();
}

// ─── Save state bridge ───────────────────────────────────────────────
//
// For local-mode games only. The bundled HTML (via assemble.js) already
// ships a localStorage polyfill that postMessages `storage-init`,
// `storage-set`, `storage-remove`, `storage-clear` to the parent. We
// answer `storage-init` with the server-stored blob and debounce-PUT
// subsequent writes back to the server. Embed-mode games run in their
// own origin and can't talk to us this way; the browser's storage
// partitioning keeps saves Augmentum-scoped per-device, which is the
// best we can do without proxying remote game content (out of scope).

function _updateSavePill(status) {
  const pill = document.getElementById('game-surface-save');
  if (!pill) return;
  const cfg = {
    idle:    { hidden: true },
    local:   { text: '\u25CB Saves on this device', cls: 'is-local', hidden: false, title: 'Saves stored by the game in your browser (persists per-device)' },
    syncing: { text: '\u21BB Saving\u2026', cls: 'is-syncing', hidden: false, title: 'Syncing save to your account' },
    saved:   { text: '\u2713 Saved to account', cls: 'is-saved', hidden: false, title: 'Save state synced to your Augmentum account' },
    error:   { text: '\u26A0 Save failed', cls: 'is-error', hidden: false, title: 'Save sync failed; next change will retry' },
  }[status] || { hidden: true };
  pill.hidden = !!cfg.hidden;
  if (!cfg.hidden) {
    pill.textContent = cfg.text;
    pill.className = 'game-surface-save ' + (cfg.cls || '');
    pill.title = cfg.title || '';
  }
}

function _installSaveBridge(artifact) {
  const playMode = artifact?.metadata?.play_mode || 'embed';
  if (_saveBridge.handle) {
    _saveBridge.handle.uninstall();
    _saveBridge.handle = null;
  }
  _saveBridge.artifactId = artifact.id;

  if (playMode !== 'local') {
    _saveBridge.active = false;
    _updateSavePill('local');
    return;
  }

  _saveBridge.active = true;
  _saveBridge.handle = installSaveBridge({
    iframe: _el.iframe,
    artifactId: artifact.id,
    onStatus: _updateSavePill,
  });
}

function _uninstallSaveBridge() {
  if (_saveBridge.handle) {
    _saveBridge.handle.uninstall();
    _saveBridge.handle = null;
  }
  _saveBridge.active = false;
  _saveBridge.artifactId = null;
  _updateSavePill('idle');
}

// ─── Screen Wake Lock ────────────────────────────────────────────────
// Without this, mobile screens dim and lock mid-game. The API requires
// a user-gesture context — the Play click that opened the surface
// satisfies that requirement. Browsers release the lock when the tab
// is hidden, so we re-acquire on visibility return.

async function _acquireWakeLock() {
  if (!('wakeLock' in navigator)) return;
  try {
    _wakeLock = await navigator.wakeLock.request('screen');
    _wakeLock.addEventListener?.('release', () => { _wakeLock = null; });
  } catch (err) {
    // Common non-fatal reasons: battery saver, not a user gesture, lost
    // focus. Not worth toasting — the game still plays.
    console.debug?.('wake lock request failed', err);
  }
}

function _releaseWakeLock() {
  if (_wakeLock) {
    _wakeLock.release().catch(() => {});
    _wakeLock = null;
  }
}

// ─── Gamepad detection ────────────────────────────────────────────────
// Show a small pill in the header when a controller is connected. The
// game itself still reads input via the standard Gamepad API — we just
// surface the fact that pairing succeeded so the user has confirmation.

function _updateGamepadPill() {
  const pad = document.getElementById('game-surface-gamepad');
  if (!pad) return;
  const pads = (navigator.getGamepads?.() || []).filter(Boolean);
  if (pads.length === 0) {
    pad.hidden = true;
    return;
  }
  pad.hidden = false;
  pad.textContent = pads.length === 1
    ? `🎮 ${pads[0].id?.split('(')[0].trim() || 'Controller'}`
    : `🎮 ${pads.length} controllers`;
  // Make the pill a remap entry-point — clicking it opens the remap
  // modal. The pill is the natural anchor for "I have a controller,
  // I want to change its bindings".
  if (!pad.dataset.remapBound) {
    pad.dataset.remapBound = '1';
    pad.style.cursor = 'pointer';
    pad.title = 'Click to remap controller bindings';
    pad.addEventListener('click', () => _openRemapModal());
  }
}

// ── Controller remap modal ───────────────────────────────────────────
// Single-instance modal — opens from the gamepad pill or from the title
// runtime's "Configure controls" affordance. Lets the user pick a system
// and edit per-action bindings. Defaults come from
// /api/controllers/profiles + /api/controllers/{id}; the user-override
// CRUD goes through /api/controllers/{id}/remap.

async function _openRemapModal(preselectSystemId = '') {
  document.getElementById('controller-remap-modal')?.remove();

  const overlay = document.createElement('div');
  overlay.id = 'controller-remap-modal';
  overlay.className = 'controller-remap-overlay';
  overlay.innerHTML = `
    <div class="controller-remap-card" role="dialog" aria-labelledby="controller-remap-title">
      <header class="controller-remap-head">
        <h3 id="controller-remap-title">Controller bindings</h3>
        <button class="controller-remap-close" aria-label="Close">✕</button>
      </header>
      <div class="controller-remap-body">
        <div class="controller-remap-systems">
          <label for="controller-remap-system">System:</label>
          <select id="controller-remap-system"></select>
          <select id="controller-remap-pad-routing" title="How multiple controllers map to player slots">
            <option value="index">By index (player 1 = pad 1)</option>
            <option value="firstpress">First press claims player 1</option>
          </select>
          <button id="controller-remap-reset" class="btn btn-sm" title="Clear override; revert to defaults">Reset</button>
        </div>
        <div id="controller-remap-actions"
             style="margin-top:var(--space-sm); max-height:50vh; overflow-y:auto;
                    border:1px solid var(--border-subtle); border-radius:var(--radius-sm)"></div>
        <div id="controller-remap-status" class="field-hint" style="margin-top:var(--space-sm)"></div>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  const close = () => overlay.remove();
  overlay.querySelector('.controller-remap-close').addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  // Load systems
  let systems = [];
  try {
    const r = await fetch('/api/controllers/profiles', { credentials: 'same-origin' });
    if (r.ok) {
      const d = await r.json();
      systems = d.profiles || [];
    }
  } catch (_) { /* surfaced via empty select below */ }

  const sysSel = overlay.querySelector('#controller-remap-system');
  if (systems.length === 0) {
    sysSel.innerHTML = '<option>None available</option>';
  } else {
    sysSel.innerHTML = systems
      .map(s => `<option value="${escapeHtml(s.system_id || s.id)}">${escapeHtml(s.display_name || s.name || s.system_id)}</option>`)
      .join('');
  }
  if (preselectSystemId) sysSel.value = preselectSystemId;

  const renderForSystem = async (systemId) => {
    const actionsEl = overlay.querySelector('#controller-remap-actions');
    const padSel = overlay.querySelector('#controller-remap-pad-routing');
    const statusEl = overlay.querySelector('#controller-remap-status');
    actionsEl.innerHTML = '<div style="padding:var(--space-sm); color:var(--text-muted)">Loading…</div>';
    statusEl.textContent = '';

    // Fetch resolved layout (defaults + override merged) for display.
    let layout;
    try {
      const r = await fetch(`/api/controllers/${encodeURIComponent(systemId)}`, { credentials: 'same-origin' });
      if (!r.ok) {
        actionsEl.innerHTML = `<div style="padding:var(--space-sm)">Failed (status ${r.status})</div>`;
        return;
      }
      const d = await r.json();
      layout = d.layout || {};
    } catch (err) {
      actionsEl.innerHTML = `<div style="padding:var(--space-sm)">${escapeHtml(String(err.message || err))}</div>`;
      return;
    }

    // Fetch raw user override separately so we know what's customised.
    let remapBindings = {};
    let remapPadRouting = 'index';
    try {
      const r = await fetch(`/api/controllers/${encodeURIComponent(systemId)}/remap`, { credentials: 'same-origin' });
      if (r.ok) {
        const d = await r.json();
        remapBindings = d.remap?.bindings || d.bindings || {};
        remapPadRouting = d.remap?.pad_routing || d.pad_routing || 'index';
      }
    } catch (_) { /* no override = empty */ }
    padSel.value = remapPadRouting;

    const actions = layout.actions || [];
    if (actions.length === 0) {
      actionsEl.innerHTML = '<div style="padding:var(--space-sm); color:var(--text-muted)">No actions defined for this system.</div>';
      return;
    }
    actionsEl.innerHTML = actions.map(a => {
      const aid = a.id || a.action_id || '';
      const label = a.label || a.name || aid;
      const overridden = !!remapBindings[aid];
      const eff = a.effective || a.binding || {};
      const kb = (overridden ? remapBindings[aid].keyboard : eff.keyboard) || '';
      const gp = (overridden ? remapBindings[aid].gamepad_button : eff.gamepad_button) || '';
      return `
        <div class="controller-remap-row" data-action-id="${escapeHtml(aid)}"
             style="display:grid; grid-template-columns: 1fr 140px 140px auto; gap:var(--space-sm);
                    align-items:center; padding:var(--space-xs) var(--space-sm); border-bottom:1px solid var(--border-subtle)">
          <span><strong>${escapeHtml(label)}</strong>${overridden ? ' <em style="color:var(--accent)">(custom)</em>' : ''}</span>
          <input type="text" class="field-input" data-field="keyboard" value="${escapeHtml(String(kb))}" placeholder="Key (e.g. Space)" />
          <input type="text" class="field-input" data-field="gamepad_button" value="${escapeHtml(String(gp))}" placeholder="Button (e.g. A)" />
          <button class="btn btn-sm" data-action="clear">Default</button>
        </div>`;
    }).join('');

    // Inline save on input change (debounce per row).
    let saveTimer = null;
    const queueSave = () => {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => _saveCurrentRemap(systemId, overlay, statusEl), 600);
    };
    actionsEl.querySelectorAll('input').forEach(inp => {
      inp.addEventListener('input', queueSave);
    });
    padSel.onchange = queueSave;
    actionsEl.querySelectorAll('[data-action="clear"]').forEach(btn => {
      btn.addEventListener('click', () => {
        const row = btn.closest('.controller-remap-row');
        row.querySelectorAll('input').forEach(i => { i.value = ''; });
        queueSave();
      });
    });
  };

  sysSel.addEventListener('change', () => renderForSystem(sysSel.value));
  overlay.querySelector('#controller-remap-reset').addEventListener('click', async () => {
    const sid = sysSel.value;
    if (!sid || !confirm('Clear ALL custom bindings for this system?')) return;
    try {
      await fetch(`/api/controllers/${encodeURIComponent(sid)}/remap`, {
        method: 'DELETE', credentials: 'same-origin',
      });
      await renderForSystem(sid);
    } catch (err) {
      showToast?.('Reset failed', 'error');
    }
  });

  if (sysSel.value) await renderForSystem(sysSel.value);
}

async function _saveCurrentRemap(systemId, overlay, statusEl) {
  const padRouting = overlay.querySelector('#controller-remap-pad-routing')?.value || 'index';
  const bindings = {};
  overlay.querySelectorAll('.controller-remap-row').forEach(row => {
    const aid = row.dataset.actionId;
    if (!aid) return;
    const kb = row.querySelector('[data-field="keyboard"]')?.value?.trim() || '';
    const gp = row.querySelector('[data-field="gamepad_button"]')?.value?.trim() || '';
    // Only emit a binding row when at least one field is non-empty, so
    // empty inputs leave the row as "inherit from default" rather than
    // saving a null-binding.
    if (kb || gp) {
      bindings[aid] = { keyboard: kb || null, gamepad_button: gp || null };
    }
  });
  if (statusEl) statusEl.textContent = 'Saving…';
  try {
    const r = await fetch(`/api/controllers/${encodeURIComponent(systemId)}/remap`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ bindings, pad_routing: padRouting }),
    });
    if (statusEl) statusEl.textContent = r.ok ? 'Saved.' : `Save failed (status ${r.status})`;
  } catch (err) {
    if (statusEl) statusEl.textContent = `Save failed: ${err.message || err}`;
  }
}

// Expose globally so the title runtime / emulator launcher can open the
// modal pre-targeted to the system it's about to launch.
window.augmentumOpenControllerRemap = (systemId) => _openRemapModal(systemId || '');

// ─── History / Android back button ────────────────────────────────────
// Push a history entry on open so the system back gesture closes the
// game instead of leaving Augmentum. Track whether we pushed our own
// entry to avoid looping on programmatic closeGameSurface calls.

function _installHistoryTrap() {
  if (_onPopState) window.removeEventListener('popstate', _onPopState);
  _onPopState = (e) => {
    if (isGameSurfaceOpen()) {
      _closingFromPopState = true;
      closeGameSurface();
      _closingFromPopState = false;
    }
  };
  window.addEventListener('popstate', _onPopState);
  try {
    history.pushState({ _augmentumGameSurface: true }, '');
    _historyEntryPushed = true;
  } catch {
    _historyEntryPushed = false;
  }
}

function _uninstallHistoryTrap() {
  if (_onPopState) {
    window.removeEventListener('popstate', _onPopState);
    _onPopState = null;
  }
  if (_historyEntryPushed && !_closingFromPopState) {
    // Only consume the history entry if close wasn't triggered BY popstate
    // (in which case the entry is already gone).
    try { history.back(); } catch {}
  }
  _historyEntryPushed = false;
}

function _onIframeLoad() {
  // Hide the loading shade once the iframe reports a first paint.
  // Different-origin embeds won't expose document state due to sandbox,
  // so `load` is the only signal we get.
  _el.loading?.classList.add('hidden');
  // The iframe sandbox prevents allow-same-origin reads across origins,
  // so we can't detect X-Frame-Options blocks by inspecting
  // contentDocument. Fallback surfaces on explicit error only.
}

function _showFallback() {
  _el.loading?.classList.add('hidden');
  _el.fallback?.classList.remove('hidden');
}

function _clearFallback() {
  _el.fallback?.classList.add('hidden');
  _el.loading?.classList.remove('hidden');
}

// ─── Local bundle mount ──────────────────────────────────────────────
//
// Local-mode games ship their full HTML + asset bundle inside the
// artifact's source_json. We fetch it, pull out the entry point, and
// feed it straight into the iframe via `srcdoc` so it runs from our
// origin. No hotlink script, no frame-ancestors CSP, no external fetch
// at play time.
//
// Multi-file bundles (index.html + auxiliary binaries) are not yet
// supported — the iframe's base URL is ``about:srcdoc`` so relative
// fetches won't resolve. We flag those at pin time with
// ``bundle_single_file=false`` and show a fallback here rather than
// mount a half-working game.

async function _mountLocalBundle(artifact) {
  const pendingArtifact = artifact;
  const meta = artifact.metadata || {};
  const entry = meta.bundle_entry || 'index.html';

  // ``source_json`` is the bundle payload. It ships in the artifact fetch
  // response, but some list-endpoint responses strip it to keep the
  // payload small — fall back to a targeted refetch when missing.
  let sourceJson = artifact.source_json;
  if (!sourceJson && artifact.id) {
    try {
      const resp = await fetch(`/api/artifacts/${artifact.id}`);
      if (resp.ok) {
        const full = await resp.json();
        sourceJson = full.source_json;
        // Keep the in-memory artifact in sync so re-opens don't refetch.
        artifact.source_json = sourceJson;
      }
    } catch {
      // Fall through to the generic fallback below.
    }
  }

  let bundle;
  try {
    bundle = typeof sourceJson === 'string' ? JSON.parse(sourceJson) : sourceJson;
  } catch {
    bundle = null;
  }
  if (!bundle || !Array.isArray(bundle.files) || bundle.files.length === 0) {
    _showFallback();
    return;
  }

  const entryFile = bundle.files.find(f => f.path === entry)
    || bundle.files.find(f => (f.path || '').toLowerCase().endsWith('.html'));
  if (!entryFile) {
    _showFallback();
    return;
  }

  // Decode the entry HTML. Text-encoded files are stored verbatim;
  // base64 would be unusual for an HTML entry but we handle it for
  // completeness.
  let html = entryFile.content || '';
  if (entryFile.encoding === 'base64') {
    try {
      html = atob(html);
    } catch {
      _showFallback();
      return;
    }
  }

  // Pre-fetch the user's save state before composing the bundle so it
  // can be baked into the shim as a literal. Without this, inline game
  // scripts that read ``localStorage`` during head parsing would race
  // any handshake and see an empty store. The save-bridge handle (set
  // by ``_installSaveBridge`` above) exposes the in-flight fetch via
  // ``getInitialSave()`` so we reuse it instead of double-fetching.
  let initialSave = {};
  if (_saveBridge.handle && _saveBridge.artifactId === artifact.id) {
    try {
      initialSave = (await _saveBridge.handle.getInitialSave()) || {};
    } catch {
      initialSave = {};
    }
  }

  if (_currentArtifact !== pendingArtifact) return;

  // Compose the HTML with the bundle bridge so relative fetches resolve
  // against the sibling files and localStorage is pre-populated with
  // any persisted state. For genuine single-file bundles the bridge
  // is mostly just the storage shim; for multi-file games (two-thirds
  // of js13k) the fetch/XHR rewrite is also doing work.
  const composed = composeBundle(html, bundle.files, entryFile.path, initialSave);

  // ``srcdoc`` carries the full document and the iframe sandbox keeps
  // it isolated from our origin's DOM/cookies. We keep
  // ``allow-same-origin`` because without it Chrome marks
  // ``window.localStorage`` as a non-configurable own-property that
  // throws, which prevents our shim's ``Object.defineProperty`` from
  // shadowing it — games then lose all save state on every reload.
  // The tradeoff: a pathological game loop can stall the parent.
  // Mitigation is out of scope here; revisit if a dedicated worker-
  // based runtime or separate preview origin becomes available.
  _el.iframe.removeAttribute('src');
  _el.iframe.srcdoc = composed;
}

// ─── Bundle composer ─────────────────────────────────────────────────
// Lives in ui/scripts/bundle-composer.js — shared with the TV cast surface
// at ui/cast-app/. composeBundle inlines siblings, injects the fetch/XHR
// shim, and wires the localStorage save bridge. Both surfaces need an
// identical render path so save-compatible state survives device handoff.

/**
 * Open the game surface and mount the given artifact.
 *
 * @param {object} artifact Library artifact of type=game
 */
export function openGameSurface(artifact) {
  _cache();
  if (!_el.root || !_el.iframe) {
    showToast('Game player unavailable', 'error');
    return;
  }

  _currentArtifact = artifact;
  const meta = artifact.metadata || {};
  const playMode = meta.play_mode || 'embed';

  _el.title.textContent = artifact.display_name || 'Game';
  if (_el.loadingName) _el.loadingName.textContent = artifact.display_name || 'Loading…';

  const metaBits = [];
  if (meta.author) metaBits.push(meta.author);
  if (meta.source) metaBits.push(meta.source);
  _el.meta.textContent = metaBits.join(' · ');

  if (_el.source) {
    const sourceUrl = meta.source_url || meta.embed_url || '';
    if (sourceUrl) {
      _el.source.href = sourceUrl;
      _el.source.removeAttribute('hidden');
    } else {
      _el.source.setAttribute('hidden', 'true');
    }
  }

  _clearFallback();

  if (playMode === 'embed') {
    // Embed-mode games provide their iframe URL directly via
    // ``meta.embed_src`` resolved at pin time. js13k entries fall
    // through to the local-bundle path below; remote embeds (a future
    // self-hosted source) just need a populated ``embed_src``.
    if (meta.embed_src) {
      _el.iframe.src = meta.embed_src;
    } else {
      _showFallback();
    }
  } else if (playMode === 'local') {
    // Install the save bridge BEFORE the mount kicks off — _mountLocalBundle
    // reads _saveBridge.handle.getInitialSave() synchronously when the
    // artifact already carries source_json (the common case), so the handle
    // must be set on the same microtask. Embed-mode skips the bridge.
    _installSaveBridge(artifact);
    // Fetch the unpacked bundle from the artifact's source_json and mount
    // it as an iframe srcdoc. Game content served from our own origin
    // side-steps the frame-ancestors CSP that third-party hosts attach,
    // which is the whole point of the local-mode path.
    _mountLocalBundle(artifact).catch((err) => {
      console.warn('Local bundle mount failed:', err);
      _showFallback();
    });
  } else {
    _showFallback();
  }

  // Hide workspace if it was visible; only one surface at a time inside Library.
  const workspace = document.getElementById('workspace');
  if (workspace) workspace.hidden = true;

  _el.root.classList.remove('hidden');
  _el.root.focus?.();
  _resetIdleTimer();

  if (_onFullscreenChange) document.removeEventListener('fullscreenchange', _onFullscreenChange);
  _onFullscreenChange = () => {
    // On fullscreen exit, make sure the header comes back once then
    // re-enters the idle cycle.
    _resetIdleTimer();
  };
  document.addEventListener('fullscreenchange', _onFullscreenChange);

  // Gamepad: attach listeners + prime the pill in case a controller is
  // already connected when the surface opens.
  if (_onGamepadConnect) window.removeEventListener('gamepadconnected', _onGamepadConnect);
  if (_onGamepadDisconnect) window.removeEventListener('gamepaddisconnected', _onGamepadDisconnect);
  _onGamepadConnect = () => _updateGamepadPill();
  _onGamepadDisconnect = () => _updateGamepadPill();
  window.addEventListener('gamepadconnected', _onGamepadConnect);
  window.addEventListener('gamepaddisconnected', _onGamepadDisconnect);
  _updateGamepadPill();

  // Wake lock: keep the screen on while playing. Re-acquire when the
  // tab regains visibility since browsers release on hide.
  _acquireWakeLock();
  if (_onVisibilityChange) document.removeEventListener('visibilitychange', _onVisibilityChange);
  _onVisibilityChange = () => {
    if (document.visibilityState === 'visible' && isGameSurfaceOpen()) {
      _acquireWakeLock();
    }
  };
  document.addEventListener('visibilitychange', _onVisibilityChange);

  // Reset fit mode per-open (session-scoped within a single play).
  _applyFit();

  // Save state bridge for the non-local paths. Local-mode already had
  // _installSaveBridge called above (must precede _mountLocalBundle) — the
  // embed/fallback branches still need the pill set, so call it here for
  // them. Calling twice is safe: the second call uninstalls the first.
  if (playMode !== 'local') {
    _installSaveBridge(artifact);
  }

  // Android back button → close. Push a history entry now so the first
  // back gesture closes the surface rather than leaving the app entirely.
  _installHistoryTrap();

  // Track in ViewStack so a mode change or library close cleanly tears us
  // down instead of leaving the game iframe running over unrelated UI.
  ViewStack.pushOverlay('game', { onClose: closeGameSurface });
}

// Re-entry guard — closeGameSurface pops the stack, which calls onClose =
// closeGameSurface. This flag short-circuits that second invocation.
let _gameCloseViaStack = false;

export function closeGameSurface() {
  if (_gameCloseViaStack) return;
  if (!_el.root) return;

  if (document.fullscreenElement) {
    document.exitFullscreen().catch(() => {});
  }
  if (_idleTimer) { clearTimeout(_idleTimer); _idleTimer = null; }
  if (_onFullscreenChange) {
    document.removeEventListener('fullscreenchange', _onFullscreenChange);
    _onFullscreenChange = null;
  }
  if (_onGamepadConnect) {
    window.removeEventListener('gamepadconnected', _onGamepadConnect);
    _onGamepadConnect = null;
  }
  if (_onGamepadDisconnect) {
    window.removeEventListener('gamepaddisconnected', _onGamepadDisconnect);
    _onGamepadDisconnect = null;
  }
  if (_onVisibilityChange) {
    document.removeEventListener('visibilitychange', _onVisibilityChange);
    _onVisibilityChange = null;
  }
  _releaseWakeLock();
  // Release any landscape lock we took — otherwise the whole page
  // stays locked to landscape after the game closes.
  try { screen.orientation?.unlock?.(); } catch {}
  _uninstallSaveBridge();
  _uninstallHistoryTrap();

  // Blank the iframe so any background tabs / audio stop immediately.
  // Clearing srcdoc matters for local-mode games — setting only .src
  // doesn't unload a srcdoc document, so the game keeps running.
  if (_el.iframe) {
    _el.iframe.removeAttribute('srcdoc');
    _el.iframe.src = 'about:blank';
  }
  _clearFallback();
  _el.root.classList.add('hidden');
  _el.root.removeAttribute('data-idle');
  _currentArtifact = null;

  // Return focus to the Library overlay so screen readers / keyboard
  // navigation land somewhere predictable. The new three-pane Library
  // owns its own search input inside the sidebar; focusing the overlay
  // root is enough — Tab from there reaches Search first.
  document.getElementById('library-shell-overlay')?.focus?.();

  // Sync ViewStack — pop after teardown so onClose re-entry hits the
  // _gameCloseViaStack guard and short-circuits.
  if (ViewStack.hasOverlay('game')) {
    _gameCloseViaStack = true;
    try { ViewStack.popOverlay('game'); }
    finally { _gameCloseViaStack = false; }
  }
}

// For debugging / imperative checks from the console.
export function isGameSurfaceOpen() {
  return !!(_el.root && !_el.root.classList.contains('hidden'));
}

// Suppress "unused" hint from escape/escapeHtml — kept for future use when we
// render richer metadata strings in the header.
void escapeHtml;
