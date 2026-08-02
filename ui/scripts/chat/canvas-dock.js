/* ==========================================================================
   Chat Module — Session Canvas dock
   A resizable right-docked panel holding one artifact beside chat that stays
   anchored while you keep chatting. The binding (which artifact a session
   shows) is persisted server-side via /api/canvas/{session_id}
   (canvas_routes.py); the dock width + open/closed preference are device-
   local UI chrome (localStorage).

   This is the FOUNDATION surface: later rich-content modes (live React
   preview, interactive charts, prose editing) render INTO this dock rather
   than spawning their own. "Edit" delegates to the existing Studio modal.

   Wiring: fed by `augmentum:session-changed` (swap which artifact shows +
   toggle the toolbar button) and `augmentum:turn-stats` (a turn finished —
   re-check the bound artifact for a new version and reload in place).
   ========================================================================== */
import { app, showToast } from '../app.js';

const WIDTH_KEY = 'augmentum-canvas-width';
const OPEN_KEY = 'augmentum-canvas-open';
const MIN_W = 320;
const DEFAULT_W = 440;
const SLOW_AFTER_MS = 12000;

// Per-format iframe sandbox — mirrors the inline artifact-card preview
// (chat/index.js::_toggleArtifactPreview): PDFs render in the browser's
// native viewer and need no scripts; HTML artifacts need scripts to be
// interactive. Neither grants allow-same-origin (safe opaque origin).
const PDF_SANDBOX = '';
const HTML_SANDBOX = 'allow-scripts allow-forms allow-modals allow-popups';
// When the preview is served from the isolated origin (:6444) the bundle
// gets a real, foreign-to-Augmentum origin where ES modules + localStorage
// work; allow-same-origin is then safe (no cookie crossover). The
// same-origin HTML_SANDBOX above is a null origin where both break.
const HTML_SANDBOX_ISOLATED = HTML_SANDBOX + ' allow-same-origin allow-popups';
function _isPdf(fmt) { return (fmt || '').toLowerCase().includes('pdf'); }


// Resolve the best origin to load an artifact preview from. Mirrors
// library/detail-pane.js::_resolveArtifactPreviewSrc — mints a one-time
// content-isolation token when the server has isolation enabled, else
// falls back to the same-origin null-origin sandbox (501).
async function _resolveArtifactPreviewSrc(artifactId, cacheBust) {
  const sameOrigin = {
    src: `/api/artifacts/${encodeURIComponent(artifactId)}/preview?v=${cacheBust}`,
    sandbox: HTML_SANDBOX,
  };
  try {
    const resp = await fetch('/api/content/preview-token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ kind: 'artifact_app', id: artifactId }),
    });
    if (!resp.ok) return sameOrigin;
    const data = await resp.json().catch(() => ({}));
    if (!data.token || !data.isolated_origin) return sameOrigin;
    return {
      src: `${data.isolated_origin}/api/artifacts/${encodeURIComponent(artifactId)}`
        + `/preview?_pvt=${encodeURIComponent(data.token)}&v=${cacheBust}`,
      sandbox: HTML_SANDBOX_ISOLATED,
    };
  } catch {
    return sameOrigin;
  }
}

let _sessionId = '';
let _summary = null;       // last canvas payload, or null when the session has no artifact
let _open = false;
let _refreshTimer = null;
let _loadTimer = null;
let _changedTimer = null;
let _els = null;
let _initialized = false;
let _versions = [];        // newest-first [{id, version_index, label}] for the stepper
let _viewIndex = 0;        // 0 = latest/live preview; >0 = browsing an older snapshot

function _readWidth() {
  const v = parseInt(localStorage.getItem(WIDTH_KEY) || '', 10);
  return Number.isFinite(v) && v >= MIN_W ? v : DEFAULT_W;
}

function _applyWidth(px) {
  const w = Math.max(MIN_W, Math.min(px, Math.round(window.innerWidth * 0.96)));
  document.documentElement.style.setProperty('--canvas-width', `${w}px`);
  return w;
}

function _setOpen(open) {
  _open = open;
  if (!_els?.dock) return;
  _els.dock.classList.toggle('hidden', !open);
  document.body.classList.toggle('canvas-open', open);
  try { localStorage.setItem(OPEN_KEY, open ? '1' : '0'); } catch { /* quota / disabled */ }
  if (_els.toggle) _els.toggle.setAttribute('aria-pressed', open ? 'true' : 'false');
}

async function _fetchBinding(sessionId) {
  if (!sessionId) return null;
  try {
    const resp = await fetch(`/api/canvas/${encodeURIComponent(sessionId)}`);
    if (!resp.ok) return null;
    const data = await resp.json();
    return data && data.artifact_id ? data : null;
  } catch {
    return null;  // canvas is optional decoration — never break chat
  }
}

function _showLoading(on) {
  if (!_els?.loading) return;
  _els.loading.classList.toggle('hidden', !on);
  if (_els.slow) _els.slow.classList.add('hidden');
  if (_loadTimer) { clearTimeout(_loadTimer); _loadTimer = null; }
  if (on) {
    // Lead with the spinner; only after a delay surface the new-tab escape
    // hatch (progress over abort-banner).
    _loadTimer = setTimeout(() => {
      if (_els.slow) _els.slow.classList.remove('hidden');
    }, SLOW_AFTER_MS);
  }
}

function _showComposer(on) {
  // The "Ask for a change" composer only applies to editable app bundles.
  if (_els?.composer) _els.composer.classList.toggle('hidden', !on);
}

function _renderEmpty() {
  if (!_els) return;
  _els.title.textContent = 'Canvas';
  _els.title.removeAttribute('title');
  _versions = [];
  _viewIndex = 0;
  _els.dock.classList.remove('viewing-history');
  if (_els.versions) _els.versions.classList.add('hidden');
  if (_els.restore) _els.restore.classList.add('hidden');
  _hideChanged();
  _els.iframe.classList.add('hidden');
  _els.iframe.removeAttribute('src');
  _showLoading(false);
  _showComposer(false);
  _els.empty.classList.remove('hidden');
}

function _reloadPreview() {
  if (!_els || !_summary) return;
  // Cache-bust by version so a turn's edit replaces a stale frame.
  const v = _summary.version_count || 0;
  // PDFs render in the native viewer (no scripts, no isolation needed).
  if (_isPdf(_summary.format)) {
    const url = `${_summary.preview_url}?v=${v}`;
    _els.iframe.setAttribute('sandbox', PDF_SANDBOX);
    if (_els.openTab) _els.openTab.href = url;
    if (_els.slowLink) _els.slowLink.href = url;
    _els.empty.classList.add('hidden');
    _showLoading(true);
    _els.iframe.src = url;
    return;
  }
  // HTML/app bundles: resolve the best origin (isolated when available)
  // async, then set sandbox+src once. Guard against the dock moving to
  // another artifact (or version) while the mint is in flight.
  const artifactId = _summary.artifact_id;
  _els.empty.classList.add('hidden');
  _showLoading(true);
  _resolveArtifactPreviewSrc(artifactId, v).then(({ src, sandbox }) => {
    if (!_els || _summary?.artifact_id !== artifactId || _viewIndex !== 0) return;
    _els.iframe.setAttribute('sandbox', sandbox);
    if (_els.openTab) _els.openTab.href = src;
    if (_els.slowLink) _els.slowLink.href = src;
    _els.iframe.src = src;
  });
}

function _renderArtifact(summary) {
  if (!_els) return;
  _summary = summary;
  _versions = Array.isArray(summary.versions) ? summary.versions : [];
  _viewIndex = 0;  // a (re)render always lands on the latest/live version
  // textContent is XSS-safe — no escaping needed for the title.
  _els.title.textContent = summary.display_name || 'Artifact';
  _els.title.title = summary.display_name || '';
  _els.dock.classList.remove('viewing-history');
  _renderVersions();
  _els.empty.classList.add('hidden');
  _els.iframe.classList.remove('hidden');
  _showComposer(!!summary.editable);
  _reloadPreview();
}

/** Paint the version stepper from `_versions` + `_viewIndex`. Hidden until
 *  an artifact has >1 version; shows a Restore button while browsing an
 *  older snapshot (and hides the live-only edit composer there). */
function _renderVersions() {
  if (!_els?.versions) return;
  const n = _versions.length;
  const show = n > 1;
  _els.versions.classList.toggle('hidden', !show);
  if (!show) {
    if (_els.restore) _els.restore.classList.add('hidden');
    _els.dock.classList.remove('viewing-history');
    return;
  }
  const cur = _versions[_viewIndex] || _versions[0];
  const total = _versions[0]?.version_index || n;
  if (_els.vlabel) {
    _els.vlabel.textContent = cur?.version_index != null
      ? `v${cur.version_index}/${total}`
      : `${n - _viewIndex}/${n}`;
  }
  // prev = older (higher array index); next = newer (lower array index).
  if (_els.verPrev) _els.verPrev.disabled = _viewIndex >= n - 1;
  if (_els.verNext) _els.verNext.disabled = _viewIndex <= 0;
  const browsing = _viewIndex > 0;
  if (_els.restore) _els.restore.classList.toggle('hidden', !browsing);
  _els.dock.classList.toggle('viewing-history', browsing);
  // The edit composer always targets the LIVE version — hide it while
  // browsing history so the action there is unambiguously "restore".
  _showComposer(!browsing && !!_summary?.editable);
}

/** Step the stepper. delta +1 = older, -1 = newer. */
function _stepVersion(delta) {
  const n = _versions.length;
  if (n <= 1) return;
  const next = Math.max(0, Math.min(n - 1, _viewIndex + delta));
  if (next === _viewIndex) return;
  _viewIndex = next;
  _renderVersions();
  _loadVersionView();
}

/** Load whichever version `_viewIndex` points at — the live preview at 0,
 *  else a read-only assembled snapshot (no mutation). */
function _loadVersionView() {
  if (!_els || !_summary) return;
  if (_viewIndex <= 0) { _reloadPreview(); return; }
  const ver = _versions[_viewIndex];
  if (!ver?.id) { _reloadPreview(); return; }
  const url = `/api/canvas/${encodeURIComponent(_sessionId)}/version/`
    + `${encodeURIComponent(ver.id)}/preview`;
  // Snapshots are always assembled app HTML — interactive sandbox.
  _els.iframe.setAttribute('sandbox', HTML_SANDBOX);
  if (_els.openTab) _els.openTab.href = url;
  if (_els.slowLink) _els.slowLink.href = url;
  _els.empty.classList.add('hidden');
  _showLoading(true);
  _els.iframe.src = url;
}

/** Bring the currently-browsed older version back as the live one. Reuses
 *  the existing artifact revert (which snapshots current first, so this is
 *  itself reversible), then re-resolves to land on the new latest. */
async function _restoreVersion() {
  const ver = _versions[_viewIndex];
  if (!ver?.id || !_summary?.artifact_id || _viewIndex <= 0) return;
  try {
    const resp = await fetch(
      `/api/artifacts/${encodeURIComponent(_summary.artifact_id)}`
      + `/revert/${encodeURIComponent(ver.id)}`,
      { method: 'POST' },
    );
    if (!resp.ok) { showToast('Couldn\'t restore that version', 'error'); return; }
    const label = ver.version_index != null ? `Restored v${ver.version_index}` : 'Version restored';
    _summary = null;            // force a full re-render at the new latest
    await _refresh(_sessionId);
    showToast(label, 'success');
  } catch {
    showToast('Couldn\'t reach the server', 'error');
  }
}

function _hideChanged() {
  if (_changedTimer) { clearTimeout(_changedTimer); _changedTimer = null; }
  if (_els?.changed) _els.changed.classList.add('hidden');
}

/** Transient "what changed" confirmation after an edit. Ownership tone,
 *  auto-dismisses. */
function _showChanged(summary) {
  if (!_els?.changed) return;
  const files = Array.isArray(summary.changed_files) ? summary.changed_files : [];
  let text;
  if (files.length === 1) {
    const f = files[0];
    const c = f.count || 0;
    text = `Updated ${f.path}${c ? ` · ${c} change${c === 1 ? '' : 's'}` : ''}`;
  } else if (files.length > 1) {
    const total = files.reduce((s, f) => s + (f.count || 0), 0);
    text = `Updated ${files.length} files${total ? ` · ${total} changes` : ''}`;
  } else {
    const n = summary.patches_applied || 0;
    text = n ? `Applied ${n} change${n === 1 ? '' : 's'}` : 'Change applied';
  }
  _els.changed.replaceChildren();
  const dot = document.createElement('span');
  dot.className = 'chip-dot';
  const span = document.createElement('span');
  span.className = 'chip-text';
  span.textContent = text;  // XSS-safe — file paths are untrusted text
  _els.changed.append(dot, span);
  _els.changed.classList.remove('hidden');
  if (_changedTimer) clearTimeout(_changedTimer);
  _changedTimer = setTimeout(() => {
    if (_els?.changed) _els.changed.classList.add('hidden');
  }, 5000);
}

/** Resolve which artifact this session shows + toggle the toolbar button
 *  (hidden when the session has no artifact). */
async function _refresh(sessionId) {
  _sessionId = sessionId || '';
  const data = await _fetchBinding(_sessionId);
  const has = !!data;
  if (_els?.toggle) _els.toggle.classList.toggle('hidden', !has);

  if (!has) {
    _summary = null;
    if (_open) _setOpen(false);
    _renderEmpty();
    return;
  }
  // Only rebuild the iframe when the artifact or its version actually changed.
  const changed = !_summary
    || _summary.artifact_id !== data.artifact_id
    || _summary.version_count !== data.version_count;
  if (changed) _renderArtifact(data);
  else _summary = data;
}

function _onTurnStats() {
  // A turn produced metrics — if the canvas is open, the bound artifact may
  // have a new version. Debounced single GET; cheap.
  if (!_open || !_sessionId) return;
  if (_refreshTimer) clearTimeout(_refreshTimer);
  _refreshTimer = setTimeout(() => { _refresh(_sessionId); }, 900);
}

function _setComposerBusy(on) {
  if (!_els?.composer) return;
  _els.composer.classList.toggle('busy', on);
  if (_els.composerInput) _els.composerInput.disabled = on;
  if (_els.composerSend) _els.composerSend.disabled = on;
}

function _autoGrow(el) {
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
}

/** Apply a natural-language change to the bound app in place. Reuses the
 *  server-side quick-edit machinery (POST /api/canvas/{id}/edit), which
 *  snapshots a version and rewrites the bundle; we then reload the iframe. */
async function _submitEdit() {
  if (!_sessionId || !_summary?.editable) return;
  const input = _els?.composerInput;
  const description = (input?.value || '').trim();
  if (!description) return;
  _setComposerBusy(true);
  try {
    const resp = await fetch(`/api/canvas/${encodeURIComponent(_sessionId)}/edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description }),
    });
    if (!resp.ok) {
      let msg = 'Couldn\'t apply that change';
      try { const e = await resp.json(); if (e?.detail) msg = e.detail; } catch { /* keep default */ }
      showToast(msg, 'error');
      return;
    }
    const data = await resp.json();
    if (input) { input.value = ''; _autoGrow(input); }
    // Force a reload even if version_count math matches — render the
    // returned summary (carries the bumped version + cache-bust).
    _summary = null;
    _renderArtifact(data);
    _showChanged(data);  // quiet "Updated index.html · 2 changes" chip
  } catch {
    showToast('Couldn\'t reach the server', 'error');
  } finally {
    _setComposerBusy(false);
  }
}

function _wireComposer() {
  const form = _els?.composer;
  const input = _els?.composerInput;
  if (!form) return;
  form.addEventListener('submit', (e) => { e.preventDefault(); _submitEdit(); });
  if (input) {
    input.addEventListener('input', () => _autoGrow(input));
    // Enter sends; Shift+Enter inserts a newline.
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        _submitEdit();
      }
    });
  }
}

function _wireResize(handle) {
  let startX = 0;
  let startW = 0;
  let dragging = false;
  const onMove = (e) => {
    if (!dragging) return;
    _applyWidth(startW + (startX - e.clientX));  // drag left → wider
  };
  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    document.body.style.userSelect = '';
    const px = parseInt(
      getComputedStyle(document.documentElement).getPropertyValue('--canvas-width'), 10);
    if (Number.isFinite(px)) {
      try { localStorage.setItem(WIDTH_KEY, String(px)); } catch { /* quota */ }
    }
  };
  handle.addEventListener('mousedown', (e) => {
    dragging = true;
    startX = e.clientX;
    startW = parseInt(
      getComputedStyle(document.documentElement).getPropertyValue('--canvas-width'), 10,
    ) || _readWidth();
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    e.preventDefault();
  });
}

/** Wire the dock once. Idempotent — safe to call from boot. */
export function initCanvasDock() {
  if (_initialized) return;
  const dock = document.getElementById('canvas-dock');
  if (!dock) return;
  _initialized = true;

  _els = {
    dock,
    toggle: document.getElementById('canvas-toggle-btn'),
    title: dock.querySelector('.canvas-dock-title'),
    versions: dock.querySelector('.canvas-dock-versions'),
    vlabel: dock.querySelector('.canvas-dock-vlabel'),
    verPrev: dock.querySelector('[data-canvas-action="ver-prev"]'),
    verNext: dock.querySelector('[data-canvas-action="ver-next"]'),
    restore: dock.querySelector('.canvas-dock-restore'),
    changed: dock.querySelector('.canvas-dock-changed'),
    iframe: dock.querySelector('.canvas-dock-iframe'),
    empty: dock.querySelector('.canvas-dock-empty'),
    loading: dock.querySelector('.canvas-dock-loading'),
    slow: dock.querySelector('.canvas-dock-slow'),
    slowLink: dock.querySelector('.canvas-dock-slow-link'),
    openTab: dock.querySelector('.canvas-dock-open-tab'),
    editBtn: dock.querySelector('[data-canvas-action="edit"]'),
    closeBtn: dock.querySelector('[data-canvas-action="close"]'),
    resize: dock.querySelector('.canvas-dock-resize'),
    composer: dock.querySelector('.canvas-dock-composer'),
    composerInput: dock.querySelector('.canvas-dock-composer-input'),
    composerSend: dock.querySelector('.canvas-dock-composer-send'),
  };

  // Clear the loading overlay once the preview paints (load fires for
  // same-origin frames regardless of sandbox, incl. server error pages).
  _els.iframe.addEventListener('load', () => {
    if (_els.iframe.getAttribute('src')) _showLoading(false);
  });

  _applyWidth(_readWidth());

  if (_els.toggle) {
    _els.toggle.addEventListener('click', () => {
      if (!_summary) _refresh(_sessionId);
      _setOpen(!_open);
    });
  }
  if (_els.closeBtn) _els.closeBtn.addEventListener('click', () => _setOpen(false));
  if (_els.editBtn) {
    _els.editBtn.addEventListener('click', () => {
      if (!_summary?.artifact_id) return;
      import('../studio.js')
        .then((m) => m.openStudio(_summary.artifact_id))
        .catch(() => showToast('Couldn\'t open the editor', 'error'));
    });
  }
  if (_els.resize) _wireResize(_els.resize);
  _wireComposer();
  if (_els.verPrev) _els.verPrev.addEventListener('click', () => _stepVersion(1));
  if (_els.verNext) _els.verNext.addEventListener('click', () => _stepVersion(-1));
  if (_els.restore) _els.restore.addEventListener('click', () => _restoreVersion());

  // Artifact cards call this global (onclick) to pin THIS artifact to the
  // session canvas and open it. Pinning persists server-side so the choice
  // survives refresh/restart (PUT /api/canvas/{session_id}).
  window._openArtifactCanvas = async function (btn) {
    const id = btn?.getAttribute?.('data-canvas-id');
    if (!id || !_sessionId) return;
    try {
      const resp = await fetch(`/api/canvas/${encodeURIComponent(_sessionId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ artifact_id: id }),
      });
      if (!resp.ok) { showToast('Couldn\'t open in canvas', 'error'); return; }
      const data = await resp.json();
      if (_els?.toggle) _els.toggle.classList.remove('hidden');
      _renderArtifact(data);
      _setOpen(true);
    } catch {
      showToast('Couldn\'t open in canvas', 'error');
    }
  };

  document.addEventListener('augmentum:session-changed', (e) => {
    _refresh(e.detail?.sessionId || '');
  });
  document.addEventListener('augmentum:turn-stats', _onTurnStats);

  // Restore the open preference, then resolve the current session. Only honor
  // "open" if the session actually has an artifact to show.
  try { _open = localStorage.getItem(OPEN_KEY) === '1'; } catch { _open = false; }
  const initialId = app?.state?.currentSessionId || '';
  _refresh(initialId).then(() => {
    _setOpen(Boolean(_open && _summary));
  });
}

// Self-init when loaded as a deferred module <script> (see index.html). The
// _initialized guard makes an explicit initCanvasDock() call elsewhere a no-op.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCanvasDock, { once: true });
} else {
  initCanvasDock();
}
