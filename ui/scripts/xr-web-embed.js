const XR_EMBED_PATH = '/ui/';

const SURFACE_EMBED_ROUTES = Object.freeze({
  chat: { mode: 'passthrough' },
  analytical: { mode: 'analytical' },
  analyze: { mode: 'analytical' },
  agentic: { mode: 'agentic' },
  build: { mode: 'agentic' },
  narrative: { mode: 'narrative' },
  story: { mode: 'narrative' },
  coder: { mode: 'coder' },
  browse: { surface: 'browse' },
  files: { surface: 'files' },
  notes: { surface: 'notes' },
  studio: { surface: 'studio' },
  media: { surface: 'media' },
  devices: { surface: 'devices' },
  games: { surface: 'games' },
});

let rootEl = null;
let windowEl = null;
let frameEl = null;
let titleEl = null;
let statusEl = null;
let minimizeEl = null;
let followEl = null;
let popoutEl = null;
let closeEl = null;
let resizeEl = null;

const state = {
  active: false,
  action: '',
  label: '',
  href: '',
  presenting: false,
  domOverlayType: '',
  immersiveSuppressed: false,
  minimized: false,
  followPanel: true,
  anchorVisible: false,
  manualRect: null,
  capabilities: {},
};

let pointerDrag = null;

function _surfaceAction(surface) {
  return String(surface?.action || surface?.id || surface || '').trim();
}

function _surfaceLabel(surface, action) {
  return String(surface?.label || action || 'Surface').trim();
}

function _sameOriginPath(url) {
  if (!url) return '';
  try {
    const parsed = new URL(url, window.location.origin);
    if (parsed.origin !== window.location.origin) return '';
    if (!parsed.pathname.startsWith(XR_EMBED_PATH)) return '';
    return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  } catch {
    return '';
  }
}

export function buildXrEmbedUrl(surface, opts = {}) {
  const action = _surfaceAction(surface);
  const route = SURFACE_EMBED_ROUTES[action] || SURFACE_EMBED_ROUTES[opts.surface] || {};
  const explicit = _sameOriginPath(surface?.embedUrl || opts.embedUrl || '');
  const url = new URL(explicit || XR_EMBED_PATH, window.location.origin);
  url.searchParams.set('xrEmbed', '1');

  const mode = opts.mode || route.mode || '';
  const routeSurface = opts.surface || route.surface || (!route.mode ? action : '');
  if (mode) url.searchParams.set('mode', mode);
  if (routeSurface) url.searchParams.set('surface', routeSurface);
  if (action) url.searchParams.set('xrSurface', action);
  if (opts.source) url.searchParams.set('source', opts.source);
  if (opts.sessionId) url.searchParams.set('xrSessionId', opts.sessionId);
  if (opts.primaryAction) url.searchParams.set('panelAction', opts.primaryAction);

  return `${url.pathname}${url.search}${url.hash}`;
}

function _setStatus(text) {
  if (statusEl) statusEl.textContent = text || '';
}

function _setVisible(visible) {
  if (!rootEl) return;
  rootEl.classList.toggle('is-open', visible);
  rootEl.classList.toggle('is-hidden', !visible);
  rootEl.setAttribute('aria-hidden', visible ? 'false' : 'true');
  state.active = visible;
}

function _icon(svg) {
  return `<svg viewBox="0 0 24 24" aria-hidden="true">${svg}</svg>`;
}

function _clampRect(rect) {
  const vw = Math.max(320, window.innerWidth || 1280);
  const vh = Math.max(240, window.innerHeight || 720);
  const width = Math.max(320, Math.min(vw, Number(rect.width || 0) || vw * 0.78));
  const height = Math.max(220, Math.min(vh, Number(rect.height || 0) || vh * 0.72));
  const left = Math.max(8, Math.min(vw - width - 8, Number(rect.left || 0)));
  const top = Math.max(8, Math.min(vh - height - 8, Number(rect.top || 0)));
  return { left, top, width, height };
}

function _applyWindowRect(rect, { anchored = false } = {}) {
  if (!windowEl) return;
  const next = _clampRect(rect);
  windowEl.style.setProperty('--xr-window-left', `${Math.round(next.left)}px`);
  windowEl.style.setProperty('--xr-window-top', `${Math.round(next.top)}px`);
  windowEl.style.setProperty('--xr-window-width', `${Math.round(next.width)}px`);
  windowEl.style.setProperty('--xr-window-height', `${Math.round(next.height)}px`);
  rootEl.classList.toggle('is-anchored', anchored);
}

function _defaultRect() {
  const vw = Math.max(320, window.innerWidth || 1280);
  const vh = Math.max(240, window.innerHeight || 720);
  const width = Math.min(vw - 24, Math.max(720, vw * 0.82));
  const height = Math.min(vh - 24, Math.max(460, vh * 0.78));
  return {
    left: (vw - width) / 2,
    top: (vh - height) / 2,
    width,
    height,
  };
}

function _refreshShellState() {
  if (!rootEl || !windowEl) return;
  rootEl.classList.toggle('is-minimized', state.minimized);
  rootEl.classList.toggle('is-following', state.followPanel);
  rootEl.classList.toggle('has-anchor', state.anchorVisible);
  rootEl.classList.toggle('is-immersive-suppressed', state.immersiveSuppressed);
  if (followEl) {
    followEl.classList.toggle('is-active', state.followPanel);
    followEl.title = state.followPanel ? 'Detach from spatial panel' : 'Follow spatial panel';
    followEl.setAttribute('aria-label', followEl.title);
  }
  if (minimizeEl) {
    minimizeEl.title = state.minimized ? 'Restore embedded page' : 'Minimize embedded page';
    minimizeEl.setAttribute('aria-label', minimizeEl.title);
  }
}

function _beginPointerDrag(event, mode) {
  if (!windowEl || event.button !== 0) return;
  if (event.target.closest('.xr-web-embed-btn')) return;
  const rect = windowEl.getBoundingClientRect();
  pointerDrag = {
    mode,
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    rect: {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    },
  };
  state.followPanel = false;
  state.manualRect = pointerDrag.rect;
  _refreshShellState();
  try { event.currentTarget.setPointerCapture?.(event.pointerId); } catch {}
  event.preventDefault();
}

function _updatePointerDrag(event) {
  if (!pointerDrag || pointerDrag.pointerId !== event.pointerId) return;
  const dx = event.clientX - pointerDrag.startX;
  const dy = event.clientY - pointerDrag.startY;
  const rect = { ...pointerDrag.rect };
  if (pointerDrag.mode === 'resize') {
    rect.width += dx;
    rect.height += dy;
  } else {
    rect.left += dx;
    rect.top += dy;
  }
  state.manualRect = _clampRect(rect);
  _applyWindowRect(state.manualRect);
  event.preventDefault();
}

function _endPointerDrag(event) {
  if (!pointerDrag || pointerDrag.pointerId !== event.pointerId) return;
  try { event.currentTarget.releasePointerCapture?.(event.pointerId); } catch {}
  pointerDrag = null;
}

function _wireRoot() {
  closeEl?.addEventListener('click', () => hideXrWebEmbed());
  minimizeEl?.addEventListener('click', () => {
    state.minimized = !state.minimized;
    _refreshShellState();
  });
  followEl?.addEventListener('click', () => {
    state.followPanel = !state.followPanel;
    if (!state.followPanel && !state.manualRect) {
      const rect = windowEl?.getBoundingClientRect?.();
      if (rect) {
        state.manualRect = {
          left: rect.left,
          top: rect.top,
          width: rect.width,
          height: rect.height,
        };
      }
    }
    if (!state.followPanel && state.manualRect) _applyWindowRect(state.manualRect);
    _refreshShellState();
  });
  popoutEl?.addEventListener('click', () => {
    if (!state.href) return;
    window.open(state.href, '_blank', 'noopener,noreferrer');
  });
  rootEl?.addEventListener('beforexrselect', (event) => {
    if (event.target?.closest?.('.xr-web-embed-window')) event.preventDefault();
  });
  windowEl?.querySelector('.xr-web-embed-header')?.addEventListener('pointerdown', (event) => {
    _beginPointerDrag(event, 'move');
  });
  resizeEl?.addEventListener('pointerdown', (event) => _beginPointerDrag(event, 'resize'));
  windowEl?.addEventListener('pointermove', _updatePointerDrag);
  windowEl?.addEventListener('pointerup', _endPointerDrag);
  windowEl?.addEventListener('pointercancel', _endPointerDrag);
  frameEl?.addEventListener('load', () => {
    _setStatus(state.domOverlayType
      ? `Embedded in ${state.domOverlayType} overlay`
      : 'Embedded app ready');
  });
}

export function ensureXrWebEmbedRoot() {
  if (rootEl) return rootEl;

  rootEl = document.createElement('section');
  rootEl.id = 'xr-web-embed-root';
  rootEl.className = 'xr-web-embed-root is-hidden';
  rootEl.setAttribute('aria-hidden', 'true');
  rootEl.setAttribute('aria-label', 'XR web embed');

  windowEl = document.createElement('div');
  windowEl.className = 'xr-web-embed-window';

  const header = document.createElement('div');
  header.className = 'xr-web-embed-header';

  const meta = document.createElement('div');
  meta.className = 'xr-web-embed-meta';

  titleEl = document.createElement('div');
  titleEl.className = 'xr-web-embed-title';
  titleEl.textContent = 'Augmentum';

  statusEl = document.createElement('div');
  statusEl.className = 'xr-web-embed-status';
  statusEl.textContent = 'Select a surface';

  meta.append(titleEl, statusEl);

  const actions = document.createElement('div');
  actions.className = 'xr-web-embed-actions';

  followEl = document.createElement('button');
  followEl.type = 'button';
  followEl.className = 'xr-web-embed-btn xr-web-embed-follow';
  followEl.title = 'Follow spatial panel';
  followEl.setAttribute('aria-label', 'Follow spatial panel');
  followEl.innerHTML = _icon('<path d="M12 2v4"/><path d="M12 18v4"/><path d="M2 12h4"/><path d="M18 12h4"/><circle cx="12" cy="12" r="4"/><path d="m15 9 4-4"/><path d="m9 15-4 4"/>');

  minimizeEl = document.createElement('button');
  minimizeEl.type = 'button';
  minimizeEl.className = 'xr-web-embed-btn xr-web-embed-minimize';
  minimizeEl.title = 'Minimize embedded page';
  minimizeEl.setAttribute('aria-label', 'Minimize embedded page');
  minimizeEl.innerHTML = _icon('<path d="M5 19h14"/><path d="M12 5v9"/><path d="m8 11 4 4 4-4"/>');

  popoutEl = document.createElement('button');
  popoutEl.type = 'button';
  popoutEl.className = 'xr-web-embed-btn';
  popoutEl.title = 'Open embedded page in a tab';
  popoutEl.setAttribute('aria-label', 'Open embedded page in a tab');
  popoutEl.innerHTML = _icon('<path d="M14 3h7v7"/><path d="M10 14 21 3"/><path d="M21 14v7h-7"/><path d="M3 10V3h7"/><path d="M3 21l7-7"/>');

  closeEl = document.createElement('button');
  closeEl.type = 'button';
  closeEl.className = 'xr-web-embed-btn xr-web-embed-close';
  closeEl.title = 'Close embedded page';
  closeEl.setAttribute('aria-label', 'Close embedded page');
  closeEl.innerHTML = _icon('<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>');

  actions.append(followEl, minimizeEl, popoutEl, closeEl);
  header.append(meta, actions);

  frameEl = document.createElement('iframe');
  frameEl.className = 'xr-web-embed-frame';
  frameEl.title = 'Augmentum XR surface';
  frameEl.referrerPolicy = 'same-origin';
  frameEl.setAttribute('allow', 'clipboard-read; clipboard-write; fullscreen; gamepad; microphone; autoplay');

  resizeEl = document.createElement('div');
  resizeEl.className = 'xr-web-embed-resize';
  resizeEl.setAttribute('aria-hidden', 'true');

  windowEl.append(header, frameEl, resizeEl);
  rootEl.append(windowEl);
  document.body.appendChild(rootEl);
  _applyWindowRect(_defaultRect());
  _refreshShellState();
  _wireRoot();
  return rootEl;
}

export function setXrWebEmbedPresenting(active, domOverlayType = '') {
  state.presenting = !!active;
  state.domOverlayType = String(domOverlayType || '').trim();
  if (!state.presenting) state.immersiveSuppressed = false;
  if (!rootEl) return;
  rootEl.dataset.presenting = state.presenting ? 'true' : 'false';
  rootEl.dataset.domOverlayType = state.domOverlayType;
  _refreshShellState();
  if (state.active) {
    _setStatus(state.domOverlayType
      ? `Embedded in ${state.domOverlayType} overlay`
      : 'Embedded app ready');
  }
}

export function setXrWebEmbedCapabilities(capabilities = {}) {
  state.capabilities = { ...capabilities };
  if (!rootEl) return;
  rootEl.dataset.handTracking = capabilities.handTracking ? 'true' : 'false';
  rootEl.dataset.layers = capabilities.layers ? 'true' : 'false';
  rootEl.dataset.mediaLayers = capabilities.mediaLayers ? 'true' : 'false';
  rootEl.dataset.domOverlay = capabilities.domOverlay ? 'true' : 'false';
  if (!state.active) return;
  const parts = [];
  if (capabilities.domOverlay) parts.push(state.domOverlayType || 'overlay');
  if (capabilities.handTracking) parts.push('hands');
  if (capabilities.mediaLayers) parts.push('media layers');
  else if (capabilities.layers) parts.push('layers');
  _setStatus(parts.length ? `XR ${parts.join(' + ')}` : 'XR compatibility mode');
}

export function setXrWebEmbedAnchor(anchor = {}) {
  ensureXrWebEmbedRoot();
  if (state.immersiveSuppressed) {
    state.anchorVisible = false;
    _refreshShellState();
    return;
  }
  state.anchorVisible = anchor.visible !== false;
  if (!state.followPanel || state.minimized || !state.anchorVisible) {
    if (!state.followPanel && state.manualRect) _applyWindowRect(state.manualRect);
    _refreshShellState();
    return;
  }
  const rect = {
    left: Number(anchor.left ?? anchor.x ?? 0),
    top: Number(anchor.top ?? anchor.y ?? 0),
    width: Number(anchor.width ?? 0),
    height: Number(anchor.height ?? 0),
  };
  _applyWindowRect(rect, { anchored: true });
  _refreshShellState();
}

export function showXrWebEmbed(surface, opts = {}) {
  ensureXrWebEmbedRoot();
  const action = _surfaceAction(surface);
  const label = _surfaceLabel(surface, action);
  const href = buildXrEmbedUrl(surface, opts);

  state.action = action;
  state.label = label;
  state.href = href;
  state.minimized = false;
  state.followPanel = opts.followPanel !== false;
  // Quest/browser DOM overlays are monocular UI layers, not true world
  // geometry. Keeping iframe embeds out of the headset during an immersive
  // session avoids per-eye mismatch while the stereo-safe 3D panel remains.
  state.immersiveSuppressed = state.presenting && opts.immersiveOverlay !== true;
  if (state.immersiveSuppressed) {
    state.followPanel = false;
    state.anchorVisible = false;
  }

  titleEl.textContent = label;
  _setStatus(state.immersiveSuppressed
    ? 'Stereo-safe mode: using the 3D panel in headset'
    : 'Loading embedded app...');
  popoutEl.disabled = !href;

  if (frameEl.getAttribute('src') !== href) {
    frameEl.setAttribute('src', href);
  }
  if (state.immersiveSuppressed) _applyWindowRect(_defaultRect());
  else if (!state.followPanel && state.manualRect) _applyWindowRect(state.manualRect);
  else if (!state.anchorVisible) _applyWindowRect(_defaultRect());
  _refreshShellState();
  _setVisible(true);
  return href;
}

export function hideXrWebEmbed(action = '') {
  if (action && state.action && action !== state.action) return false;
  state.immersiveSuppressed = false;
  _setVisible(false);
  _refreshShellState();
  return true;
}

export function getXrWebEmbedState() {
  return { ...state };
}
