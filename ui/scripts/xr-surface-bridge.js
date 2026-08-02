/* ==========================================================================
   XR Surface Bridge

   Turns non-immersive XR surface/panel events into real Augmentum app
   actions. While an immersive session is presenting, the WebXR scene owns the
   visible surface and voice routing; this bridge stays out of the browser UI
   so a headset select never pops panels or navigates out of XR.
   ========================================================================== */

let _wired = false;
let _deps = null;

const MODE_BY_SURFACE = Object.freeze({
  chat: 'passthrough',
  analytical: 'analytical',
  agentic: 'agentic',
  narrative: 'narrative',
  coder: 'coder',
});

// Orb media-surface primary actions → Files panel chip filter.
// Mirrors discovery.js::_KIND_TO_FILES_CHIP; the orb advertises these
// action ids in avatar-xr.js's XR_SURFACES media entry.
const _MEDIA_ACTION_TO_FILES_CHIP = Object.freeze({
  shows_movies: 'movies',
  comics: 'comics',
  audiobooks: 'audiobooks',
  images: 'images',
  local_files: '',
  continue: '',
  games: 'games',
});

const CODER_ACTION_REQUESTS = Object.freeze({
  show_plan: 'Show the current coder plan, workspace status, and any next approval needed.',
  review_diff: 'Review the current git diff and summarize the important risks and changes.',
  run_checks: 'Run the focused tests or checks that make sense for the current workspace.',
  approve: 'Show me the pending coder approval details before taking action.',
});

function _detail(event) {
  return event?.detail || {};
}

function _surface(detail) {
  return String(detail?.action || detail?.surface || '').trim();
}

function _panelAction(detail) {
  return String(detail?.primaryAction || detail?.panelAction || '').trim();
}

function _isImmersive(detail) {
  return detail?.immersive === true || detail?.presentation === 'immersive';
}

function _toast(message, type = 'info') {
  try { _deps?.app?.showToast?.(message, type, 2400); } catch {}
}

function _dismissExcept(except) {
  try {
    _deps?.app?.dismissOverlays?.(except);
  } catch {}
}

function _click(selector) {
  const el = document.querySelector(selector);
  if (!el) return false;
  el.click();
  return true;
}

function _focusSoon(selector) {
  requestAnimationFrame(() => {
    const el = document.querySelector(selector);
    try { el?.focus?.({ preventScroll: true }); } catch { el?.focus?.(); }
    try { el?.select?.(); } catch {}
  });
}

function _switchBrowseTab(tab) {
  document.dispatchEvent(new CustomEvent('augmentum:switch-browse-tab', {
    detail: { tab },
  }));
}

function _openBrowse({ tab = 'browse', focusSearch = false } = {}) {
  _dismissExcept('browse');
  _deps?.openBrowsePanel?.({ skipAutoFocus: true });
  _switchBrowseTab(tab);
  if (focusSearch) _focusSoon('#browse-search-input');
}

function _openFiles({ search = '', focusSearch = false } = {}) {
  _dismissExcept('files');
  if (search) {
    window.dispatchEvent(new CustomEvent('files:open-with-filter', {
      detail: { search },
    }));
  } else {
    _deps?.openFiles?.({ focusSearch });
  }
  if (focusSearch) _focusSoon('#files-search-input');
}

function _openImagePanel({ focusPrompt = false, prompt = '' } = {}) {
  const panel = document.getElementById('image-panel');
  if (panel?.classList.contains('hidden')) {
    document.getElementById('toggle-image-btn')?.click();
  }
  const promptEl = document.getElementById('img-prompt');
  if (prompt && promptEl) {
    promptEl.value = prompt;
    promptEl.dispatchEvent(new Event('input', { bubbles: true }));
  }
  if (focusPrompt) _focusSoon('#img-prompt');
}

async function _openDevices() {
  _dismissExcept('devices');
  try {
    const mod = await import('./media-servers.js');
    await mod.openMediaServers?.();
  } catch (err) {
    console.warn('[xr-surface-bridge] open devices failed', err);
    _toast('Connected Devices could not open', 'error');
  }
}

async function _openMediaPanel(tab = 'discover') {
  try {
    await import('./youtube-panel.js');
  } catch (err) {
    console.warn('[xr-surface-bridge] media panel import failed', err);
  }
  window.dispatchEvent(new CustomEvent('media:open-panel', {
    detail: { tab },
  }));
}

async function _openLibrary(tab = '') {
  _dismissExcept('library');
  try {
    const mod = await import('./library.js');
    // Map legacy tab arg → three-pane sidebar selection. The XR bridge
    // historically passed 'game' to land on Games artifacts; everything
    // else (apps/docs/etc.) maps the same way using Title-Cased labels.
    const TAB_TO_LABEL = {
      game: 'Games', app: 'Apps', doc: 'Documents', md: 'Notes',
      pdf: 'Documents', epub: 'Books', pptx: 'Slides', xlsx: 'Sheets',
      image: 'Images',
    };
    const opts = tab && TAB_TO_LABEL[tab]
      ? { initialSelection: { kind: 'type', id: TAB_TO_LABEL[tab] } }
      : {};
    await mod.openLibrary?.(opts);
  } catch (err) {
    console.warn('[xr-surface-bridge] open library failed', err);
    _toast('Library could not open', 'error');
  }
}

function _openSurface(surface, detail = {}) {
  const mode = MODE_BY_SURFACE[surface];
  if (mode) {
    _deps?.app?.setMode?.(mode);
  }

  switch (surface) {
    case 'chat':
      _deps?.app?.setMode?.('passthrough');
      _focusSoon('#chat-input');
      break;
    case 'analytical':
      _deps?.app?.setMode?.('analytical');
      _focusSoon('#chat-input');
      break;
    case 'agentic':
      _deps?.app?.setMode?.('agentic');
      _focusSoon('#chat-input');
      break;
    case 'narrative':
      _deps?.app?.setMode?.('narrative');
      break;
    case 'coder':
      _deps?.app?.setMode?.('coder');
      break;
    case 'browse':
      _openBrowse({ tab: 'browse', focusSearch: detail.focusSearch });
      break;
    case 'files':
      _openFiles({ focusSearch: detail.focusSearch });
      break;
    case 'notes':
      _openBrowse({ tab: 'notes' });
      break;
    case 'studio':
      _openImagePanel({ focusPrompt: detail.focusPrompt });
      break;
    case 'media':
      // The XR orb's media surface advertises Files-style content
      // (continue / shows_movies / comics / audiobooks / images /
      // local_files / games) — route to Files with the matching chip
      // filter rather than the YouTube discover panel.
      _openFiles({});
      if (detail.primaryAction) {
        const chip = _MEDIA_ACTION_TO_FILES_CHIP[detail.primaryAction] || '';
        if (chip) {
          window.dispatchEvent(new CustomEvent('files:open-with-filter', {
            detail: { chip },
          }));
        }
      }
      break;
    case 'devices':
      _openDevices();
      break;
    case 'games':
      _openLibrary('game');
      break;
  }
}

function _runCoderAction(action) {
  _deps?.app?.setMode?.('coder');
  const request = CODER_ACTION_REQUESTS[action];
  if (!request) return;
  document.dispatchEvent(new CustomEvent('coder:agent-request', {
    detail: { request, source: 'xr-panel' },
  }));
}

function _runBrowseAction(action, detail) {
  const normalized = action === 'summarize_page' ? 'summarize' : action;
  _openBrowse({ tab: 'browse', focusSearch: normalized === 'search' });
  if (detail.query) {
    document.dispatchEvent(new CustomEvent('augmentum:browse-search', {
      detail: { query: detail.query },
    }));
    return;
  }
  if (normalized === 'search') return;
  if (normalized === 'save_source') {
    if (!_click('[data-article-action="bookmark"]')) _toast('Open a page first, then save it.');
    return;
  }
  const aiAction = normalized === 'summarize' ? 'summarize' : normalized;
  if (!_click(`[data-article-ai="${CSS.escape(aiAction)}"]`)) {
    _toast('Open a Browse page first, then use that panel action.');
  }
}

function _runFilesAction(action, detail) {
  const search = detail.query || '';
  _openFiles({ search, focusSearch: action === 'open' || !!search });
  if (action === 'compare' || action === 'attach') {
    _toast('Files is open. Select the items you want to use in VR.');
  }
}

function _runNotesAction(action) {
  _openBrowse({ tab: 'notes' });
  if (action === 'dictate') {
    _click('#browse-new-note-btn');
    _focusSoon('#note-editor-body .ProseMirror, #note-editor-body');
  } else if (action === 'clip') {
    _toast('Notes is open. Use voice to say what you want clipped.');
  }
}

function _runStudioAction(action, detail) {
  _openImagePanel({ focusPrompt: action === 'generate', prompt: detail.prompt || '' });
  if (action === 'save') {
    _openLibrary();
  }
}

function _runMediaAction(action, detail) {
  if (action === 'play' && detail.videoId) {
    import('./youtube-panel.js')
      .then(() => window.dispatchEvent(new CustomEvent('media:play', { detail })))
      .catch(() => window.dispatchEvent(new CustomEvent('media:play', { detail })));
    return;
  }
  _openMediaPanel(action === 'search' ? 'discover' : 'queue');
}

function _runPanelAction(detail) {
  const surface = _surface(detail);
  const action = _panelAction(detail);
  if (!surface) return;
  _openSurface(surface, detail);
  if (!action) return;

  switch (surface) {
    case 'coder':
      _runCoderAction(action);
      break;
    case 'browse':
      _runBrowseAction(action, detail);
      break;
    case 'files':
      _runFilesAction(action, detail);
      break;
    case 'notes':
      _runNotesAction(action);
      break;
    case 'studio':
      _runStudioAction(action, detail);
      break;
    case 'media':
      _runMediaAction(action, detail);
      break;
    case 'devices':
      _openDevices();
      break;
    case 'games':
      _openLibrary('game');
      break;
  }
}

export function initXrSurfaceBridge(deps = {}) {
  _deps = deps;
  if (_wired) return;
  _wired = true;

  window.addEventListener('augmentum:xr-open-surface', (event) => {
    const detail = _detail(event);
    if (_isImmersive(detail)) return;
    if (_panelAction(detail)) return;
    const surface = _surface(detail);
    if (surface) _openSurface(surface, detail);
  });

  window.addEventListener('augmentum:xr-panel-action', (event) => {
    const detail = _detail(event);
    if (_isImmersive(detail)) return;
    _runPanelAction(detail);
  });
}
