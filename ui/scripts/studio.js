/* ==========================================================================
   Artifact Studio — Edit documents, presentations, spreadsheets, charts
   ========================================================================== */

import { escapeHtml, showToast, updateToast, app } from './app.js';
import { ViewStack } from './view-stack.js';
import { copyToClipboard } from './clipboard.js';
import { renderMarkdown, highlightCodeDeferred } from './chat/markdown.js';
import { makeStreamRenderer } from './chat/stream-render.js';

// Lazily-loaded EPUB read-aloud controls ({el, destroy}) shown above the
// rendered book viewer. Torn down whenever the Studio body is swapped.
let _activeReaderControls = null;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  open: false,
  artifactId: null,
  artifactInfo: null,  // full metadata from API
  source: null,        // parsed source JSON
  format: null,        // pdf, docx, pptx, xlsx, png
  sourceType: null,    // document, presentation, spreadsheet, chart
  dirty: false,
  saving: false,
  editor: null,        // Milkdown instance (for documents)
  slides: [],          // array of slide objects for PPTX editor
  currentSlide: 0,     // index of currently selected slide
  gridSheets: [],      // array of sheet objects for XLSX
  currentSheet: 0,     // active sheet tab index
  chartConfig: null,   // chart configuration object
  chartInstance: null,  // Chart.js instance
  aiAbort: null,       // AbortController for AI requests
  theme: 'slate',       // current theme name
  previewOpen: false,   // preview pane visible
  previewDebounce: null, // debounce timer for preview refresh
  openedFromLibrary: false,
  viewMode: 'edit',      // edit | overview
  taskSnapshot: null,    // latest Build snapshot from agentic.js
  sessionPane: 'artifact', // artifact | build (mobile/tablet toggle)
};

let dom = {};

// ---------------------------------------------------------------------------
// AI block output renders through the shared chat markdown (compact mode) +
// incremental stream engine — see studioAi() below. The local
// renderSimpleMarkdown was removed 2026-06-16 for parity with chat/browse.
// ---------------------------------------------------------------------------
function _sessionStatusLabel(status) {
  const s = String(status || '').toLowerCase();
  if (s === 'completed' || s === 'complete') return 'Ready';
  if (s === 'running') return 'Building';
  if (s === 'approval_pending') return 'Awaiting approval';
  if (s === 'failed' || s === 'error') return 'Needs attention';
  return 'Open';
}

function _getSessionLayout() {
  if (!dom.body) return { layout: null, main: null, panel: null, tabs: null };
  const layout = dom.body.querySelector('.studio-session-layout');
  return {
    layout,
    main: layout?.querySelector('.studio-session-main') || null,
    panel: layout?.querySelector('.studio-session-panel') || null,
    tabs: layout?.querySelector('.studio-session-tabs') || null,
  };
}

function _getSessionMainHost() {
  return _getSessionLayout().main || dom.body;
}

function _setSessionPane(pane = 'artifact') {
  state.sessionPane = pane === 'build' ? 'build' : 'artifact';
  if (dom.body) dom.body.dataset.sessionPane = state.sessionPane;
  const tabs = _getSessionLayout().tabs;
  tabs?.querySelectorAll('[data-studio-pane]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.studioPane === state.sessionPane);
  });
}

function _ensureSessionLayout() {
  if (!dom.body) return _getSessionLayout();
  const existing = _getSessionLayout();
  if (existing.layout) {
    _setSessionPane(state.sessionPane || 'artifact');
    return existing;
  }

  const layout = document.createElement('div');
  layout.className = 'studio-session-layout';
  const tabs = document.createElement('div');
  tabs.className = 'studio-session-tabs';
  tabs.innerHTML = `
    <button type="button" class="studio-session-tab active" data-studio-pane="artifact">Artifact</button>
    <button type="button" class="studio-session-tab" data-studio-pane="build">Build</button>
  `;
  const main = document.createElement('div');
  main.className = 'studio-session-main';
  const panel = document.createElement('aside');
  panel.className = 'studio-session-panel';

  const existingChildren = Array.from(dom.body.childNodes);
  existingChildren.forEach((child) => main.appendChild(child));

  layout.appendChild(tabs);
  layout.appendChild(main);
  layout.appendChild(panel);
  dom.body.appendChild(layout);
  _setSessionPane(state.sessionPane || 'artifact');
  return { layout, main, panel, tabs };
}

function _matchingTaskSnapshot() {
  const snapshot = state.taskSnapshot;
  if (!snapshot || typeof snapshot !== 'object') return null;
  const artifacts = Array.isArray(snapshot.artifacts) ? snapshot.artifacts : [];
  if (!state.artifactId) return null;
  return artifacts.some((artifact) => String(artifact?.id || '') === String(state.artifactId))
    ? snapshot
    : null;
}

function _renderSessionStep(step = {}) {
  const label = escapeHtml(step.label || 'Step');
  const stateLabel = escapeHtml(step.state || _sessionStatusLabel(step.status));
  const timing = step.timing ? ` · ${escapeHtml(step.timing)}` : '';
  const output = String(step.output || '').trim();
  const openAttr = output && (step.status === 'running' || step.expanded || output.length < 220) ? ' open' : '';
  return `
    <details class="studio-session-step ${escapeHtml(String(step.status || 'pending'))}"${openAttr}>
      <summary>
        <span class="studio-session-step-index">${String(step.index || 0).padStart(2, '0')}</span>
        <span class="studio-session-step-copy">
          <span class="studio-session-step-label">${label}</span>
          <span class="studio-session-step-meta">${stateLabel}${timing}</span>
        </span>
      </summary>
      ${output ? `<pre class="studio-session-step-output">${escapeHtml(output)}</pre>` : ''}
    </details>
  `;
}

function _refreshStudioSessionPanel() {
  const { layout, panel, tabs } = _ensureSessionLayout();
  if (!panel) return;

  const snapshot = _matchingTaskSnapshot();
  if (layout) layout.dataset.hasSnapshot = snapshot ? 'true' : 'false';
  const artifact = state.artifactInfo || {};
  const filename = artifact.display_name || artifact.filename || 'Artifact';
  const format = String(artifact.format || '').toUpperCase();
  const status = snapshot?.status || '';
  const steps = Array.isArray(snapshot?.steps) ? snapshot.steps : [];
  const siblingArtifacts = Array.isArray(snapshot?.artifacts) ? snapshot.artifacts : [];
  const canEdit = state.viewMode !== 'edit' || !!state.source || !!state.artifactInfo;
  const progress = snapshot?.progress_text || (format ? `${format} artifact` : 'Artifact workspace');
  const sessionLabel = snapshot ? 'Live Build Session' : 'Artifact Workspace';
  const showEditAction = state.viewMode !== 'edit';
  const actionButtons = [
    state.viewMode !== 'overview'
      ? '<button type="button" class="studio-session-action" data-studio-session-action="overview">Overview</button>'
      : '',
    showEditAction && canEdit
      ? '<button type="button" class="studio-session-action primary" data-studio-session-action="edit">Edit</button>'
      : '',
    _artifactSupportsAskAi()
      ? '<button type="button" class="studio-session-action" data-studio-session-action="ask-chat">Ask AI</button>'
      : '',
    artifact.download_url
      ? '<button type="button" class="studio-session-action" data-studio-session-action="download">Download</button>'
      : '',
  ].filter(Boolean).join('');

  panel.innerHTML = `
    <div class="studio-session-scroll">
      <section class="studio-session-hero ${snapshot ? 'is-live' : ''}">
        <div class="studio-session-kicker">${escapeHtml(sessionLabel)}</div>
        <div class="studio-session-title">${escapeHtml(filename)}</div>
        <div class="studio-session-subtitle">${escapeHtml(progress)}</div>
        <div class="studio-session-chips">
          ${format ? `<span class="studio-session-chip">${escapeHtml(format)}</span>` : ''}
          <span class="studio-session-chip ${snapshot ? 'is-live' : ''}">${escapeHtml(_sessionStatusLabel(status))}</span>
          ${snapshot && steps.length ? `<span class="studio-session-chip">${steps.filter((step) => /complete/i.test(step.status || '')).length}/${steps.length} steps</span>` : ''}
        </div>
        <div class="studio-session-actions">${actionButtons}</div>
      </section>

      ${snapshot ? `
        <section class="studio-session-section">
          <div class="studio-session-section-head">
            <h3>Build Timeline</h3>
            <span>${escapeHtml(snapshot.title || '')}</span>
          </div>
          <div class="studio-session-steps">
            ${steps.map((step) => _renderSessionStep(step)).join('')}
          </div>
        </section>
      ` : `
        <section class="studio-session-section">
          <div class="studio-session-empty">
            Opened from the library or outside an active build. This workspace still supports overview, editing, and AI-assisted revisions.
          </div>
        </section>
      `}

      ${snapshot && siblingArtifacts.length > 1 ? `
        <section class="studio-session-section">
          <div class="studio-session-section-head">
            <h3>Outputs</h3>
            <span>${siblingArtifacts.length} files</span>
          </div>
          <div class="studio-session-artifacts">
            ${siblingArtifacts.map((item) => `
              <button type="button"
                      class="studio-session-artifact${String(item?.id || '') === String(state.artifactId) ? ' active' : ''}"
                      data-studio-session-artifact="${escapeHtml(String(item?.id || ''))}">
                <span class="studio-session-artifact-title">${escapeHtml(item?.title || 'Artifact')}</span>
                <span class="studio-session-artifact-meta">${escapeHtml(item?.meta || '')}</span>
              </button>
            `).join('')}
          </div>
        </section>
      ` : ''}
    </div>
  `;

  tabs?.querySelector('[data-studio-pane="build"]')?.classList.toggle('has-live', !!snapshot);
  _setSessionPane(state.sessionPane || 'artifact');
}

// ---------------------------------------------------------------------------
// Init (called lazily on first open)
// ---------------------------------------------------------------------------
let _initialized = false;

function _ensureInit() {
  if (_initialized) return;
  _initialized = true;

  const overlay = document.getElementById('studio-overlay');
  if (!overlay) return;

  dom = {
    overlay,
    backBtn: overlay.querySelector('#studio-back-btn'),
    closeBtn: overlay.querySelector('#studio-close-btn'),
    artifactName: overlay.querySelector('#studio-artifact-name'),
    formatBadge: overlay.querySelector('#studio-format-badge'),
    toolbar: overlay.querySelector('#studio-toolbar'),
    body: overlay.querySelector('#studio-body'),
    loading: overlay.querySelector('#studio-loading'),
    aiBtn: overlay.querySelector('#studio-ai-btn'),
    askAiBtn: overlay.querySelector('#studio-ask-ai-btn'),
    convertWrap: overlay.querySelector('#studio-convert-wrap'),
    convertBtn: overlay.querySelector('#studio-convert-btn'),
    convertMenu: overlay.querySelector('#studio-convert-menu'),
    downloadBtn: overlay.querySelector('#studio-download-btn'),
    castBtn: overlay.querySelector('#studio-cast-btn'),
    versionsBtn: overlay.querySelector('#studio-versions-btn'),
    saveBtn: overlay.querySelector('#studio-save-btn'),
    askBar: overlay.querySelector('#studio-ask-bar'),
    askInput: overlay.querySelector('#studio-ask-input'),
    askBtn: overlay.querySelector('#studio-ask-btn'),
    headerActions: overlay.querySelector('.studio-header-actions'),
    themeBtn: overlay.querySelector('#studio-theme-btn'),
    themePicker: overlay.querySelector('#studio-theme-picker'),
    previewBtn: overlay.querySelector('#studio-preview-btn'),
    listenBtn: overlay.querySelector('#studio-listen-btn'),
    shortcutHint: overlay.querySelector('#studio-shortcut-hint'),
    shortcutsOverlay: overlay.querySelector('#studio-shortcuts-overlay'),
    shortcutsClose: overlay.querySelector('#studio-shortcuts-close'),
  };

  // Back button — return to library
  dom.backBtn?.addEventListener('click', () => {
    const wasFromLibrary = closeStudio();
    if (wasFromLibrary) {
      import('./library.js').then(m => m.openLibrary()).catch(() => {});
    }
  });

  // Close button — close studio (and library if it's behind)
  dom.closeBtn?.addEventListener('click', () => {
    const wasFromLibrary = closeStudio();
    if (wasFromLibrary) {
      import('./library.js').then(m => m.closeLibrary()).catch(() => {});
    }
  });

  // Save button
  dom.saveBtn?.addEventListener('click', saveArtifact);

  // Download button — flush any pending edit first so the downloaded file
  // reflects the latest changes (e.g. a just-picked ebook theme), then bust
  // the browser cache so a re-download after another edit isn't served stale.
  dom.downloadBtn?.addEventListener('click', async () => {
    const url = state.artifactInfo?.download_url;
    if (!url) return;
    await _flushPendingSave();
    window.open(`${url}${url.includes('?') ? '&' : '?'}v=${Date.now()}`, '_blank');
  });

  // Cast button — send the artifact preview to a paired TV. Flush pending
  // edits first so the receiver renders the latest content (mirrors the
  // downloadBtn flush). studio/cast.js owns receiver-picker + error toasts;
  // we catch 'cancelled' silently so closing the picker isn't an error.
  dom.castBtn?.addEventListener('click', async () => {
    if (!state.artifactId) return;
    await _flushPendingSave();
    try {
      const { castArtifactPreview } = await import('./studio/cast.js');
      await castArtifactPreview(
        state.artifactId,
        state.artifactInfo?.display_name || 'Artifact',
      );
    } catch (err) {
      if (err?.message !== 'cancelled') {
        console.warn('[studio] cast failed', err);
      }
    }
  });

  // Versions button — opens the right-edge drawer listing manual-save
  // snapshots. The drawer is its own module; we just hand it the artifact
  // id + a callback that re-opens Studio after a restore so the editor
  // surface reflects the rolled-back source.
  dom.versionsBtn?.addEventListener('click', async () => {
    if (!state.artifactId) return;
    await _flushPendingSave();
    const restoredArtifactId = state.artifactId;
    try {
      const { openVersionsDrawer } = await import('./studio/versions.js');
      await openVersionsDrawer({
        artifactId: restoredArtifactId,
        onRestored: () => { openStudio(restoredArtifactId, {}); },
      });
    } catch (err) {
      console.warn('[studio] versions drawer failed', err);
    }
  });

  // Ask bar
  dom.askInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const q = dom.askInput.value.trim();
      if (q) { studioAiAction('ask', q); dom.askInput.value = ''; }
    }
  });
  dom.askBtn?.addEventListener('click', () => {
    const q = dom.askInput?.value.trim();
    if (q) { studioAiAction('ask', q); dom.askInput.value = ''; }
  });

  // AI button — prefer the palette tool if it's mounted (Phase 3+); fall
  // back to the legacy popover for editors that haven't grown a palette yet.
  dom.aiBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (_activePalette?.activate) {
      _activePalette.activate('ai');
      return;
    }
    const popover = dom.headerActions?.querySelector('.studio-ai-popover');
    popover?.classList.toggle('hidden');
  });

  // Universal "Ask AI about this file" — hands off to app.js via a custom
  // event that ingests the artifact as a chat attachment in a fresh session.
  dom.askAiBtn?.addEventListener('click', () => {
    if (state.artifactId) _artifactAskAi(state.artifactId);
  });

  // Convert dropdown — populated when the artifact opens (see _updateConvertMenu).
  // Toggle + click-outside dismissal mirrors the theme-picker pattern.
  dom.convertBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    dom.convertMenu?.classList.toggle('hidden');
  });
  document.addEventListener('click', (e) => {
    if (dom.convertMenu && !dom.convertMenu.classList.contains('hidden') &&
        !e.target.closest('#studio-convert-menu') &&
        !e.target.closest('#studio-convert-btn')) {
      dom.convertMenu.classList.add('hidden');
    }
  });
  dom.convertMenu?.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-convert-to]');
    if (!btn) return;
    _convertArtifact(btn.dataset.convertTo);
    dom.convertMenu.classList.add('hidden');
  });

  // Close popover on click outside
  document.addEventListener('click', (e) => {
    const popover = dom.headerActions?.querySelector('.studio-ai-popover');
    if (popover && !popover.classList.contains('hidden') &&
        !e.target.closest('.studio-ai-popover') &&
        !e.target.closest('#studio-ai-btn')) {
      popover.classList.add('hidden');
    }
  });

  // Theme picker toggle
  dom.themeBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    dom.themePicker?.classList.toggle('hidden');
  });

  // Close theme picker on click outside
  document.addEventListener('click', (e) => {
    if (dom.themePicker && !dom.themePicker.classList.contains('hidden') &&
        !e.target.closest('#studio-theme-picker') &&
        !e.target.closest('#studio-theme-btn')) {
      dom.themePicker.classList.add('hidden');
    }
  });

  // Preview toggle
  dom.previewBtn?.addEventListener('click', togglePreview);

  // Listen (read-aloud) — document artifacts only. Uses the Milkdown
  // editor's current markdown when available so unsaved edits get read,
  // falls back to the cached source_json sections on initial open.
  dom.listenBtn?.addEventListener('click', _onListenClick);

  // Shortcut help
  dom.shortcutHint?.addEventListener('click', _toggleShortcutHelp);
  dom.shortcutsClose?.addEventListener('click', _closeShortcutHelp);
  dom.shortcutsOverlay?.addEventListener('click', (e) => {
    if (e.target === dom.shortcutsOverlay) _closeShortcutHelp();
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (!state.open) return;
    if (e.key === 'Escape') {
      // Nested overlays eat Escape first so pressing it inside them
      // doesn't tear down the whole Studio. Find modal, slide sorter,
      // and image crop each register their own listeners — bail here so
      // the close-Studio path doesn't run in the same keystroke.
      if (document.querySelector('.studio-find-modal.is-open')) return;
      if (state._sorterActive) return;
      if (state._cropActive) return;
      if (state._inpaintActive) {
        // Mask editor owns Esc — cancel inpaint rather than closing Studio.
        e.preventDefault();
        _imageInpaintExit(dom.overlay.querySelector('.studio-image-viewer'));
        return;
      }
      if (!dom.shortcutsOverlay?.classList.contains('hidden')) {
        _closeShortcutHelp();
        return;
      }
      const wasFromLibrary = closeStudio();
      if (wasFromLibrary) {
        import('./library.js').then(m => m.openLibrary()).catch(() => {});
      }
    } else if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      saveArtifact();
    } else if ((e.ctrlKey || e.metaKey) && e.key === '/') {
      e.preventDefault();
      _toggleShortcutHelp();
    } else if ((e.ctrlKey || e.metaKey) && (e.key === 'f' || e.key === 'F')) {
      // Claim Ctrl+F only when an editor is open and registered a find
      // provider. Otherwise leave the native browser Find alone.
      if (_findScopeForCurrentEditor()) {
        e.preventDefault();
        import('./studio-find.js').then(m => m.openFind(_findScopeForCurrentEditor()));
      }
    } else if ((e.ctrlKey || e.metaKey) && (e.key === 'h' || e.key === 'H')) {
      if (_findScopeForCurrentEditor()) {
        e.preventDefault();
        import('./studio-find.js').then(m => m.openFind(_findScopeForCurrentEditor(), { replace: true }));
      }
    } else if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
      // Grid owns Ctrl+Z / Ctrl+Shift+Z because it keeps a coarse snapshot
      // stack. Other editors rely on the browser's native contenteditable
      // undo buffer, which we leave untouched.
      if (state.sourceType === 'spreadsheet') {
        e.preventDefault();
        if (e.shiftKey) _gridRedo(); else _gridUndo();
      }
    } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y')) {
      if (state.sourceType === 'spreadsheet') {
        e.preventDefault();
        _gridRedo();
      }
    }
  });

  // AI block actions (event delegation on body)
  dom.body?.addEventListener('click', (e) => {
    const paneBtn = e.target.closest('[data-studio-pane]');
    if (paneBtn) {
      _setSessionPane(paneBtn.dataset.studioPane || 'artifact');
      return;
    }

    const sessionArtifactBtn = e.target.closest('[data-studio-session-artifact]');
    if (sessionArtifactBtn) {
      const targetId = sessionArtifactBtn.dataset.studioSessionArtifact;
      if (targetId) openStudio(targetId, { mode: state.viewMode, fromLibrary: state.openedFromLibrary });
      return;
    }

    const sessionActionBtn = e.target.closest('[data-studio-session-action]');
    if (sessionActionBtn) {
      const action = sessionActionBtn.dataset.studioSessionAction;
      if (action === 'edit' && state.artifactId) {
        openStudio(state.artifactId, { mode: 'edit', fromLibrary: state.openedFromLibrary });
        return;
      }
      if (action === 'overview' && state.artifactId) {
        openStudio(state.artifactId, { mode: 'overview', fromLibrary: state.openedFromLibrary });
        return;
      }
      if (action === 'ask-chat' && state.artifactId) {
        _artifactAskAi(state.artifactId);
        return;
      }
      if (action === 'download' && state.artifactInfo?.download_url) {
        const url = state.artifactInfo.download_url;
        _flushPendingSave().then(() => window.open(`${url}${url.includes('?') ? '&' : '?'}v=${Date.now()}`, '_blank'));
        return;
      }
    }

    const btn = e.target.closest('.studio-ai-block-btn');
    if (!btn) return;
    const block = btn.closest('.studio-ai-block');
    if (!block) return;
    const action = btn.dataset.action;
    const md = block.dataset.markdown || '';
    if (action === 'insert') {
      insertAiBlock(md);
      block.remove();
    } else if (action === 'copy') {
      copyToClipboard(md)
        .then((ok) => showToast(ok ? 'Copied' : 'Copy failed', ok ? 'success' : 'error'));
    } else if (action === 'remove') {
      block.style.opacity = '0';
      block.style.transform = 'translateY(-8px)';
      block.style.transition = 'all 0.2s ease';
      setTimeout(() => block.remove(), 200);
    }
  });

  document.addEventListener('augmentum:agentic-task-snapshot', (e) => {
    state.taskSnapshot = e.detail || null;
    if (state.open) _refreshStudioSessionPanel();
  });
}

// ---------------------------------------------------------------------------
// Open / Close
// ---------------------------------------------------------------------------
/**
 * @param {string} artifactId
 * @param {Object} [opts]
 * @param {boolean} [opts.forceVisualPdf] — force the visual PDF editor even if source_json exists
 * @param {"edit"|"overview"} [opts.mode] — preview-first overview or editable studio
 */
export async function openStudio(artifactId, opts = {}) {
  _ensureInit();
  if (!dom.overlay) return;

  // Switching mode (Edit ⇄ Overview) or to a sibling artifact re-enters
  // openStudio rather than going through closeStudio, and the state reset
  // below clears state.dirty — which would cancel the debounced autosave and
  // silently drop the last edit (e.g. a theme pick made seconds before
  // clicking Overview). Flush it first so the change is persisted (and the
  // re-render that produces the new preview has actually run) before we tear
  // the current editor down.
  if (state.open) await _flushPendingSave();

  state.artifactId = artifactId;
  state.open = true;
  state.dirty = false;
  state.source = null;
  state.editor = null;
  state._forceVisualPdf = opts.forceVisualPdf || false;
  state.openedFromLibrary = opts.fromLibrary || false;
  state.viewMode = opts.mode === 'overview' ? 'overview' : 'edit';
  state.sessionPane = 'artifact';

  try {
    const agentic = await import('./agentic.js');
    if (typeof agentic.getAgenticTaskSnapshot === 'function') {
      state.taskSnapshot = agentic.getAgenticTaskSnapshot() || state.taskSnapshot;
    }
  } catch { /* best effort */ }

  if (dom.backBtn) dom.backBtn.style.display = state.openedFromLibrary ? '' : 'none';
  dom.overlay.classList.remove('hidden', 'leaving');
  dom.overlay.classList.add('entering');
  dom.overlay.addEventListener('animationend', () => dom.overlay.classList.remove('entering'), { once: true });

  // Track in ViewStack. onClose forces a close (skipping the dirty-state
  // confirm) — programmatic pops (mode change, library close) are
  // authoritative, and a blocking confirm mid-pop would desync the stack.
  ViewStack.pushOverlay('studio', { onClose: () => _doCloseStudio({ force: true }) });
  dom.loading.style.display = '';
  dom.loading.classList.remove('hidden');
  dom.body.style.display = '';
  dom.body.innerHTML = '';
  dom.body.appendChild(dom.loading);
  dom.askBar.style.display = 'none';
  dom.saveBtn.disabled = true;
  dom.saveBtn.style.display = '';
  _updateDirtyState(false);

  // Fetch artifact source
  try {
    const resp = await fetch(`/api/studio/${artifactId}`);
    if (!resp.ok) {
      showToast('Failed to load artifact', 'error');
      closeStudio();
      return;
    }
    const data = await resp.json();
    state.artifactInfo = data;
    state.source = data.source;
    state.format = data.format;
    state.sourceType = data.source?.type || null;

    // Update header
    dom.artifactName.textContent = data.display_name || data.filename || 'Untitled';
    dom.formatBadge.textContent = (data.format || '').toUpperCase();
    dom.formatBadge.dataset.format = data.format || '';

    // Route to format-specific editor
    // PDFs go to visual editor when: no source (imported), or explicitly forced (annotate button)
    const isPdf = data.format === 'pdf' || (data.filename || '').endsWith('.pdf');
    const isPdfVisual = isPdf && (!state.source || state._forceVisualPdf);
    const ext = (data.filename || '').split('.').pop()?.toLowerCase() || data.format || '';
    const isOverview = state.viewMode === 'overview';

    if (isOverview) {
      openArtifactOverview(data.id, data.filename);
    } else if (isPdfVisual) {
      await openPdfVisualEditor(data.id);
    } else if (!state.source && ['txt', 'md', 'markdown', 'rst', 'log'].includes(ext)) {
      // Plain text / markdown — load content and edit with Milkdown
      await openImportedTextEditor(data.id, ext);
    } else if (!state.source && ['png', 'jpg', 'jpeg', 'svg', 'gif', 'webp'].includes(ext)) {
      // Images — preview viewer
      openImageViewer(data.id, data.filename);
    } else if (!state.source && ['html', 'htm'].includes(ext)) {
      // HTML — iframe preview with source view
      openHtmlPreview(data.id, data.filename);
    } else if (!state.source && ['csv'].includes(ext)) {
      // CSV — parse and show in a simple table
      await openCsvViewer(data.id);
    } else if (!state.source && ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'opus', 'aac'].includes(ext)) {
      // Audio — native <audio> with metadata + transcribe hook.
      openAudioViewer(data.id, data.filename, ext);
    } else if (!state.source && ['mp4', 'mov', 'webm', 'mkv', 'avi'].includes(ext)) {
      // Video — native <video> with metadata + transcribe hook.
      openVideoViewer(data.id, data.filename, ext);
    } else if (!state.source && ['docx', 'pptx', 'xlsx', 'epub', 'json'].includes(ext)) {
      // Binary/structured formats the backend can render — use preview iframe
      openBackendPreview(data.id, data.filename);
    } else if (!state.source) {
      // Unknown format — show download-only fallback
      showFileInfo(data);
    } else if (state.sourceType === 'document') {
      await openDocumentEditor();
    } else if (state.sourceType === 'presentation') {
      await openSlideEditor();
    } else if (state.sourceType === 'spreadsheet') {
      openGridEditor();
    } else if (state.sourceType === 'chart') {
      await openChartEditor();
    } else if (state.sourceType === 'ebook') {
      openEbookEditor();
    } else {
      showLegacyNotice();
    }

    // Show/hide preview button based on format
    // PDF and DOCX get split-view preview; others have built-in visual editors
    const showPreview = !isOverview && ['document'].includes(state.sourceType);
    if (dom.previewBtn) dom.previewBtn.style.display = showPreview ? '' : 'none';
    const isReadOnlyStudioSource = !state.source && ['docx', 'pptx', 'xlsx', 'epub', 'json'].includes(ext);
    if (dom.saveBtn) dom.saveBtn.style.display = (isOverview || isReadOnlyStudioSource) ? 'none' : '';

    // Theme picker: shown in the editor for any themeable type, and also in
    // the read-only Overview for ebooks — picking a theme there re-renders the
    // EPUB and reloads the preview iframe in place (other types need their
    // editor open to be re-rendered safely, so they stay editor-only).
    const canTheme = _THEMEABLE_SOURCE_TYPES.has(state.sourceType)
      && (!isOverview || state.sourceType === 'ebook');
    if (dom.themeBtn) dom.themeBtn.style.display = canTheme ? '' : 'none';
    if (!canTheme) dom.themePicker?.classList.add('hidden');
    if (canTheme && isOverview) {
      // Overview doesn't run openEbookEditor, so seed the picker here.
      state.theme = state.source?.theme?.preset || state.source?.theme || 'storybook';
      loadThemes();
    }

    // "Ask AI about this file" only makes sense for formats the chat can
    // actually ingest — hide it for ebooks, audio, video, ROMs, etc.
    if (dom.askAiBtn) dom.askAiBtn.style.display = _artifactSupportsAskAi() ? '' : 'none';

    // Populate the convert dropdown based on source type + file format.
    _updateConvertMenu();

    // Listen button: visible whenever the artifact carries readable prose —
    // sectioned documents (markdown/docx) and imported plain-text. Slides,
    // sheets, charts, images and PDFs in visual mode have no linear text
    // flow a TTS engine can read coherently, so we keep it hidden there.
    const canListen = (
      state.sourceType === 'document' ||
      state.sourceType === 'imported_text'
    );
    if (dom.listenBtn) dom.listenBtn.style.display = !isOverview && canListen ? '' : 'none';
    _refreshStudioSessionPanel();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
    const wasFromLibrary = closeStudio();
    if (wasFromLibrary) {
      import('./library.js').then(m => m.openLibrary()).catch(() => {});
    }
  }
}

// Re-entry guard for ViewStack sync. The close path pops the stack at the
// end, which fires onClose → _doCloseStudio — this flag catches that re-entry.
let _studioCloseViaStack = false;

export function closeStudio() {
  return _doCloseStudio({ force: false });
}

function _doCloseStudio({ force = false } = {}) {
  if (_studioCloseViaStack) return false;
  if (!state.open) return false;
  if (!force && state.dirty && !confirm('You have unsaved changes. Close anyway?')) return false;

  const returnToLibrary = state.openedFromLibrary;

  state.open = false;
  state.artifactId = null;
  state.source = null;
  state.dirty = false;
  state.openedFromLibrary = false;
  state.viewMode = 'edit';
  state.sessionPane = 'artifact';
  _updateDirtyState(false);

  // Stop any active read-aloud — the module is only imported if we ever
  // triggered Listen, so this is cheap when TTS was never used.
  import('./read-aloud.js').then(m => { if (m.isReadAloudActive()) m.stopReadAloud(); }).catch(() => {});
  if (_activeReaderControls) { try { _activeReaderControls.destroy(); } catch { /* noop */ } _activeReaderControls = null; }

  // Tear down the Tool Palette so its in-mount listeners + any pending
  // staged-image discard fires before we drop state.
  if (_activePalette) {
    try { _activePalette.destroy(); } catch { /* noop */ }
    _activePalette = null;
    _activeImageTool = null;
  }

  // Detach image viewer resize observer if it was installed this session.
  if (state._imageViewerDispose) {
    try { state._imageViewerDispose(); } catch {}
    state._imageViewerDispose = null;
  }
  // Clear autosave timers so a scheduled save from the prior artifact
  // can't fire against a closed/empty studio state.
  clearTimeout(state._autosaveTimer);
  clearInterval(state._saveTicker);
  clearInterval(state._wordCountTicker);
  clearInterval(state._outlineTicker);
  // Sorter mode installs a document-level keydown listener — detach it so
  // closing Studio mid-sort doesn't leave a dangling handler.
  if (state._sorterKeyHandler) {
    document.removeEventListener('keydown', state._sorterKeyHandler);
    state._sorterKeyHandler = null;
  }
  state._sorterActive = false;
  state._sorterSel = null;
  state._sorterAnchor = null;
  // Tear down the inpaint mask editor if one is mounted — its ResizeObserver
  // would otherwise outlive the Studio session.
  if (state._inpaintMask) {
    try { state._inpaintMask.destroy(); } catch { /* noop */ }
    state._inpaintMask = null;
  }
  state._inpaintActive = false;
  state._saveStatus = 'idle';
  _renderSaveStatus();
  dom.overlay?.classList.remove('studio-focus-mode');
  // Close the find modal if it was open for the prior artifact.
  import('./studio-find.js').then(m => { if (m.isFindOpen()) m.closeFind(); }).catch(() => {});

  // Animate out then hide
  if (dom.overlay) {
    dom.overlay.classList.add('leaving');
    dom.overlay.addEventListener('animationend', () => {
      dom.overlay.classList.remove('leaving');
      dom.overlay.classList.add('hidden');
    }, { once: true });
  }

  _destroyEditor();

  // Destroy PDF visual editor
  if (state._pdfDestroy) {
    try { state._pdfDestroy(); } catch {}
    state._pdfDestroy = null;
    state._pdfGetBytes = null;
  }

  // Destroy Chart.js instance
  if (state.chartInstance) {
    try { state.chartInstance.destroy(); } catch {}
    state.chartInstance = null;
  }
  state.chartConfig = null;

  // Abort any AI request
  if (state.aiAbort) {
    state.aiAbort.abort();
    state.aiAbort = null;
  }

  state.previewOpen = false;
  state.theme = 'slate';
  dom.body.style.flexDirection = '';
  delete dom.body.dataset.sessionPane;

  // Sync ViewStack — skip if we were re-entered via the stack's onClose.
  if (!force && ViewStack.hasOverlay('studio')) {
    _studioCloseViaStack = true;
    try { ViewStack.popOverlay('studio'); }
    finally { _studioCloseViaStack = false; }
  }

  return returnToLibrary;
}

export function initStudio() {
  // Lazy init — nothing to do here, _ensureInit() runs on first open
}

// ---------------------------------------------------------------------------
// Legacy notice (no source_json)
// ---------------------------------------------------------------------------
function showLegacyNotice() {
  _hideLoading();
  dom.body.innerHTML = `
    <div class="studio-legacy-notice">
      <div class="studio-fallback-icon">\uD83D\uDD27</div>
      <h3>Editing not available</h3>
      <p>This artifact was created before the Studio editor was available.
         To enable editing, re-create the artifact — new artifacts automatically
         store their source data for editing.</p>
      <a class="browse-action-pill" href="${escapeHtml(state.artifactInfo?.download_url || '')}"
         target="_blank" rel="noopener noreferrer" style="display:inline-flex">
        Download original
      </a>
    </div>
  `;
}

/** Hide the loading skeleton */
function _hideLoading() {
  if (dom.loading) {
    dom.loading.style.display = 'none';
    dom.loading.classList.add('hidden');
  }
}

// ---------------------------------------------------------------------------
// Imported Text Editor (TXT/MD/RST/LOG — load content, edit with Milkdown)
// ---------------------------------------------------------------------------
async function openImportedTextEditor(artifactId, ext) {
  _hideLoading();
  dom.askBar.style.display = '';
  dom.saveBtn.disabled = false;

  try {
    const resp = await fetch(`/api/artifacts/${artifactId}/download`);
    if (!resp.ok) throw new Error('Failed to fetch file');
    const text = await resp.text();

    // For non-markdown, wrap in a code block
    const markdown = (ext === 'md' || ext === 'markdown') ? text : text;

    const pageEl = document.createElement('div');
    pageEl.id = 'studio-doc-page';
    pageEl.className = 'studio-doc-page';
    dom.body.innerHTML = '';
    dom.body.appendChild(pageEl);

    await loadMilkdownEditor(pageEl, markdown);
    state.sourceType = 'imported_text';
    state._importedArtifactId = artifactId;
    state._importedExt = ext;
    _registerDocFindProvider();
    _wordCountInstall(() => getEditorMarkdown());
  } catch (e) {
    dom.body.innerHTML = `<div class="studio-legacy-notice"><p>Could not load file: ${escapeHtml(e.message)}</p></div>`;
  }
}

// ---------------------------------------------------------------------------
// Image Viewer (PNG/JPG/SVG — simple view with zoom)
// ---------------------------------------------------------------------------
// Image viewer with wheel-zoom (toward cursor), click-drag pan, fit/100%/fill
// pill controls, double-click to toggle fit↔100%, and keyboard shortcuts
// (+/-/0). We drive a CSS transform on the <img> rather than resizing the
// element itself — keeps the math simple and the GPU happy on large photos.
function openImageViewer(artifactId, filename) {
  _hideLoading();
  dom.saveBtn.disabled = true;

  const src = `/api/artifacts/${escapeHtml(artifactId)}/download`;
  dom.body.innerHTML = `
    <div class="studio-image-viewer" id="studio-image-viewer" tabindex="0" data-artifact-id="${escapeHtml(artifactId)}">
      <div class="studio-image-stage" id="studio-image-stage">
        <img class="studio-image-el" id="studio-image-el"
             src="${src}" alt="${escapeHtml(filename || 'Image')}"
             draggable="false" decoding="async">
        <div class="studio-crop-overlay hidden" id="studio-crop-overlay" aria-hidden="true">
          <div class="studio-crop-mask" data-mask="top"></div>
          <div class="studio-crop-mask" data-mask="left"></div>
          <div class="studio-crop-mask" data-mask="right"></div>
          <div class="studio-crop-mask" data-mask="bottom"></div>
          <div class="studio-crop-rect" id="studio-crop-rect">
            <span class="studio-crop-handle" data-handle="nw"></span>
            <span class="studio-crop-handle" data-handle="n"></span>
            <span class="studio-crop-handle" data-handle="ne"></span>
            <span class="studio-crop-handle" data-handle="e"></span>
            <span class="studio-crop-handle" data-handle="se"></span>
            <span class="studio-crop-handle" data-handle="s"></span>
            <span class="studio-crop-handle" data-handle="sw"></span>
            <span class="studio-crop-handle" data-handle="w"></span>
          </div>
        </div>
      </div>
      <div class="studio-image-controls" id="studio-image-controls">
        <button class="studio-image-ctrl" data-zoom="out" title="Zoom out (-)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </button>
        <span class="studio-image-pct" id="studio-image-pct">100%</span>
        <button class="studio-image-ctrl" data-zoom="in" title="Zoom in (+)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </button>
        <span class="studio-image-sep"></span>
        <button class="studio-image-ctrl" data-fit="fit" title="Fit (0)">Fit</button>
        <button class="studio-image-ctrl" data-fit="100" title="Actual size">100%</button>
        <button class="studio-image-ctrl" data-fit="fill" title="Fill">Fill</button>
        <span class="studio-image-sep"></span>
        <button class="studio-image-ctrl" data-rotate="-90" title="Rotate left">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
        </button>
        <button class="studio-image-ctrl" data-rotate="90" title="Rotate right">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
        </button>
        <span class="studio-image-sep"></span>
        <button class="studio-image-ctrl" data-crop="enter" title="Crop">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/></svg>
          Crop
        </button>
        <button class="studio-image-ctrl studio-image-ctrl--primary hidden" data-crop="apply" title="Apply crop (Enter)">Apply</button>
        <button class="studio-image-ctrl hidden" data-crop="cancel" title="Cancel (Esc)">Cancel</button>
        <span class="studio-image-sep"></span>
        <button class="studio-image-ctrl" data-ai="upscale" title="Upscale 4x (AI)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v18m-9-9h18"/><path d="m17 7-5 5-5-5M7 17l5-5 5 5" opacity="0.45"/></svg>
          Upscale
        </button>
        <button class="studio-image-ctrl" data-ai="remove-bg" title="Remove background (AI)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 15l6-6 5 5 4-4 3 3" opacity="0.35"/><circle cx="9" cy="9" r="1.5"/></svg>
          Remove BG
        </button>
        <button class="studio-image-ctrl" data-ai="inpaint" title="Inpaint a region (AI)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>
          Inpaint
        </button>
      </div>
      <div class="studio-image-hint" id="studio-image-hint">Scroll to zoom · drag to pan · double-click to toggle</div>
    </div>
  `;

  _wireImageViewer();
}

function _wireImageViewer() {
  const stage  = document.getElementById('studio-image-stage');
  const img    = document.getElementById('studio-image-el');
  const root   = document.getElementById('studio-image-viewer');
  const pctEl  = document.getElementById('studio-image-pct');
  if (!stage || !img || !root) return;

  // view state: scale is absolute (1 = natural pixels); tx/ty pan the image;
  // rot rotates in 90° increments. baseFit holds the computed "fit-to-
  // viewport" scale — it's the scale we return to on "Fit" / reset and also
  // serves as the zoom-out floor. When rot is odd (90°/270°), naturalWidth
  // and naturalHeight swap for fit/fill computations.
  const view = { scale: 1, tx: 0, ty: 0, baseFit: 1, mode: 'fit', rot: 0 };
  const MIN_ABS = 0.05;
  const MAX_ABS = 32;

  const apply = () => {
    img.style.transform = `translate(${view.tx}px, ${view.ty}px) rotate(${view.rot}deg) scale(${view.scale})`;
    if (pctEl) pctEl.textContent = `${Math.round(view.scale * 100)}%`;
    root.querySelectorAll('.studio-image-ctrl[data-fit]').forEach(b => {
      b.classList.toggle('active', b.dataset.fit === view.mode);
    });
  };

  const _dims = () => {
    // Odd 90° rotations swap dimensions for fit/fill reasoning.
    const sr = stage.getBoundingClientRect();
    let nw = img.naturalWidth || sr.width;
    let nh = img.naturalHeight || sr.height;
    if (((view.rot % 360) + 360) % 360 % 180 !== 0) [nw, nh] = [nh, nw];
    return { sr, nw, nh };
  };
  const computeFit = () => {
    const { sr, nw, nh } = _dims();
    if (!nw || !nh) return 1;
    return Math.min(sr.width / nw, sr.height / nh, 1);
  };
  const computeFill = () => {
    const { sr, nw, nh } = _dims();
    if (!nw || !nh) return 1;
    return Math.max(sr.width / nw, sr.height / nh);
  };

  const resetTo = (mode) => {
    view.mode = mode;
    view.tx = 0;
    view.ty = 0;
    view.scale = mode === 'fit' ? view.baseFit : mode === 'fill' ? computeFill() : 1;
    apply();
  };

  const onReady = () => {
    view.baseFit = computeFit();
    resetTo('fit');
  };
  if (img.complete && img.naturalWidth) onReady();
  else img.addEventListener('load', onReady, { once: true });

  // Zoom toward the cursor: compute the image-space point under the cursor,
  // scale, then translate so that same point lands back under the cursor.
  const zoomAt = (clientX, clientY, factor) => {
    const sr = stage.getBoundingClientRect();
    const cx = clientX - sr.left - sr.width / 2;
    const cy = clientY - sr.top - sr.height / 2;
    const minS = Math.max(MIN_ABS, view.baseFit * 0.5);
    const next = Math.min(MAX_ABS, Math.max(minS, view.scale * factor));
    const ratio = next / view.scale;
    view.tx = cx - (cx - view.tx) * ratio;
    view.ty = cy - (cy - view.ty) * ratio;
    view.scale = next;
    view.mode = Math.abs(next - 1) < 0.01 ? '100'
              : Math.abs(next - view.baseFit) < 0.01 ? 'fit'
              : '';
    apply();
  };

  stage.addEventListener('wheel', (e) => {
    // In crop mode the view is frozen — a stray wheel shouldn't move the
    // image out from under the crop rectangle.
    if (state._cropActive) return;
    e.preventDefault();
    // Trackpad pinches arrive as wheel with ctrlKey; deltaY is finer-grained.
    const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
    zoomAt(e.clientX, e.clientY, factor);
  }, { passive: false });

  // Drag-to-pan. Only engage on primary button; secondary click still opens
  // a context menu so users can save the image via the browser.
  let dragging = false, lastX = 0, lastY = 0;
  stage.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    // Crop overlay intercepts its own pointer events; pan-drag is disabled.
    if (state._cropActive) return;
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    stage.setPointerCapture(e.pointerId);
    stage.classList.add('is-dragging');
  });
  stage.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    view.tx += e.clientX - lastX;
    view.ty += e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    view.mode = '';
    apply();
  });
  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    try { stage.releasePointerCapture(e.pointerId); } catch {}
    stage.classList.remove('is-dragging');
  };
  stage.addEventListener('pointerup', endDrag);
  stage.addEventListener('pointercancel', endDrag);

  // Double-click toggles Fit ↔ 100% centered on the cursor so a user can
  // zoom into detail without reaching for the wheel.
  stage.addEventListener('dblclick', (e) => {
    if (state._cropActive) return;
    const goingToFit = view.mode !== 'fit';
    if (goingToFit) resetTo('fit');
    else zoomAt(e.clientX, e.clientY, 1 / view.scale);
  });

  root.querySelector('.studio-image-controls')?.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    if (btn.dataset.crop === 'enter')  return _imageCropEnter(root, stage, img, view);
    if (btn.dataset.crop === 'apply')  return _imageCropApply(root, stage, img, view);
    if (btn.dataset.crop === 'cancel') return _imageCropExit(root);
    if (btn.dataset.ai === 'inpaint')  return _imageInpaintEnter(root, stage, img, view);
    if (btn.dataset.ai)                return _imageAiAction(root, stage, img, view, btn);
    // All other controls are inert while cropping so zoom/rotate can't
    // desync the crop rect from the underlying image.
    if (state._cropActive) return;
    if (btn.dataset.zoom === 'in')  zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1.25);
    if (btn.dataset.zoom === 'out') zoomAt(window.innerWidth / 2, window.innerHeight / 2, 0.8);
    if (btn.dataset.fit)            resetTo(btn.dataset.fit);
    if (btn.dataset.rotate) {
      view.rot = (view.rot + Number(btn.dataset.rotate)) % 360;
      // Rotation changes the effective bounds, so recompute baseFit and
      // snap back to Fit mode — otherwise the image may escape the stage.
      view.baseFit = computeFit();
      resetTo('fit');
    }
  });

  root.addEventListener('keydown', (e) => {
    if (state._cropActive) {
      if (e.key === 'Enter')  { e.preventDefault(); _imageCropApply(root, stage, img, view); return; }
      if (e.key === 'Escape') { e.preventDefault(); _imageCropExit(root); return; }
      return; // swallow other shortcuts in crop mode
    }
    if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomAt(window.innerWidth/2, window.innerHeight/2, 1.25); }
    else if (e.key === '-')             { e.preventDefault(); zoomAt(window.innerWidth/2, window.innerHeight/2, 0.8); }
    else if (e.key === '0')             { e.preventDefault(); resetTo('fit'); }
    else if (e.key === '1')             { e.preventDefault(); resetTo('100'); }
  });
  root.focus({ preventScroll: true });

  // Viewport resizes (orb sidebar toggle, window resize) change baseFit.
  // Recompute and — if the user hadn't panned away — re-apply Fit.
  const ro = new ResizeObserver(() => {
    const prevMode = view.mode;
    view.baseFit = computeFit();
    if (prevMode === 'fit')  resetTo('fit');
    else if (prevMode === 'fill') resetTo('fill');
  });
  ro.observe(stage);
  state._imageViewerDispose = () => ro.disconnect();
}

// ---------------------------------------------------------------------------
// Image crop mode — overlay rectangle with 8 resize handles, apply uploads
// a rasterized PNG of the cropped region (honoring current rotation).
// ---------------------------------------------------------------------------
// Rect is stored in stage-local pixels so we can position the overlay
// div directly from x/y/w/h. On Apply we convert to natural-image pixels
// using the same transform the live view uses.
const _CROP_MIN_PX = 16;

function _imageCropEnter(root, stage, img, view) {
  if (state._cropActive) return;
  if (!img.naturalWidth) { showToast('Image still loading', 'warning'); return; }
  state._cropActive = true;

  // Seed the rect to 80% of the stage (not of the image) so a user
  // immediately sees the overlay regardless of current zoom level.
  const sr = stage.getBoundingClientRect();
  const w = sr.width * 0.8, h = sr.height * 0.8;
  const x = (sr.width - w) / 2, y = (sr.height - h) / 2;
  state._cropRect = { x, y, w, h };

  root.classList.add('is-cropping');
  const overlay = root.querySelector('#studio-crop-overlay');
  overlay?.classList.remove('hidden');
  overlay?.setAttribute('aria-hidden', 'false');

  // Hide pan/zoom/rotate controls + hint; show Apply/Cancel.
  root.querySelectorAll('[data-zoom], [data-fit], [data-rotate], [data-crop="enter"], [data-ai], #studio-image-pct, .studio-image-sep, #studio-image-hint')
    .forEach(el => el.classList.add('hidden'));
  root.querySelectorAll('[data-crop="apply"], [data-crop="cancel"]').forEach(el => el.classList.remove('hidden'));

  _wireCropOverlay(root, stage);
  _renderCropRect(root);
}

function _imageCropExit(root) {
  state._cropActive = false;
  state._cropRect = null;
  root.classList.remove('is-cropping');
  const overlay = root.querySelector('#studio-crop-overlay');
  overlay?.classList.add('hidden');
  overlay?.setAttribute('aria-hidden', 'true');
  // Restore pan/zoom/rotate controls + hint; hide Apply/Cancel.
  root.querySelectorAll('[data-zoom], [data-fit], [data-rotate], [data-crop="enter"], [data-ai], #studio-image-pct, .studio-image-sep, #studio-image-hint')
    .forEach(el => el.classList.remove('hidden'));
  root.querySelectorAll('[data-crop="apply"], [data-crop="cancel"]').forEach(el => el.classList.add('hidden'));
  root.focus({ preventScroll: true });
}

function _renderCropRect(root) {
  const rect = state._cropRect;
  const rectEl = root.querySelector('#studio-crop-rect');
  if (!rect || !rectEl) return;
  rectEl.style.left   = `${rect.x}px`;
  rectEl.style.top    = `${rect.y}px`;
  rectEl.style.width  = `${rect.w}px`;
  rectEl.style.height = `${rect.h}px`;
  // Dim masks — four rectangles surrounding the crop window. Position them
  // off the rect's edges; the mask container is absolute-positioned to fill
  // the stage so these are stage-local.
  const masks = root.querySelectorAll('.studio-crop-mask');
  const sr = root.querySelector('#studio-image-stage').getBoundingClientRect();
  const mTop    = masks[0], mLeft = masks[1], mRight = masks[2], mBottom = masks[3];
  if (mTop)    { mTop.style.left    = '0'; mTop.style.top    = '0';               mTop.style.width    = `${sr.width}px`; mTop.style.height    = `${rect.y}px`; }
  if (mLeft)   { mLeft.style.left   = '0'; mLeft.style.top   = `${rect.y}px`;     mLeft.style.width   = `${rect.x}px`;   mLeft.style.height   = `${rect.h}px`; }
  if (mRight)  { mRight.style.left  = `${rect.x + rect.w}px`; mRight.style.top  = `${rect.y}px`; mRight.style.width  = `${sr.width - rect.x - rect.w}px`; mRight.style.height  = `${rect.h}px`; }
  if (mBottom) { mBottom.style.left = '0'; mBottom.style.top = `${rect.y + rect.h}px`; mBottom.style.width = `${sr.width}px`; mBottom.style.height = `${sr.height - rect.y - rect.h}px`; }
}

function _wireCropOverlay(root, stage) {
  const rectEl = root.querySelector('#studio-crop-rect');
  if (!rectEl || rectEl._cropWired) return;
  rectEl._cropWired = true;

  // Dragging the rect body translates the whole rect; dragging a handle
  // resizes the matching edge/corner. Both paths use the same pointer
  // capture pattern so we don't lose input if the cursor escapes the stage.
  const onPointer = (mode, handle) => (e) => {
    e.preventDefault();
    e.stopPropagation();
    const sr = stage.getBoundingClientRect();
    const start = { x: e.clientX, y: e.clientY, ...state._cropRect };
    const target = e.currentTarget;
    try { target.setPointerCapture(e.pointerId); } catch {}

    const onMove = (ev) => {
      const dx = ev.clientX - start.x;
      const dy = ev.clientY - start.y;
      let { x, y, w, h } = start;
      if (mode === 'move') {
        x = Math.max(0, Math.min(sr.width  - w, start.x + dx));
        y = Math.max(0, Math.min(sr.height - h, start.y + dy));
      } else {
        // Resize. Each direction letter in `handle` moves a single edge;
        // horizontal letters (w/e) move left/right, vertical (n/s) top/bottom.
        if (handle.includes('w')) { const nx = Math.max(0, Math.min(start.x + start.w - _CROP_MIN_PX, start.x + dx)); w = start.w + (start.x - nx); x = nx; }
        if (handle.includes('e')) { w = Math.max(_CROP_MIN_PX, Math.min(sr.width - start.x, start.w + dx)); }
        if (handle.includes('n')) { const ny = Math.max(0, Math.min(start.y + start.h - _CROP_MIN_PX, start.y + dy)); h = start.h + (start.y - ny); y = ny; }
        if (handle.includes('s')) { h = Math.max(_CROP_MIN_PX, Math.min(sr.height - start.y, start.h + dy)); }
      }
      state._cropRect = { x, y, w, h };
      _renderCropRect(root);
    };
    const onUp = (ev) => {
      try { target.releasePointerCapture(ev.pointerId); } catch {}
      target.removeEventListener('pointermove', onMove);
      target.removeEventListener('pointerup', onUp);
      target.removeEventListener('pointercancel', onUp);
    };
    target.addEventListener('pointermove', onMove);
    target.addEventListener('pointerup', onUp);
    target.addEventListener('pointercancel', onUp);
  };

  rectEl.addEventListener('pointerdown', (e) => {
    // Handle pointerdowns bubble; detect and route to resize handler.
    const handle = e.target.closest('[data-handle]');
    if (handle) onPointer('resize', handle.dataset.handle)(e);
    else onPointer('move')(e);
  });
}

async function _imageCropApply(root, stage, img, view) {
  const rect = state._cropRect;
  if (!rect) { _imageCropExit(root); return; }
  const artifactId = root.dataset.artifactId;
  if (!artifactId) { showToast('Missing artifact id', 'error'); return; }

  // Post-rotation dimensions — the image as the user sees it with current
  // rotation baked in. Even though view.rot can be negative or > 360, the
  // modulo below normalizes it for the axis-swap check.
  const rot = ((view.rot % 360) + 360) % 360;
  const rotSwaps = rot % 180 !== 0;
  const nw = img.naturalWidth, nh = img.naturalHeight;
  const effW = rotSwaps ? nh : nw;
  const effH = rotSwaps ? nw : nh;

  // Convert stage-local crop rect → post-rotation-image pixels using the
  // same transform the live view applies. scale/tx/ty are already baked
  // into the view state.
  const sr = stage.getBoundingClientRect();
  const imgLeftStage = sr.width  / 2 - (effW * view.scale) / 2 + view.tx;
  const imgTopStage  = sr.height / 2 - (effH * view.scale) / 2 + view.ty;

  let cx = (rect.x - imgLeftStage) / view.scale;
  let cy = (rect.y - imgTopStage)  / view.scale;
  let cw = rect.w / view.scale;
  let ch = rect.h / view.scale;
  // Clamp to image bounds — users may drag the rect outside the image
  // when the image is smaller than the stage.
  if (cx < 0)         { cw += cx; cx = 0; }
  if (cy < 0)         { ch += cy; cy = 0; }
  if (cx + cw > effW) cw = effW - cx;
  if (cy + ch > effH) ch = effH - cy;
  cw = Math.max(1, Math.round(cw));
  ch = Math.max(1, Math.round(ch));
  cx = Math.round(cx);
  cy = Math.round(cy);
  if (cw < _CROP_MIN_PX || ch < _CROP_MIN_PX) {
    showToast('Crop region is empty — drag the handles to select an area.', 'warning');
    return;
  }

  // Draw the rotated source into a canvas sized to the crop window. Using
  // a single drawImage call (not a two-stage pipeline) keeps the path short
  // and avoids an intermediate full-image canvas allocation.
  const canvas = document.createElement('canvas');
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext('2d');

  // Translate the canvas origin so that drawing the rotated image lands
  // the crop window at (0, 0). We rotate around the effective center, then
  // shift so that the crop origin (cx, cy) in post-rotation space maps to
  // the canvas origin.
  ctx.save();
  ctx.translate(-cx, -cy);
  ctx.translate(effW / 2, effH / 2);
  ctx.rotate(rot * Math.PI / 180);
  ctx.drawImage(img, -nw / 2, -nh / 2, nw, nh);
  ctx.restore();

  const btn = root.querySelector('[data-crop="apply"]');
  if (btn) { btn.disabled = true; btn.textContent = 'Applying…'; }
  try {
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('Failed to rasterize crop');
    const form = new FormData();
    form.append('file', blob, 'cropped.png');
    const resp = await fetch(`/api/artifacts/${artifactId}/upload`, { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `Upload failed (${resp.status})`);
    }
    await _imageReplaceAndRefit(root, stage, img, view, artifactId);
    showToast('Image cropped', 'success');
    _imageCropExit(root);
  } catch (e) {
    showToast(`Crop failed: ${e.message}`, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Apply'; }
  }
}

// ---------------------------------------------------------------------------
// Inpaint mode — mounts the shared mask editor (ui/scripts/mask-editor.js)
// inside the image stage. Pan/zoom/rotate controls hide during inpaint the
// same way they do during crop; Generate POSTs to the artifact-keyed
// inpaint route and opens the freshly painted sibling artifact on success.
// ---------------------------------------------------------------------------
async function _imageInpaintEnter(root, stage, img, view) {
  if (state._cropActive || state._inpaintActive) return;
  if (!img.naturalWidth) { showToast('Image still loading', 'warning'); return; }
  const artifactId = root.dataset.artifactId;
  if (!artifactId) { showToast('Missing artifact id', 'error'); return; }

  state._inpaintActive = true;
  root.classList.add('is-inpainting');

  // Hide pan/zoom/rotate/crop/AI buttons + the image element itself. The
  // mask editor takes over the stage and brings its own source image
  // (drawn into its `bg` canvas), so we don't want the pan/zoom <img>
  // showing through under the mask.
  root.querySelectorAll('[data-zoom], [data-fit], [data-rotate], [data-crop="enter"], [data-ai], #studio-image-pct, .studio-image-sep, #studio-image-hint')
    .forEach(el => el.classList.add('hidden'));

  // Mount the module as a child of the stage so it covers the stage's full
  // area (including the image element it hides). Stage already has
  // position:relative + overflow:hidden so the module's absolute inset:0
  // clips correctly.
  const { createMaskEditor } = await import('./mask-editor.js');
  state._inpaintMask = createMaskEditor({
    container: stage,
    sourceImg: img,
    variant: 'studio',
    showPromptStrip: true,
    initialPrompt: '',
    onCancel: () => _imageInpaintExit(root),
    onGenerate: (payload) => _imageInpaintGenerate(root, stage, img, view, artifactId, payload),
    generateLabel: 'Inpaint',
  });
  // Hide the raw <img> while the editor owns the stage — its bg canvas
  // carries the same pixels, and leaving <img> visible just doubles up.
  img.style.visibility = 'hidden';
}

function _imageInpaintExit(root) {
  if (!state._inpaintActive) return;
  state._inpaintActive = false;
  if (state._inpaintMask) {
    try { state._inpaintMask.destroy(); } catch { /* already torn down */ }
    state._inpaintMask = null;
  }
  root.classList.remove('is-inpainting');
  root.querySelectorAll('[data-zoom], [data-fit], [data-rotate], [data-crop="enter"], [data-ai], #studio-image-pct, .studio-image-sep, #studio-image-hint')
    .forEach(el => el.classList.remove('hidden'));
  const img = root.querySelector('#studio-image-el');
  if (img) img.style.visibility = '';
  root.focus({ preventScroll: true });
}

async function _imageInpaintGenerate(root, stage, img, view, artifactId, payload) {
  const mask = state._inpaintMask;
  if (!mask) return;
  mask.setBusy(true, 'Inpainting…');
  try {
    const resp = await fetch(`/api/artifacts/${artifactId}/inpaint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: payload.prompt,
        negative_prompt: payload.negativePrompt,
        mask_image: payload.maskBase64,
        mode: payload.mode,
        strength: payload.strength,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || data.error || `Failed (${resp.status})`);
    showToast(`Inpainted (${data.width}×${data.height})`, 'success');
    _imageInpaintExit(root);
    // Open the freshly inpainted sibling — keeps "opened from Library" if
    // that was the entry point so the back button still returns there.
    if (data.artifact_id) {
      const fromLibrary = state.openedFromLibrary;
      await openStudio(data.artifact_id, { fromLibrary });
    }
  } catch (e) {
    mask.setBusy(false);
    showToast(`Inpaint failed: ${e.message}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// AI image actions — bridge Studio to Augmentum's image postprocess pipeline.
// Both endpoints replace the artifact binary in place, so the refresh path
// is the same: swap the <img> src with a cache-buster, await the load, then
// recompute the view's baseFit against the new natural dimensions.
// ---------------------------------------------------------------------------
const _AI_ACTION_ROUTES = {
  'upscale':   { path: 'upscale',   body: { scale: 4 }, label: 'Upscaling…',   toast: 'Upscaled' },
  'remove-bg': { path: 'remove-bg', body: null,         label: 'Removing BG…', toast: 'Background removed' },
};

async function _imageAiAction(root, stage, img, view, btn) {
  const action = btn.dataset.ai;
  const route = _AI_ACTION_ROUTES[action];
  if (!route) return;
  const artifactId = root.dataset.artifactId;
  if (!artifactId) { showToast('Missing artifact id', 'error'); return; }

  // Disable the whole controls row — long-running AI ops shouldn't let the
  // user queue up a second action on stale bytes.
  const controls = root.querySelector('#studio-image-controls');
  controls?.querySelectorAll('button').forEach(b => { b.disabled = true; });
  const originalLabel = btn.innerHTML;
  btn.textContent = route.label;

  try {
    const resp = await fetch(`/api/artifacts/${artifactId}/${route.path}`, {
      method: 'POST',
      headers: route.body ? { 'Content-Type': 'application/json' } : {},
      body: route.body ? JSON.stringify(route.body) : undefined,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `Request failed (${resp.status})`);
    }
    const data = await resp.json();
    await _imageReplaceAndRefit(root, stage, img, view, artifactId);
    showToast(`${route.toast} (${data.width}×${data.height})`, 'success');
  } catch (e) {
    showToast(`${action === 'upscale' ? 'Upscale' : 'Background removal'} failed: ${e.message}`, 'error');
  } finally {
    controls?.querySelectorAll('button').forEach(b => { b.disabled = false; });
    btn.innerHTML = originalLabel;
  }
}

// Swap the img src with a cache-buster, wait for the load event, then reset
// the view to Fit against the new natural dimensions. Shared by the crop
// and AI action paths since both replace the artifact binary in place.
async function _imageReplaceAndRefit(root, stage, img, view, artifactId) {
  await new Promise((resolve, reject) => {
    const onLoad = () => { img.removeEventListener('error', onErr); resolve(); };
    const onErr  = () => { img.removeEventListener('load', onLoad); reject(new Error('Failed to reload image')); };
    img.addEventListener('load', onLoad, { once: true });
    img.addEventListener('error', onErr, { once: true });
    img.src = `/api/artifacts/${artifactId}/download?t=${Date.now()}`;
  });
  view.rot = 0; view.tx = 0; view.ty = 0;
  const newBase = Math.min(
    stage.clientWidth  / (img.naturalWidth  || 1),
    stage.clientHeight / (img.naturalHeight || 1),
    1,
  );
  view.baseFit = newBase;
  view.scale = newBase;
  view.mode = 'fit';
  img.style.transform = `translate(0px, 0px) rotate(0deg) scale(${newBase})`;
  const pctEl = root.querySelector('#studio-image-pct');
  if (pctEl) pctEl.textContent = `${Math.round(newBase * 100)}%`;
}

// ---------------------------------------------------------------------------
// Audio viewer — native HTML5 audio, metadata sidebar, transcribe hook.
// Intentionally minimal: browser's own transport controls cover play/pause/
// seek/volume perfectly well; the chrome around it is what makes this feel
// like a Studio viewer rather than a raw file download.
// ---------------------------------------------------------------------------
function openAudioViewer(artifactId, filename, ext) {
  _hideLoading();
  dom.saveBtn.disabled = true;
  const src = `/api/artifacts/${escapeHtml(artifactId)}/download`;
  dom.body.innerHTML = `
    <div class="studio-media-viewer studio-media-viewer--audio" data-artifact-id="${escapeHtml(artifactId)}">
      <div class="studio-media-stage">
        <div class="studio-media-audio-card">
          <div class="studio-media-audio-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="72" height="72" fill="none" stroke="currentColor" stroke-width="1.4">
              <path d="M9 18V5l12-2v13"/>
              <circle cx="6" cy="18" r="3"/>
              <circle cx="18" cy="16" r="3"/>
            </svg>
          </div>
          <div class="studio-media-audio-title">${escapeHtml(filename || 'Audio')}</div>
          <div class="studio-media-audio-ext">${escapeHtml(ext.toUpperCase())}</div>
          <audio class="studio-media-audio" id="studio-media-audio" src="${src}" controls preload="metadata"></audio>
        </div>
      </div>
      ${_renderMediaSidebar(artifactId, ext)}
    </div>
  `;
  _wireMediaViewer(document.getElementById('studio-media-audio'), artifactId, ext);
}

// ---------------------------------------------------------------------------
// Video viewer — native <video>. Range requests on /download make scrubbing
// work out of the box; no custom player needed.
// ---------------------------------------------------------------------------
function openVideoViewer(artifactId, filename, ext) {
  _hideLoading();
  dom.saveBtn.disabled = true;
  const src = `/api/artifacts/${escapeHtml(artifactId)}/download`;
  dom.body.innerHTML = `
    <div class="studio-media-viewer studio-media-viewer--video" data-artifact-id="${escapeHtml(artifactId)}">
      <div class="studio-media-stage">
        <video class="studio-media-video" id="studio-media-video" src="${src}" controls preload="metadata" playsinline></video>
        <div class="studio-media-caption">${escapeHtml(filename || 'Video')}</div>
      </div>
      ${_renderMediaSidebar(artifactId, ext)}
    </div>
  `;
  _wireMediaViewer(document.getElementById('studio-media-video'), artifactId, ext);
}

// Shared metadata/action sidebar for audio + video viewers. Populated
// lazily as the media element loads metadata events — duration is the only
// field the browser hands us directly, so we compute it client-side and
// surface artifact-level info (format, size) from the artifact metadata.
function _renderMediaSidebar(artifactId, ext) {
  return `
    <aside class="studio-media-sidebar">
      <div class="studio-media-sidebar-section">
        <h3>File</h3>
        <dl class="studio-media-meta">
          <dt>Format</dt><dd>${escapeHtml((ext || '').toUpperCase())}</dd>
          <dt>Duration</dt><dd id="studio-media-duration">—</dd>
          <dt>Dimensions</dt><dd id="studio-media-dims">—</dd>
        </dl>
      </div>
      <div class="studio-media-sidebar-section">
        <h3>Actions</h3>
        <button class="studio-media-action" id="studio-media-transcribe" title="Transcribe speech to text">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 5h12"/><path d="M3 12h10"/><path d="M3 19h8"/><path d="M15 14l5 5M20 14l-5 5"/></svg>
          Transcribe
        </button>
        <button class="studio-media-action" id="studio-media-ask-ai" title="Open a chat with this file as context">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3a9 9 0 1 0 9 9v-2"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/><path d="M16 3l5 5M21 3l-5 5"/></svg>
          Ask AI
        </button>
        <a class="studio-media-action" id="studio-media-download" href="/api/artifacts/${escapeHtml(artifactId)}/download" download title="Download">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v14"/><path d="M7 12l5 5 5-5"/><path d="M5 21h14"/></svg>
          Download
        </a>
      </div>
      <div class="studio-media-sidebar-section" id="studio-media-transcript-wrap" style="display:none">
        <h3>Transcript</h3>
        <div class="studio-media-transcript" id="studio-media-transcript"></div>
      </div>
    </aside>
  `;
}

function _wireMediaViewer(media, artifactId, ext) {
  if (!media) return;
  const durEl = document.getElementById('studio-media-duration');
  const dimEl = document.getElementById('studio-media-dims');
  const isVideo = media.tagName === 'VIDEO';
  // Hide Dimensions row for audio — it's always "—" which is meaningless.
  if (!isVideo && dimEl?.parentElement) {
    const dt = dimEl.previousElementSibling;
    if (dt?.tagName === 'DT') dt.style.display = 'none';
    dimEl.style.display = 'none';
  }
  media.addEventListener('loadedmetadata', () => {
    if (durEl) durEl.textContent = _formatDuration(media.duration);
    if (isVideo && dimEl) dimEl.textContent = `${media.videoWidth} × ${media.videoHeight}`;
  });
  document.getElementById('studio-media-transcribe')?.addEventListener('click', () => _mediaTranscribe(artifactId, ext));
  document.getElementById('studio-media-ask-ai')?.addEventListener('click', () => _mediaAskAi(artifactId, ext));
}

// HH:MM:SS when ≥ 1 hour, else M:SS. Audio files commonly run under an hour
// so we leave the leading zero off minutes to match YouTube/podcast conventions.
function _formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, '0');
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${ss}`;
  return `${m}:${ss}`;
}

async function _mediaTranscribe(artifactId, ext) {
  // Stubbed here; route is wired in a follow-up task. Surfacing the control
  // in this pass so wiring it up is a one-line swap rather than a viewer
  // rewrite.
  const btn = document.getElementById('studio-media-transcribe');
  if (!btn) return;
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = 'Transcribing…';
  try {
    const resp = await fetch(`/api/artifacts/${artifactId}/transcribe`, { method: 'POST' });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `Failed (${resp.status})`);
    }
    const data = await resp.json();
    const wrap = document.getElementById('studio-media-transcript-wrap');
    const tEl  = document.getElementById('studio-media-transcript');
    if (wrap) wrap.style.display = '';
    if (tEl)  tEl.textContent = data.transcript || '(no speech detected)';
    showToast(`Transcript saved${data.artifact_id ? ' as sibling artifact' : ''}`, 'success');
  } catch (e) {
    showToast(`Transcribe failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = original;
  }
}

async function _mediaAskAi(artifactId, ext) {
  // Hand off to the universal Ask-AI flow with the media format so app.js
  // routes through /transcribe instead of downloading the raw audio/video.
  _artifactAskAi(artifactId, ext);
}

// Convert targets per source kind — mirrors _CONVERT_MATRIX in
// augmentum/proxy/artifact_routes.py. Kept in sync manually; the backend
// is the source of truth so a mismatch just produces a 400 at click time.
const _CONVERT_TARGETS = {
  png:  ['jpg', 'webp'],
  jpg:  ['png', 'webp'],
  jpeg: ['png', 'webp'],
  webp: ['png', 'jpg'],
  '*document': ['pdf', 'docx'],
};

function _convertTargetsForCurrent() {
  // Structured source types (document/presentation/spreadsheet) use their
  // source_json type as the lookup key. Everything else keys off the
  // artifact format / extension.
  if (state.sourceType === 'document') return _CONVERT_TARGETS['*document'] || [];
  const info = state.artifactInfo || {};
  const fmt = (info.format || '').toLowerCase();
  if (fmt in _CONVERT_TARGETS) return _CONVERT_TARGETS[fmt];
  const name = (info.filename || '').toLowerCase();
  const ext = name.includes('.') ? name.split('.').pop() : '';
  return _CONVERT_TARGETS[ext] || [];
}

function _updateConvertMenu() {
  if (!dom.convertWrap || !dom.convertMenu) return;
  const targets = _convertTargetsForCurrent();
  if (!targets.length) {
    dom.convertWrap.style.display = 'none';
    return;
  }
  dom.convertWrap.style.display = '';
  dom.convertMenu.innerHTML = targets.map(t =>
    `<button class="studio-convert-item" data-convert-to="${t}" role="menuitem">
      <span class="studio-convert-item-icon">→</span>
      <span class="studio-convert-item-label">${t.toUpperCase()}</span>
    </button>`,
  ).join('');
}

async function _convertArtifact(target) {
  const artifactId = state.artifactId;
  if (!artifactId || !target) return;
  const toastId = showToast(`Converting to ${target.toUpperCase()}…`, 'loading');
  try {
    const resp = await fetch(`/api/artifacts/${artifactId}/convert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ to: target }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || data.error || `Failed (${resp.status})`);
    updateToast(toastId, `Converted to ${target.toUpperCase()}`, 'success');
    // Open the newly created sibling artifact — stays in Studio, just swaps
    // the artifact being viewed. Respects "openedFromLibrary" so closing
    // still returns to the library if that was the entry point.
    if (data.artifact_id) {
      const fromLibrary = state.openedFromLibrary;
      await openStudio(data.artifact_id, { fromLibrary });
    }
  } catch (e) {
    updateToast(toastId, `Convert failed: ${e.message}`, 'error');
  }
}

// Extensions the document store will ingest, plus the image formats that
// attach inline as vision input. Mirrors ALLOWED_EXTENSIONS in
// augmentum/proxy/document_routes.py — keep the two in sync. Used to gate the
// "Ask AI about this file" affordance up front so the user never clicks it
// only to hit a server-side "unsupported file type" error.
const _ASK_AI_EXTENSIONS = new Set([
  'txt', 'md', 'markdown', 'csv', 'json', 'log', 'pdf', 'docx', 'pptx', 'xlsx',
  'html', 'htm', 'py', 'js', 'ts', 'yaml', 'yml', 'toml', 'xml', 'rst',
  'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg',
]);

// Source types whose Studio export is NOT something the chat can ingest, so
// the "Ask AI about this file" affordance must stay hidden (clicking it would
// only hit a server-side "unsupported file type" error):
//   - application  → exports .zip app-bundle (binary archive, not ingestible)
// eBooks (.epub) used to be excluded too, but the attach handler now derives
// their chapter text via /epub-text, so they're referenceable again. Audio and
// video aren't gated here — they use the media viewer's own Ask-AI button,
// which derives a transcript via /transcribe.
const _ASK_AI_EXCLUDED_SOURCE_TYPES = new Set(['application']);

// True when the currently-open artifact can be handed to the chat as an
// attachment. Source-backed artifacts authored in Studio download in an
// ingestible container EXCEPT the excluded types above; eBooks derive text;
// imported files are gated purely on extension.
function _artifactSupportsAskAi() {
  const info = state.artifactInfo || {};
  const ext = (info.filename || '').split('.').pop()?.toLowerCase() || '';
  const fmt = String(info.format || state.format || '').toLowerCase();
  // eBooks export as .epub — not directly ingestible, but the attach handler
  // derives chapter text via /epub-text, so they ARE referenceable.
  if (ext === 'epub' || fmt === 'epub' || state.sourceType === 'ebook') return true;
  if (state.source && state.sourceType
      && !_ASK_AI_EXCLUDED_SOURCE_TYPES.has(state.sourceType)) return true;
  return _ASK_AI_EXTENSIONS.has(ext) || _ASK_AI_EXTENSIONS.has(fmt);
}

// Source types whose editors expose a theme picker (loadThemes is wired in
// their open* functions). Documents/slides/sheets/charts re-render a themed
// preview; ebooks carry the theme through to the EPUB export. Everything
// else — PDFs, images, backend-preview formats — has no themeable output, so
// we hide the button rather than let it toggle an empty popover.
const _THEMEABLE_SOURCE_TYPES = new Set(['document', 'presentation', 'spreadsheet', 'chart', 'ebook']);

// Universal Ask-AI — dispatches the app-level event so the chat module
// (in app.js) can fetch the artifact, ingest it as a document / image
// attachment, and route the user into a fresh session. Must read the
// filename BEFORE closeStudio since that clears state.artifactInfo.
function _artifactAskAi(artifactId, formatOverride) {
  if (!artifactId) return;
  const filename = state.artifactInfo?.display_name
               || state.artifactInfo?.filename
               || '';
  // Format drives the attach strategy in app.js: epub → /epub-text, audio/
  // video → /transcribe, everything else → binary download. The media viewer
  // passes its ext explicitly; otherwise resolve from artifact metadata.
  const format = String(
    formatOverride
    || state.artifactInfo?.format
    || state.format
    || (state.sourceType === 'ebook' ? 'epub' : ''),
  ).toLowerCase();
  document.dispatchEvent(new CustomEvent('augmentum:ask-ai-about-artifact', {
    detail: { artifactId, filename, format },
  }));
  // Close Studio so the chat surface is visible when the attachment toast
  // lands. closeStudio respects unsaved changes — don't confirm twice.
  if (state.open) closeStudio();
}

function openArtifactOverview(artifactId, filename) {
  _hideLoading();
  dom.askBar.style.display = 'none';
  dom.saveBtn.disabled = true;

  const safeId = escapeHtml(artifactId);
  const title = escapeHtml(filename || state.artifactInfo?.display_name || 'Artifact');
  dom.body.innerHTML = `
    <div class="studio-overview-frame">
      <div class="studio-overview-header">
        <div class="studio-overview-copy">
          <span class="studio-overview-kicker">Overview</span>
          <span class="studio-overview-name">${title}</span>
        </div>
        <button type="button" class="studio-overview-edit" data-studio-session-action="edit">Open editor</button>
      </div>
      <iframe src="/api/artifacts/${safeId}/preview?v=${Date.now()}"
              class="studio-overview-iframe"
              sandbox="allow-scripts allow-forms allow-modals allow-popups"
              loading="lazy"
              title="${title}"></iframe>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// HTML Preview (iframe with source toggle)
// ---------------------------------------------------------------------------
function openHtmlPreview(artifactId, filename) {
  _hideLoading();
  dom.saveBtn.disabled = true;

  dom.body.innerHTML = `
    <div style="display:flex;flex-direction:column;flex:1;overflow:hidden">
      <iframe src="/api/artifacts/${escapeHtml(artifactId)}/preview?v=${Date.now()}"
              sandbox="allow-scripts allow-forms allow-modals allow-popups"
              style="flex:1;border:none;background:#fff;border-radius:var(--radius-sm);margin:var(--space-md)"
              loading="lazy"></iframe>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Backend Preview (docx/pptx/xlsx/epub/json — server renders to HTML)
// ---------------------------------------------------------------------------
function openBackendPreview(artifactId, filename) {
  _hideLoading();
  // Tear down any previous read-aloud controls (stops active playback).
  if (_activeReaderControls) { try { _activeReaderControls.destroy(); } catch { /* noop */ } _activeReaderControls = null; }
  dom.saveBtn.disabled = true;

  const isEpub = /\.epub$/i.test(filename || '');
  dom.body.innerHTML = `
    <div style="display:flex;flex-direction:column;flex:1;overflow:hidden">
      <div class="studio-backend-reader" style="padding:8px var(--space-md) 0;display:none"></div>
      <iframe src="/api/artifacts/${escapeHtml(artifactId)}/preview?v=${Date.now()}"
              sandbox="allow-scripts allow-forms allow-modals allow-popups"
              style="flex:1;border:none;background:var(--bg);border-radius:var(--radius-sm);margin:var(--space-md)"
              loading="lazy"></iframe>
    </div>
  `;
  if (isEpub) {
    const host = dom.body.querySelector('.studio-backend-reader');
    if (host) {
      const textUrl = `/api/artifacts/${encodeURIComponent(artifactId)}/epub-text`;
      const narrationUrl = `/api/artifacts/${encodeURIComponent(artifactId)}/narration`;
      import('./epub-reader-controls.js')
        .then(m => {
          const ctl = m.createReaderControls({ textUrl, narrationUrl });
          host.appendChild(ctl.el);
          host.style.display = '';
          _activeReaderControls = ctl;
        })
        .catch(e => console.debug('[studio] reader controls load failed', e));
    }
  }
}

// ---------------------------------------------------------------------------
// CSV Viewer (parse + table)
// ---------------------------------------------------------------------------
function _parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else { inQuotes = false; }
      } else {
        field += ch;
      }
    } else {
      if (ch === '"') inQuotes = true;
      else if (ch === ',') { row.push(field); field = ''; }
      else if (ch === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
      else if (ch === '\r') { /* skip */ }
      else field += ch;
    }
  }
  if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
  return rows.filter(r => r.length > 0 && !(r.length === 1 && r[0] === ''));
}

// Escape a cell value for CSV: only quote when necessary (comma, quote,
// newline) to keep diffs small and the round-trip clean.
function _csvEscapeCell(v) {
  const s = v == null ? '' : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function _rowsToCsv(rows) {
  return rows.map(r => r.map(_csvEscapeCell).join(',')).join('\n');
}

// CSV editor: contenteditable cells, click-header-to-sort (asc/desc/none),
// add row/column, and save-back through the artifact binary-upload endpoint
// so the edits land in storage as a real CSV file. Sort state is presentational
// only — the underlying `_state.rows` array preserves the edit order, so
// users can switch sort direction without losing their changes.
async function openCsvViewer(artifactId) {
  _hideLoading();
  dom.saveBtn.disabled = false;
  dom.askBar.style.display = 'none';

  let text;
  try {
    const resp = await fetch(`/api/artifacts/${artifactId}/download`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    text = await resp.text();
  } catch (e) {
    dom.body.innerHTML = `<div class="studio-legacy-notice"><p>Could not load CSV: ${escapeHtml(e.message)}</p></div>`;
    return;
  }

  let rows;
  try { rows = _parseCsv(text); }
  catch (e) {
    dom.body.innerHTML = `<div class="studio-legacy-notice"><p>Could not parse CSV: ${escapeHtml(e.message)}</p></div>`;
    return;
  }

  if (rows.length === 0) rows = [['Column 1']];
  // Pad every row to the widest so the grid is rectangular — makes editing
  // behaviour predictable and sort indices consistent.
  const width = Math.max(1, ...rows.map(r => r.length));
  for (const r of rows) while (r.length < width) r.push('');

  state.sourceType = 'csv';
  state._csv = {
    artifactId,
    headers: rows[0],
    rows: rows.slice(1),
    // Sort column index + dir ('asc' | 'desc' | null). Stored presentationally;
    // actual row order in _csv.rows stays in insertion order.
    sortCol: -1,
    sortDir: null,
  };

  dom.body.innerHTML = `
    <div class="studio-csv-wrap">
      <div class="studio-csv-toolbar">
        <button class="studio-csv-btn" data-csv-action="add-row">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          Add row
        </button>
        <button class="studio-csv-btn" data-csv-action="add-col">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          Add column
        </button>
        <span class="studio-csv-sep"></span>
        <span class="studio-csv-stats" id="studio-csv-stats"></span>
      </div>
      <div class="studio-csv-scroll" id="studio-csv-scroll"></div>
    </div>
  `;
  _renderCsvTable();
  _wireCsvHandlers();
  _registerCsvFindProvider();
}

async function _registerCsvFindProvider() {
  const mod = await import('./studio-find.js');
  mod.registerProvider('csv', {
    getMatches(re) {
      const s = state._csv;
      if (!s) return [];
      const out = [];
      const push = (row, col, text) => {
        let m;
        while ((m = re.exec(text)) !== null) {
          if (m[0].length === 0) { re.lastIndex++; continue; }
          out.push({ row, col, start: m.index, end: m.index + m[0].length });
        }
      };
      s.headers.forEach((h, c) => push(-1, c, String(h ?? '')));
      s.rows.forEach((r, ri) => r.forEach((v, c) => push(ri, c, String(v ?? ''))));
      return out;
    },
    focusMatch(match) {
      const sel = match.row === -1
        ? `.studio-csv-table [data-header="${match.col}"]`
        : `.studio-csv-table td.studio-csv-cell[data-row="${match.row}"][data-col="${match.col}"]`;
      const el = document.querySelector(sel);
      if (el instanceof HTMLElement) {
        el.focus();
        el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    },
    applyReplace(match, replacement) {
      const s = state._csv;
      if (!s) return;
      if (match.row === -1) {
        const v = String(s.headers[match.col] ?? '');
        s.headers[match.col] = v.slice(0, match.start) + replacement + v.slice(match.end);
      } else {
        if (!s.rows[match.row]) s.rows[match.row] = [];
        const v = String(s.rows[match.row][match.col] ?? '');
        s.rows[match.row][match.col] = v.slice(0, match.start) + replacement + v.slice(match.end);
      }
      markDirty();
      _renderCsvTable();
    },
  });
}

function _sortedCsvRowIndices() {
  const s = state._csv;
  if (!s || s.sortCol < 0 || !s.sortDir) return s ? s.rows.map((_, i) => i) : [];
  const col = s.sortCol;
  const dir = s.sortDir === 'asc' ? 1 : -1;
  // Detect numeric column — only when EVERY non-empty value in that column
  // parses as a finite number. One stray string demotes the whole column
  // to lexicographic sort so dates/codes don't get mangled.
  const isNumeric = s.rows.every(r => {
    const v = (r[col] ?? '').trim();
    return v === '' || Number.isFinite(Number(v));
  });
  const idx = s.rows.map((_, i) => i);
  idx.sort((a, b) => {
    const va = s.rows[a][col] ?? '';
    const vb = s.rows[b][col] ?? '';
    if (isNumeric) {
      const na = va === '' ? Infinity * dir : Number(va);
      const nb = vb === '' ? Infinity * dir : Number(vb);
      return (na - nb) * dir;
    }
    return String(va).localeCompare(String(vb)) * dir;
  });
  return idx;
}

function _renderCsvTable() {
  const s = state._csv;
  const host = document.getElementById('studio-csv-scroll');
  if (!s || !host) return;
  const order = _sortedCsvRowIndices();

  const arrow = (ci) => {
    if (s.sortCol !== ci || !s.sortDir) return '';
    return s.sortDir === 'asc'
      ? '<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M7 14l5-5 5 5z"/></svg>'
      : '<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M7 10l5 5 5-5z"/></svg>';
  };

  let html = '<table class="studio-csv-table"><thead><tr><th class="studio-csv-rownum"></th>';
  s.headers.forEach((h, ci) => {
    html += `<th class="studio-csv-header${s.sortCol === ci && s.sortDir ? ' is-sorted' : ''}" data-col="${ci}">
      <span class="studio-csv-header-label" contenteditable="true" data-header="${ci}">${escapeHtml(h || '')}</span>
      <button class="studio-csv-sort-btn" data-sort="${ci}" aria-label="Sort column">${arrow(ci)}</button>
    </th>`;
  });
  html += '</tr></thead><tbody>';
  order.forEach((ri, visibleIdx) => {
    const row = s.rows[ri];
    html += `<tr data-row="${ri}"><td class="studio-csv-rownum">${visibleIdx + 1}</td>`;
    for (let ci = 0; ci < s.headers.length; ci++) {
      html += `<td class="studio-csv-cell" contenteditable="true" data-row="${ri}" data-col="${ci}">${escapeHtml(row[ci] || '')}</td>`;
    }
    html += '</tr>';
  });
  html += '</tbody></table>';
  host.innerHTML = html;

  const statsEl = document.getElementById('studio-csv-stats');
  if (statsEl) statsEl.textContent = `${s.rows.length} row${s.rows.length === 1 ? '' : 's'} · ${s.headers.length} column${s.headers.length === 1 ? '' : 's'}`;
}

function _wireCsvHandlers() {
  const s = state._csv;
  if (!s) return;

  dom.body.addEventListener('click', (e) => {
    const sortBtn = e.target.closest('[data-sort]');
    if (sortBtn) {
      const ci = Number(sortBtn.dataset.sort);
      // Tri-state cycle: none → asc → desc → none. Keeps the original row
      // order reachable without having to re-open the file.
      if (s.sortCol !== ci) { s.sortCol = ci; s.sortDir = 'asc'; }
      else if (s.sortDir === 'asc') s.sortDir = 'desc';
      else if (s.sortDir === 'desc') { s.sortCol = -1; s.sortDir = null; }
      else s.sortDir = 'asc';
      _renderCsvTable();
      return;
    }
    const action = e.target.closest('[data-csv-action]')?.dataset.csvAction;
    if (action === 'add-row') {
      s.rows.push(new Array(s.headers.length).fill(''));
      markDirty();
      _renderCsvTable();
    } else if (action === 'add-col') {
      s.headers.push(`Column ${s.headers.length + 1}`);
      s.rows.forEach(r => r.push(''));
      markDirty();
      _renderCsvTable();
    }
  });

  // Stop contenteditable inputs from bubbling into the sort-button handler.
  // We commit on blur to avoid a dirty-flag storm on every keystroke.
  dom.body.addEventListener('blur', (e) => {
    const cell = e.target.closest('.studio-csv-cell');
    if (cell) {
      const ri = Number(cell.dataset.row);
      const ci = Number(cell.dataset.col);
      const v = cell.textContent;
      if (s.rows[ri] && s.rows[ri][ci] !== v) {
        s.rows[ri][ci] = v;
        markDirty();
      }
      return;
    }
    const hdr = e.target.closest('[data-header]');
    if (hdr) {
      const ci = Number(hdr.dataset.header);
      const v = hdr.textContent;
      if (s.headers[ci] !== v) {
        s.headers[ci] = v;
        markDirty();
      }
    }
  }, true);

  // Enter commits without adding a newline inside the cell — feels right for
  // tabular data and matches Excel/Numbers.
  dom.body.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.closest('[contenteditable]')) {
      e.preventDefault();
      e.target.blur();
    }
  });
}

// Called by saveArtifact when sourceType === 'csv'. Serializes the grid to
// CSV text and replaces the artifact's binary through the /upload route.
async function _saveCsvArtifact() {
  const s = state._csv;
  if (!s) return false;
  // Flush any focused contenteditable so its pending edit is committed.
  if (document.activeElement?.blur) document.activeElement.blur();
  const csv = _rowsToCsv([s.headers, ...s.rows]);
  const form = new FormData();
  form.append('file', new Blob([csv], { type: 'text/csv' }), 'edited.csv');
  const resp = await fetch(`/api/artifacts/${s.artifactId}/upload`, { method: 'POST', body: form });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    showToast(err.detail || err.error || 'Save failed', 'error');
    return false;
  }
  const data = await resp.json();
  _updateDirtyState(false);
  showToast(`Saved (${(data.size_bytes / 1024).toFixed(1)} KB)`, 'success');
  return true;
}

// ---------------------------------------------------------------------------
// Generic File Info (unknown formats — show metadata + download)
// ---------------------------------------------------------------------------
function showFileInfo(data) {
  _hideLoading();
  dom.loading.classList.add('hidden');
  dom.saveBtn.disabled = true;

  const fmt = (data.format || '').toUpperCase();
  const size = data.size_bytes ? `${(data.size_bytes / 1024).toFixed(1)} KB` : '';
  const ext = (data.filename || '').split('.').pop()?.toLowerCase() || '';
  const icon = _fileTypeIcon(ext || data.format);

  dom.body.innerHTML = `
    <div class="studio-legacy-notice">
      <div class="studio-fallback-icon">${icon}</div>
      <h3>${escapeHtml(data.display_name || data.filename || 'File')}</h3>
      <div class="studio-fallback-meta">
        ${fmt ? `<span>${fmt}</span>` : ''}
        ${size ? `<span>${size}</span>` : ''}
      </div>
      <p>Direct editing isn't available for this file type yet. You can download it or ask the AI to work with it.</p>
      <a class="browse-action-pill" href="/api/artifacts/${escapeHtml(data.id)}/download"
         target="_blank" rel="noopener noreferrer" style="display:inline-flex">
        Download
      </a>
    </div>
  `;
}

function _fileTypeIcon(ext) {
  const icons = {
    pdf: '\uD83D\uDCC4', docx: '\uD83D\uDCC4', doc: '\uD83D\uDCC4',
    pptx: '\uD83D\uDCCA', ppt: '\uD83D\uDCCA',
    xlsx: '\uD83D\uDCCA', xls: '\uD83D\uDCCA', csv: '\uD83D\uDCCA',
    png: '\uD83D\uDDBC\uFE0F', jpg: '\uD83D\uDDBC\uFE0F', jpeg: '\uD83D\uDDBC\uFE0F',
    gif: '\uD83D\uDDBC\uFE0F', svg: '\uD83D\uDDBC\uFE0F', webp: '\uD83D\uDDBC\uFE0F',
    mp3: '\uD83C\uDFB5', wav: '\uD83C\uDFB5', ogg: '\uD83C\uDFB5',
    mp4: '\uD83C\uDFA5', mov: '\uD83C\uDFA5', avi: '\uD83C\uDFA5',
    zip: '\uD83D\uDCE6', tar: '\uD83D\uDCE6', gz: '\uD83D\uDCE6',
    py: '\uD83D\uDC0D', js: '\u2B22', ts: '\u2B22', html: '\uD83C\uDF10',
  };
  return icons[ext] || '\uD83D\uDCC1';
}

function showComingSoon(editorName) {
  _hideLoading();
  dom.body.innerHTML = `
    <div class="studio-legacy-notice">
      <h3>${escapeHtml(editorName)} — Coming Soon</h3>
      <p>The ${escapeHtml(editorName.toLowerCase())} is being built.
         For now, you can download and edit externally.</p>
      <a class="browse-action-pill" href="${escapeHtml(state.artifactInfo?.download_url || '')}"
         target="_blank" rel="noopener noreferrer" style="display:inline-flex">
        Download
      </a>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Dirty state management
// ---------------------------------------------------------------------------
function _updateDirtyState(dirty) {
  state.dirty = dirty;
  // Toggle dirty dot in header
  dom.artifactName?.classList.toggle('dirty', dirty);
  // Toggle save button visual state
  if (dom.saveBtn) {
    dom.saveBtn.disabled = !dirty;
    dom.saveBtn.classList.toggle('dirty', dirty);
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcut help
// ---------------------------------------------------------------------------
function _toggleShortcutHelp() {
  dom.shortcutsOverlay?.classList.toggle('hidden');
}

function _closeShortcutHelp() {
  dom.shortcutsOverlay?.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Visual PDF Editor (for PDFs without source — uploaded/external)
// ---------------------------------------------------------------------------
async function openPdfVisualEditor(artifactId) {
  _hideLoading();
  dom.askBar.style.display = 'none'; // No AI bar for visual PDF editing
  dom.saveBtn.disabled = false;

  try {
    // Fetch PDF binary
    const resp = await fetch(`/api/artifacts/${artifactId}/download`);
    if (!resp.ok) throw new Error(`Failed to fetch PDF: ${resp.status}`);
    const pdfBytes = await resp.arrayBuffer();

    // Lazy-load the PDF editor module
    const { initPdfEditor, getPdfBytes, destroyPdfEditor } = await import('./pdf-editor.js');

    // Initialize in the studio body
    await initPdfEditor(dom.body, pdfBytes, {
      onDirty: () => { _updateDirtyState(true); },
      onSave: async () => {
        const bytes = await getPdfBytes();
        await _savePdfBytes(artifactId, bytes);
      },
    });

    // Wire save button to PDF save
    state._pdfGetBytes = getPdfBytes;
    state._pdfDestroy = destroyPdfEditor;
    state.sourceType = 'pdf_visual';
  } catch (e) {
    console.error('[studio] PDF editor failed:', e);
    dom.body.innerHTML = `<div style="padding:24px;color:var(--text-muted);text-align:center">
      <p>Could not load PDF editor</p>
      <p style="font-size:12px;opacity:0.6">${escapeHtml(e.message || '')}</p>
    </div>`;
  }
}

async function _savePdfBytes(artifactId, bytes) {
  try {
    const blob = new Blob([bytes], { type: 'application/pdf' });
    const form = new FormData();
    form.append('file', blob, 'document.pdf');
    const resp = await fetch(`/api/artifacts/${artifactId}/upload`, {
      method: 'POST',
      body: form,
    });
    if (resp.ok) {
      _updateDirtyState(false);
      showToast('PDF saved', 'success');
      return true;
    }
    showToast(`PDF save failed (HTTP ${resp.status})`, 'error');
    return false;
  } catch (e) {
    console.error('[studio] PDF save failed:', e);
    showToast(`PDF save failed: ${e.message || e}`, 'error');
    return false;
  }
}

// ---------------------------------------------------------------------------
// Document Editor (PDF/DOCX)
// ---------------------------------------------------------------------------
async function openDocumentEditor() {
  _hideLoading();
  dom.askBar.style.display = '';
  dom.saveBtn.disabled = false;

  // Convert sections to markdown
  const markdown = sectionsToMarkdown(state.source);

  // Build the document editor HTML — now with an outline sidebar slot
  // that toggles visibility via the toolbar button. Hidden by default so
  // short documents don't surface empty chrome.
  dom.body.innerHTML = `
    <div class="studio-doc-editor">
      <aside class="studio-outline-panel hidden" id="studio-outline-panel" aria-hidden="true">
        <div class="studio-outline-header">
          <span>Outline</span>
          <button class="studio-outline-close" id="studio-outline-close" title="Hide outline">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
        <nav class="studio-outline-list" id="studio-outline-list"></nav>
      </aside>
      <div class="studio-doc-page" id="studio-doc-page"></div>
      <div class="studio-ai-blocks" id="studio-ai-blocks"></div>
      <div id="studio-palette-mount"></div>
    </div>
  `;

  // Build formatting toolbar
  buildDocToolbar();

  // Restore sidebar collapsed preference.
  _restoreSessionSidebar();

  // Build AI popover
  buildAiPopover('document');

  // Load Milkdown into the page
  const pageEl = document.getElementById('studio-doc-page');
  await loadMilkdownEditor(pageEl, markdown);

  _registerDocFindProvider();
  _wordCountInstall(() => getEditorMarkdown());
  _renderOutline();

  // Mount the Tool Palette so documents get Image / Search / Generate
  // alongside slides + ebooks. Cursor-position insertion comes later;
  // v1 appends a Markdown image to the doc.
  await _mountStudioPalette('document');

  // Load theme from source
  state.theme = state.source?.theme?.preset || state.source?.theme || 'slate';
  loadThemes();
}

// The Milkdown editor renders into a .ProseMirror root inside the page.
// We resolve it lazily at every call so a re-init (theme swap, layout
// change) can't strand the provider pointing at a detached element.
async function _registerDocFindProvider() {
  const mod = await import('./studio-find.js');
  mod.registerProvider('doc', mod.contentEditableProvider(
    () => document.querySelector('#studio-doc-page .ProseMirror, #studio-doc-page [contenteditable]'),
    () => { markDirty(); },
  ));
}

function sectionsToMarkdown(source) {
  if (!source || !source.sections) return '';
  let md = '';
  if (source.title) md += `# ${source.title}\n\n`;
  if (source.author) md += `*${source.author}*\n\n`;

  for (const s of source.sections) {
    const level = '#'.repeat(Math.min((s.level || 1) + 1, 5));
    md += `${level} ${s.heading || 'Untitled Section'}\n\n`;
    if (s.body) md += `${s.body}\n\n`;
    if (s.image_url) {
      md += `![${s.image_caption || ''}](${s.image_url})\n`;
      if (s.image_caption) md += `*${s.image_caption}*\n`;
      md += '\n';
    }
  }
  return md.trim();
}

function markdownToSource(md) {
  // Parse markdown back into source JSON
  const source = { ...state.source, sections: [] };

  // Extract title from first # heading
  const titleMatch = md.match(/^# (.+)$/m);
  if (titleMatch) source.title = titleMatch[1];

  // Extract author from first italic line after title
  const authorMatch = md.match(/^\*([^*]+)\*$/m);
  if (authorMatch && md.indexOf(authorMatch[0]) < 100) {
    source.author = authorMatch[1];
  }

  // Split on ## through ##### headings
  const headingRegex = /^(#{2,5})\s+(.+)$/gm;
  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = headingRegex.exec(md)) !== null) {
    if (parts.length > 0) {
      parts[parts.length - 1].body = md.slice(lastIndex, match.index).trim();
    }
    parts.push({
      level: match[1].length - 1, // ## = level 1, ### = level 2, etc.
      heading: match[2],
      body: '',
      image_url: '',
      image_caption: '',
    });
    lastIndex = match.index + match[0].length;
  }

  // Last section's body
  if (parts.length > 0) {
    parts[parts.length - 1].body = md.slice(lastIndex).trim();
  }

  // Extract images from each section's body
  for (const section of parts) {
    const imgMatch = section.body.match(/!\[([^\]]*)\]\(([^)]+)\)/);
    if (imgMatch) {
      section.image_caption = imgMatch[1];
      section.image_url = imgMatch[2];
      // Remove the image markdown from body
      section.body = section.body.replace(/!\[[^\]]*\]\([^)]+\)\n?\*?[^*]*\*?\n?/g, '').trim();
    }
  }

  source.sections = parts;
  return source;
}

// ---------------------------------------------------------------------------
// Ebook Editor (EPUB)
// ---------------------------------------------------------------------------
function _normalizeEbookSource(source) {
  const src = source && typeof source === 'object' ? source : {};
  const chapters = Array.isArray(src.chapters) ? src.chapters : [];
  return {
    ...src,
    type: 'ebook',
    title: String(src.title || state.artifactInfo?.display_name || 'Untitled Ebook').replace(/\.epub$/i, ''),
    author: String(src.author || ''),
    cover_image_url: String(src.cover_image_url || src.cover_url || ''),
    chapters: chapters.map((chapter, i) => ({
      heading: String(chapter?.heading || `Chapter ${i + 1}`),
      body: Array.isArray(chapter?.body)
        ? chapter.body.join('\n\n')
        : String(chapter?.body || ''),
      image_url: String(chapter?.image_url || ''),
      image_caption: String(chapter?.image_caption || ''),
    })),
  };
}

function _ebookTextForCount() {
  const root = document.getElementById('studio-ebook-editor');
  if (!root) return '';
  return Array.from(root.querySelectorAll('input, textarea'))
    .map((el) => el.value || '')
    .join('\n');
}

function openEbookEditor() {
  _hideLoading();
  dom.askBar.style.display = '';
  dom.saveBtn.disabled = false;
  state.source = _normalizeEbookSource(state.source);

  buildEbookToolbar();
  buildAiPopover('ebook');
  renderEbookEditor();
  _wordCountInstall(_ebookTextForCount);

  // Theme picker — ebooks carry the chosen theme into the EPUB export
  // (see _build_epub_css on the backend). 'storybook' is the warm-parchment
  // default and is injected as an option by loadThemes for ebooks.
  state.theme = state.source?.theme?.preset || state.source?.theme || 'storybook';
  loadThemes();
}

function buildEbookToolbar() {
  if (!dom.toolbar) return;
  dom.toolbar.innerHTML = `
    <span style="font-size:var(--text-xs);color:var(--text-muted)">EPUB Editor</span>
    <span class="studio-toolbar-divider"></span>
    <button class="studio-toolbar-btn" id="studio-ebook-add-chapter" title="Add chapter">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      Chapter
    </button>
    <span class="studio-toolbar-divider"></span>
    <span class="studio-word-count" id="studio-word-count" aria-live="polite"></span>
  `;
  document.getElementById('studio-ebook-add-chapter')?.addEventListener('click', () => {
    const source = getEbookSource();
    source.chapters.push({
      heading: `Chapter ${source.chapters.length + 1}`,
      body: '',
      image_url: '',
      image_caption: '',
    });
    state.source = source;
    renderEbookEditor();
    markDirty();
  });
}

function renderEbookEditor() {
  const source = _normalizeEbookSource(state.source);
  state.source = source;
  const chapters = source.chapters.length
    ? source.chapters
    : [{ heading: 'Chapter 1', body: '', image_url: '', image_caption: '' }];
  source.chapters = chapters;

  const host = _getSessionMainHost();
  // E2 fix: left rail of chapter chips so a 30-chapter novel doesn't force
  // the user to scroll-hunt to the next chapter. Each chip jumps the right-
  // pane to its chapter and visually marks it active. Hidden on narrow
  // viewports via .studio-ebook-layout responsive CSS.
  host.innerHTML = `
    <div class="studio-ebook-layout">
      <aside class="studio-ebook-toc" id="studio-ebook-toc">
        <div class="studio-ebook-toc-head">Chapters</div>
        <ol class="studio-ebook-toc-list">
          ${chapters.map((chapter, i) => `
            <li>
              <button type="button" class="studio-ebook-toc-item${i === 0 ? ' active' : ''}" data-ebook-toc-jump="${i}">
                <span class="studio-ebook-toc-num">${String(i + 1).padStart(2, '0')}</span>
                <span class="studio-ebook-toc-title">${escapeHtml(chapter.heading || `Chapter ${i + 1}`)}</span>
              </button>
            </li>
          `).join('')}
        </ol>
      </aside>
      <div class="studio-ebook-editor" id="studio-ebook-editor">
        <section class="studio-ebook-meta">
          <label>Title<input type="text" data-ebook-field="title" value="${escapeHtml(source.title)}"></label>
          <label>Author<input type="text" data-ebook-field="author" value="${escapeHtml(source.author)}"></label>
          <label>Cover image URL${_renderImagePicker('cover_image_url', source.cover_image_url)}</label>
        </section>
        <div class="studio-ebook-chapters">
          ${chapters.map((chapter, i) => _renderEbookChapter(chapter, i, chapters.length)).join('')}
        </div>
        <div class="studio-ai-blocks" id="studio-ai-blocks"></div>
      </div>
      <div id="studio-palette-mount"></div>
    </div>
  `;
  wireEbookEditor();
  _wireEbookToc();
  // Track the most recently focused image input so the Tool Palette knows
  // where to write Library / Search / Generate selections.
  _installEbookImageFocusTracker();
  _mountStudioPalette('ebook');
}

let _ebookFocusedImageInput = null;
function _installEbookImageFocusTracker() {
  const root = document.getElementById('studio-ebook-editor');
  if (!root) return;
  root.addEventListener('focusin', (e) => {
    const inp = e.target.closest('input[data-ebook-field], input[data-ebook-chapter-field]');
    if (inp && /image_url|cover_image_url/.test(inp.dataset.ebookField || inp.dataset.ebookChapterField || '')) {
      _ebookFocusedImageInput = inp;
    }
  });
}

// E2: chapter rail scroll + active-chapter highlighting. Clicking a chip
// jumps the editor scroll to that chapter; scrolling the editor flips the
// active chip based on which chapter is closest to the top.
function _wireEbookToc() {
  const toc = document.getElementById('studio-ebook-toc');
  const editor = document.getElementById('studio-ebook-editor');
  if (!toc || !editor) return;

  toc.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-ebook-toc-jump]');
    if (!btn) return;
    const idx = Number(btn.dataset.ebookTocJump);
    const target = editor.querySelector(`.studio-ebook-chapter[data-chapter-index="${idx}"]`);
    target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    toc.querySelectorAll('.studio-ebook-toc-item.active').forEach(el => el.classList.remove('active'));
    btn.classList.add('active');
  });

  // Sync the active chip with whichever chapter is closest to the editor's
  // top — debounced so a smooth-scroll jump doesn't flicker through every
  // chapter it passes.
  let scrollTick = null;
  const onScroll = () => {
    if (scrollTick) cancelAnimationFrame(scrollTick);
    scrollTick = requestAnimationFrame(() => {
      const chapters = editor.querySelectorAll('.studio-ebook-chapter');
      if (!chapters.length) return;
      const editorTop = editor.getBoundingClientRect().top;
      let nearestIdx = 0;
      let nearestDist = Infinity;
      chapters.forEach((ch, i) => {
        const dist = Math.abs(ch.getBoundingClientRect().top - editorTop - 60);
        if (dist < nearestDist) { nearestDist = dist; nearestIdx = i; }
      });
      toc.querySelectorAll('.studio-ebook-toc-item.active').forEach(el => el.classList.remove('active'));
      toc.querySelector(`[data-ebook-toc-jump="${nearestIdx}"]`)?.classList.add('active');
    });
  };
  editor.addEventListener('scroll', onScroll, { passive: true });
}

function _renderEbookChapter(chapter, index, total) {
  return `
    <section class="studio-ebook-chapter" data-chapter-index="${index}">
      <div class="studio-ebook-chapter-head">
        <span class="studio-ebook-chapter-num">${String(index + 1).padStart(2, '0')}</span>
        <input type="text" data-ebook-chapter-field="heading" value="${escapeHtml(chapter.heading || '')}" placeholder="Chapter heading">
        <button type="button" class="studio-ebook-remove" data-ebook-action="remove" title="Remove chapter" ${total <= 1 ? 'disabled' : ''}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <textarea data-ebook-chapter-field="body" rows="10" placeholder="Chapter text">${escapeHtml(chapter.body || '')}</textarea>
      <div class="studio-ebook-image-row">
        <label>Image URL${_renderImagePicker('image_url', chapter.image_url)}</label>
        <label>Caption<input type="text" data-ebook-chapter-field="image_caption" value="${escapeHtml(chapter.image_caption || '')}"></label>
      </div>
    </section>
  `;
}

// E1: shared image-URL input pattern. Renders the text input next to a
// "Browse" button that pops an inline picker fed from the user's library
// of generated images. Avoids the LLM-only "/api/image/abc123" magic-URL
// path that everyone forgets the format of.
function _renderImagePicker(fieldName, value) {
  return `
    <span class="studio-image-picker-input">
      <input type="text" data-ebook-chapter-field="${escapeHtml(fieldName)}" data-ebook-field="${escapeHtml(fieldName)}" value="${escapeHtml(value || '')}" placeholder="/api/image/...">
      <button type="button" class="studio-image-picker-btn" data-image-pick title="Browse images">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
      </button>
    </span>
  `;
}

function wireEbookEditor() {
  const root = document.getElementById('studio-ebook-editor');
  if (!root) return;
  root.addEventListener('input', () => {
    state.source = getEbookSource();
    state._wordCountCompute?.();
    markDirty();
  });
  root.addEventListener('click', (e) => {
    const removeBtn = e.target.closest('[data-ebook-action="remove"]');
    if (removeBtn && !removeBtn.disabled) {
      const idx = Number(removeBtn.closest('.studio-ebook-chapter')?.dataset.chapterIndex);
      if (!Number.isFinite(idx)) return;
      const source = getEbookSource();
      if (source.chapters.length <= 1) return;
      source.chapters.splice(idx, 1);
      state.source = source;
      renderEbookEditor();
      state._wordCountCompute?.();
      markDirty();
      return;
    }
    // E1: image picker button opens the shared library-image popover
    // anchored to the input next to the clicked button.
    const pickBtn = e.target.closest('[data-image-pick]');
    if (pickBtn) {
      e.preventDefault();
      const input = pickBtn.parentElement?.querySelector('input');
      if (input) _openImagePicker(input);
    }
  });
}

// E1: tiny image-library picker. Lists the user's existing PNG / image
// artifacts as thumbnails; click one to write its download URL into the
// bound input. Falls back to a "no images yet" message if the library is
// empty. Inline overlay anchored to the page — no separate route needed.
let _imagePickerEl = null;
async function _openImagePicker(targetInput) {
  _closeImagePicker();
  const pop = document.createElement('div');
  pop.className = 'studio-image-picker-pop';
  pop.innerHTML = `
    <div class="studio-image-picker-head">
      <span>Pick an image</span>
      <button type="button" class="studio-image-picker-close" aria-label="Close">&times;</button>
    </div>
    <div class="studio-image-picker-body">
      <div class="studio-image-picker-loading">Loading…</div>
    </div>
  `;
  document.body.appendChild(pop);
  _imagePickerEl = pop;
  const rect = targetInput.getBoundingClientRect();
  pop.style.left = `${Math.max(8, rect.left)}px`;
  pop.style.top = `${rect.bottom + 6}px`;

  pop.querySelector('.studio-image-picker-close')?.addEventListener('click', _closeImagePicker);
  const closeOnOutside = (ev) => {
    if (!pop.contains(ev.target) && ev.target !== targetInput) _closeImagePicker();
  };
  setTimeout(() => document.addEventListener('mousedown', closeOnOutside, { once: true }), 0);

  try {
    const resp = await fetch('/api/artifacts');
    const data = await resp.json();
    const list = Array.isArray(data) ? data : (data.artifacts || data.items || []);
    const images = list.filter(a => (a.format || '').toLowerCase() === 'png' || /^image\//.test(a.mime_type || ''));
    const body = pop.querySelector('.studio-image-picker-body');
    if (!body) return;
    if (!images.length) {
      body.innerHTML = '<div class="studio-image-picker-empty">No images in your library yet. Generate one in chat first.</div>';
      return;
    }
    body.innerHTML = `<div class="studio-image-picker-grid">${images.slice(0, 30).map(img => `
      <button type="button" class="studio-image-picker-tile" data-image-url="${escapeHtml(img.download_url || '')}" title="${escapeHtml(img.display_name || img.filename || '')}">
        <img src="${escapeHtml(img.download_url || '')}" alt="" loading="lazy">
      </button>
    `).join('')}</div>`;
    body.addEventListener('click', (ev) => {
      const tile = ev.target.closest('[data-image-url]');
      if (!tile) return;
      targetInput.value = tile.dataset.imageUrl || '';
      targetInput.dispatchEvent(new Event('input', { bubbles: true }));
      _closeImagePicker();
    });
  } catch (err) {
    const body = pop.querySelector('.studio-image-picker-body');
    if (body) body.innerHTML = '<div class="studio-image-picker-empty">Could not load image library.</div>';
  }
}

function _closeImagePicker() {
  _imagePickerEl?.remove();
  _imagePickerEl = null;
}

function getEbookSource() {
  const root = document.getElementById('studio-ebook-editor');
  const base = _normalizeEbookSource(state.source);
  if (!root) return base;
  const field = (name) => root.querySelector(`[data-ebook-field="${name}"]`)?.value || '';
  const chapters = Array.from(root.querySelectorAll('.studio-ebook-chapter')).map((section, i) => {
    const chapterField = (name) => section.querySelector(`[data-ebook-chapter-field="${name}"]`)?.value || '';
    return {
      heading: chapterField('heading') || `Chapter ${i + 1}`,
      body: chapterField('body'),
      image_url: chapterField('image_url'),
      image_caption: chapterField('image_caption'),
    };
  });
  return {
    ...base,
    title: field('title') || 'Untitled Ebook',
    author: field('author'),
    cover_image_url: field('cover_image_url'),
    chapters,
  };
}

// ---------------------------------------------------------------------------
// Slide Editor (PPTX)
// ---------------------------------------------------------------------------
// Available slide layouts — each declares which edit fields are visible on
// the canvas. Layout id is persisted on the slide so decks can mix styles.
// Iconography doubles as the layout picker's thumbnails: nothing fancy,
// just thick-strokes showing the structural skeleton.
const _SLIDE_LAYOUTS = [
  { id: 'title',      label: 'Title',      icon: '<rect x="3" y="9" width="18" height="3" rx="0.5"/><rect x="6" y="14" width="12" height="2" rx="0.5"/>' },
  { id: 'content',    label: 'Content',    icon: '<rect x="3" y="4" width="18" height="3" rx="0.5"/><rect x="3" y="10" width="18" height="2"/><rect x="3" y="13" width="14" height="2"/><rect x="3" y="16" width="16" height="2"/>' },
  { id: 'section',    label: 'Section',    icon: '<rect x="3" y="10" width="18" height="4" rx="0.5"/>' },
  { id: 'two-column', label: 'Two column', icon: '<rect x="3" y="4" width="18" height="2" rx="0.5"/><rect x="3" y="9" width="8" height="10"/><rect x="13" y="9" width="8" height="10"/>' },
  { id: 'blank',      label: 'Blank',      icon: '<rect x="3" y="6" width="18" height="12" rx="1" stroke-dasharray="2 2" fill="none" stroke="currentColor"/>' },
];

function _layoutIconSvg(layoutId) {
  const l = _SLIDE_LAYOUTS.find(x => x.id === layoutId) || _SLIDE_LAYOUTS[1];
  return `<svg viewBox="0 0 24 24" width="28" height="18" fill="currentColor" stroke="none">${l.icon}</svg>`;
}

async function openSlideEditor() {
  _hideLoading();
  dom.askBar.style.display = '';
  dom.saveBtn.disabled = false;

  // Load slides from source. body2 is only populated for two-column layouts
  // but we always preserve it round-trip so switching layouts back and forth
  // doesn't destroy content.
  state.slides = (state.source.slides || []).map((s) => ({
    layout: s.layout || 'content',
    title: s.title || '',
    body: s.body || '',
    body2: s.body2 || '',
    notes: s.notes || '',
    image_url: s.image_url || '',
    additional_images: Array.isArray(s.additional_images) ? s.additional_images.slice(0, 3) : [],
  }));
  state.currentSlide = 0;
  // Image focus slot — which slot the Tool Palette's Image tab acts on.
  // Defaults to the slide's primary image; per-slide additional slots get
  // focused via the additional-images thumbnail strip below the canvas.
  state.imageFocusSlot = { field: 'image_url' };

  // Build slide editor HTML. The canvas is data-layout-driven so CSS alone
  // can hide/reposition title + body + body2 based on the current slide's
  // layout, without us having to tear down and rebuild the DOM on switch.
  dom.body.innerHTML = `
    <div class="studio-slide-editor" id="studio-slide-editor">
      <div class="studio-slide-panel" id="studio-slide-panel">
        <div id="studio-slide-thumbs"></div>
        <button class="studio-slide-add-btn" id="studio-slide-add">+ Add Slide</button>
      </div>
      <div class="studio-slide-canvas-area">
        <div class="studio-slide-sorter hidden" id="studio-slide-sorter" aria-hidden="true">
          <div class="studio-slide-sorter-head">
            <span class="studio-slide-sorter-title">Slide sorter</span>
            <span class="studio-slide-sorter-sel" id="studio-slide-sorter-sel" aria-live="polite"></span>
            <span class="studio-slide-sorter-hint">Click to select · Shift/Ctrl for multi · Drag to reorder · Del to remove · Dbl-click or Esc to edit</span>
            <button class="studio-toolbar-btn" id="studio-slide-sorter-exit" title="Back to edit (Esc)">Done</button>
          </div>
          <div class="studio-slide-sorter-grid" id="studio-slide-sorter-grid" tabindex="0"></div>
        </div>
        <div class="studio-slide-layout-bar studio-slide-layout-bar-slim" id="studio-slide-layout-bar">
          <span class="studio-slide-layout-label">Layout in palette →</span>
          <button class="studio-slide-layout-action" data-slide-action="duplicate" title="Duplicate slide">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Duplicate
          </button>
        </div>
        <div class="studio-slide-canvas" id="studio-slide-canvas" data-layout="content">
          <input class="studio-slide-title" id="studio-slide-title" type="text" placeholder="Slide title...">
          <div class="studio-slide-body" id="studio-slide-body" contenteditable="true" data-placeholder="Slide content..."></div>
          <div class="studio-slide-body studio-slide-body2" id="studio-slide-body2" contenteditable="true" data-placeholder="Second column..."></div>
        </div>
        <div class="studio-slide-notes">
          <textarea id="studio-slide-notes" placeholder="Speaker notes..."></textarea>
        </div>
        <div class="studio-slide-additional-strip" id="studio-slide-additional-strip"></div>
        <div class="studio-ai-blocks" id="studio-ai-blocks"></div>
      </div>
      <div id="studio-palette-mount"></div>
    </div>
  `;

  // Build toolbar
  dom.toolbar.innerHTML = `
    <span style="font-size:var(--text-xs);color:var(--text-muted)">Slide Editor</span>
    <span class="studio-toolbar-divider"></span>
    <span style="font-size:var(--text-xs);color:var(--text-muted)" id="studio-slide-count">${state.slides.length} slides</span>
    <span class="studio-toolbar-divider"></span>
    <button class="studio-toolbar-btn" id="studio-slide-sorter-btn" title="Slide sorter (grid of all slides)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      Sorter
    </button>
    <button class="studio-preview-btn" id="studio-present-btn" title="Present fullscreen">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
      Present
    </button>
    <span class="studio-toolbar-divider"></span>
    <span class="studio-word-count" id="studio-word-count" aria-live="polite"></span>
  `;

  document.getElementById('studio-present-btn')?.addEventListener('click', startPresentation);
  document.getElementById('studio-slide-sorter-btn')?.addEventListener('click', _enterSlideSorter);
  document.getElementById('studio-slide-sorter-exit')?.addEventListener('click', _exitSlideSorter);

  // Build AI popover for presentations
  buildAiPopover('presentation');

  // Render thumbnails
  renderSlideThumbs();

  // Load first slide
  loadSlide(0);

  // Wire events
  const titleEl = document.getElementById('studio-slide-title');
  const bodyEl = document.getElementById('studio-slide-body');
  const body2El = document.getElementById('studio-slide-body2');
  const notesEl = document.getElementById('studio-slide-notes');
  const addBtn = document.getElementById('studio-slide-add');

  titleEl?.addEventListener('input', () => {
    state.slides[state.currentSlide].title = titleEl.value;
    markDirty();
    updateThumbTitle(state.currentSlide);
  });

  bodyEl?.addEventListener('input', () => {
    state.slides[state.currentSlide].body = bodyEl.innerText;
    markDirty();
  });

  body2El?.addEventListener('input', () => {
    state.slides[state.currentSlide].body2 = body2El.innerText;
    markDirty();
  });

  notesEl?.addEventListener('input', () => {
    state.slides[state.currentSlide].notes = notesEl.value;
    markDirty();
  });

  addBtn?.addEventListener('click', () => {
    addSlide();
  });

  // Layout picker + duplicate action — delegated for brevity.
  document.getElementById('studio-slide-layout-bar')?.addEventListener('click', (e) => {
    const layoutBtn = e.target.closest('[data-layout]');
    if (layoutBtn) {
      _applySlideLayout(layoutBtn.dataset.layout);
      return;
    }
    if (e.target.closest('[data-slide-action="duplicate"]')) {
      _duplicateCurrentSlide();
    }
  });

  _registerSlideFindProvider();
  _wordCountInstall(() => state.slides.map(s => `${s.title}\n${s.body}\n${s.body2 || ''}\n${s.notes}`).join('\n'));

  // Render the additional-images strip and wire its events.
  _renderAdditionalImagesStrip();

  // Mount the Tool Palette + Image tool for this artifact.
  await _mountStudioPalette('presentation');

  // Load theme from source
  state.theme = state.source?.theme?.preset || state.source?.theme || 'slate';
  loadThemes();
}

// Slides search across title + body + body2 + notes of every slide. Matches
// are scoped by slide index + field name so focusMatch can jump to the
// slide, load it, and put the selection on the right field.
async function _registerSlideFindProvider() {
  const mod = await import('./studio-find.js');
  const FIELDS = ['title', 'body', 'body2', 'notes'];
  mod.registerProvider('slides', {
    getMatches(re) {
      const out = [];
      state.slides.forEach((slide, si) => {
        for (const f of FIELDS) {
          const text = String(slide[f] || '');
          let m;
          while ((m = re.exec(text)) !== null) {
            if (m[0].length === 0) { re.lastIndex++; continue; }
            out.push({ slide: si, field: f, start: m.index, end: m.index + m[0].length, text: m[0] });
          }
        }
      });
      return out;
    },
    focusMatch(match) {
      if (state.currentSlide !== match.slide) {
        saveCurrentSlideState();
        state.currentSlide = match.slide;
        loadSlide(match.slide);
        renderSlideThumbs();
      }
      const id = match.field === 'title' ? 'studio-slide-title'
               : match.field === 'body'  ? 'studio-slide-body'
               : match.field === 'body2' ? 'studio-slide-body2'
               :                           'studio-slide-notes';
      const el = document.getElementById(id);
      if (!el) return;
      el.focus();
      // For native inputs / textareas we can set a selection range directly;
      // for contenteditable bodies we fall back to scrollIntoView.
      if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
        try { el.setSelectionRange(match.start, match.end); } catch {}
      } else {
        el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    },
    applyReplace(match, replacement) {
      const slide = state.slides[match.slide];
      if (!slide) return;
      const v = String(slide[match.field] || '');
      slide[match.field] = v.slice(0, match.start) + replacement + v.slice(match.end);
      if (state.currentSlide === match.slide) loadSlide(match.slide);
      markDirty();
    },
  });
}

function _applySlideLayout(layoutId) {
  const slide = state.slides[state.currentSlide];
  if (!slide) return;
  if (!_SLIDE_LAYOUTS.some(l => l.id === layoutId)) return;
  slide.layout = layoutId;
  markDirty();
  _syncCanvasLayout();
  // Layout swatch in the thumbnail has to match so the panel gives an
  // honest preview. Cheap — we re-render the whole strip.
  renderSlideThumbs();
}

// ---------------------------------------------------------------------------
// Tool Palette wiring (Phase 1 — Image tool)
// ---------------------------------------------------------------------------

let _activePalette = null;
let _activeImageTool = null;

function _currentSlide() {
  return state.slides[state.currentSlide];
}

function _suggestedQueryForFocus() {
  const slide = _currentSlide();
  if (!slide) return '';
  const t = (slide.title || '').trim();
  if (t) return t;
  const firstLine = (slide.body || '').split('\n').find((l) => l.trim());
  return (firstLine || '').replace(/^[-*]\s*/, '').trim();
}

function _applyImageToFocus(url, { append = false } = {}) {
  const slide = _currentSlide();
  if (!slide || !url) return;
  if (append || state.imageFocusSlot?.field === 'additional_images') {
    if (!Array.isArray(slide.additional_images)) slide.additional_images = [];
    if (slide.additional_images.length >= 3) return;
    slide.additional_images.push(url);
    state.imageFocusSlot = { field: 'additional_images', index: slide.additional_images.length - 1 };
  } else {
    slide.image_url = url;
    state.imageFocusSlot = { field: 'image_url' };
  }
  markDirty();
  _renderAdditionalImagesStrip();
}

function _setImageFocus(slot) {
  state.imageFocusSlot = slot || { field: 'image_url' };
  _renderAdditionalImagesStrip();
  if (_activePalette) {
    _activePalette.setCtx({
      getFocusSlot: _focusSlotForPalette,
    });
  }
}

function _focusSlotForPalette() {
  const slide = _currentSlide();
  return {
    field: state.imageFocusSlot?.field || 'image_url',
    index: state.imageFocusSlot?.index,
    suggestedQuery: _suggestedQueryForFocus(),
    currentUrl: state.imageFocusSlot?.field === 'additional_images'
      ? (slide?.additional_images || [])[state.imageFocusSlot.index] || ''
      : slide?.image_url || '',
  };
}

function _renderAdditionalImagesStrip() {
  const strip = document.getElementById('studio-slide-additional-strip');
  if (!strip) return;
  const slide = _currentSlide();
  if (!slide) { strip.innerHTML = ''; return; }
  const extras = Array.isArray(slide.additional_images) ? slide.additional_images : [];
  const primaryFocused = state.imageFocusSlot?.field === 'image_url';
  const primaryUrl = slide.image_url || '';
  const parts = [`<span class="studio-slide-additional-strip-label">Slide images</span>`];

  parts.push(`
    <button type="button" class="studio-slide-additional-thumb"
            data-focus-primary
            data-focused="${primaryFocused ? 'true' : 'false'}"
            title="Primary image">
      ${primaryUrl
        ? `<img src="${_escapeAttr(primaryUrl)}" alt="Primary slide image" loading="lazy">`
        : `<span style="font-size:11px;color:var(--text-muted);padding:6px;display:block">Primary</span>`}
      ${primaryUrl ? `<button type="button" class="studio-slide-additional-thumb-remove" data-clear-primary aria-label="Remove primary image">×</button>` : ''}
    </button>
  `);

  extras.forEach((url, i) => {
    const focused = state.imageFocusSlot?.field === 'additional_images'
                  && state.imageFocusSlot?.index === i;
    parts.push(`
      <button type="button" class="studio-slide-additional-thumb"
              data-focus-extra="${i}"
              data-focused="${focused ? 'true' : 'false'}"
              title="Additional image ${i + 1}">
        <img src="${_escapeAttr(url)}" alt="Additional slide image ${i + 1}" loading="lazy">
        <button type="button" class="studio-slide-additional-thumb-remove" data-remove-extra="${i}" aria-label="Remove">×</button>
      </button>
    `);
  });

  if (extras.length < 3) {
    parts.push(`
      <button type="button" class="studio-slide-additional-add" data-add-extra title="Add another image (max 3)">+</button>
    `);
  }

  strip.innerHTML = parts.join('');
  strip.onclick = _onAdditionalStripClick;
}

function _escapeAttr(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function _onAdditionalStripClick(e) {
  const slide = _currentSlide();
  if (!slide) return;

  const clearPrimary = e.target.closest('[data-clear-primary]');
  if (clearPrimary) {
    e.stopPropagation();
    slide.image_url = '';
    markDirty();
    _renderAdditionalImagesStrip();
    return;
  }
  const removeExtra = e.target.closest('[data-remove-extra]');
  if (removeExtra) {
    e.stopPropagation();
    const idx = Number(removeExtra.dataset.removeExtra);
    if (Array.isArray(slide.additional_images)) {
      slide.additional_images.splice(idx, 1);
    }
    // If we just removed the focused slot, fall back to primary.
    if (state.imageFocusSlot?.field === 'additional_images' && state.imageFocusSlot.index === idx) {
      state.imageFocusSlot = { field: 'image_url' };
    }
    markDirty();
    _renderAdditionalImagesStrip();
    return;
  }
  const focusPrimary = e.target.closest('[data-focus-primary]');
  if (focusPrimary) {
    _setImageFocus({ field: 'image_url' });
    _activePalette?.activate?.('image');
    return;
  }
  const focusExtra = e.target.closest('[data-focus-extra]');
  if (focusExtra) {
    const idx = Number(focusExtra.dataset.focusExtra);
    _setImageFocus({ field: 'additional_images', index: idx });
    _activePalette?.activate?.('image');
    return;
  }
  if (e.target.closest('[data-add-extra]')) {
    // "+" → focus the next free additional slot (lazy add: actual slot
    // gets created on first applyImage). Mark focus so the next picked
    // tile lands there.
    const nextIdx = (slide.additional_images || []).length;
    _setImageFocus({ field: 'additional_images', index: nextIdx });
    _activePalette?.activate?.('image');
  }
}

function _focusSlotForArtifact(artifactType) {
  if (artifactType === 'document') {
    return () => {
      // Suggest from the first heading we can find in the current markdown.
      const md = (typeof getEditorMarkdown === 'function' ? getEditorMarkdown() : '') || '';
      const firstHeading = (md.match(/^#+\s+(.+)$/m) || [])[1] || '';
      return {
        field: 'document_body',
        suggestedQuery: firstHeading,
        currentUrl: '',
      };
    };
  }
  if (artifactType === 'ebook') {
    return () => {
      const inp = _ebookFocusedImageInput;
      const suggested = (() => {
        // Suggest from the matching ebook field — chapter heading for
        // image_url inputs, ebook title for cover_image_url.
        if (!inp) return '';
        const field = inp.dataset.ebookField || inp.dataset.ebookChapterField || '';
        if (field === 'cover_image_url') {
          return document.querySelector('[data-ebook-field="title"]')?.value || '';
        }
        // Find the closest heading input
        const chapterEl = inp.closest('.studio-ebook-chapter');
        return chapterEl?.querySelector('[data-ebook-chapter-field="heading"]')?.value || '';
      })();
      return {
        field: inp ? (inp.dataset.ebookField || inp.dataset.ebookChapterField || 'image_url') : 'image_url',
        suggestedQuery: suggested,
        currentUrl: inp?.value || '',
      };
    };
  }
  return _focusSlotForPalette;
}

function _applyImageToSlotForArtifact(artifactType) {
  if (artifactType === 'document') {
    return (url) => {
      if (!url) return;
      _insertImageIntoDocument(url);
    };
  }
  if (artifactType === 'ebook') {
    return (url) => {
      const inp = _ebookFocusedImageInput;
      if (!inp || !url) return;
      inp.value = url;
      // Trigger the editor's input handler so source.* and dirty flag fire.
      inp.dispatchEvent(new Event('input', { bubbles: true }));
    };
  }
  return (url, opts) => _applyImageToFocus(url, opts || {});
}

// ---------------------------------------------------------------------------
// Document image insertion at cursor position.
// --------------------------------------------------------------------------
// The Tool Palette emits a URL when the user picks / generates an image.
// For documents, we want the image to appear where the user was working,
// not appended to the bottom (the original v1 papercut). Crepe / Milkdown
// doesn't expose a public "insert at cursor" API at this version, so we
// take a pragmatic route: find which top-level block in the ProseMirror
// DOM contains the selection, map it to its matching block in the
// markdown source, and splice the image in after that block. Then we
// re-mount the editor and scroll the new image into view with a brief
// highlight so the user sees where it landed.
//
// Multi-paragraph documents with duplicate text still resolve correctly
// because we count block INDEX rather than matching block text — duplicate
// paragraphs would otherwise collide on the first match.

function _insertImageIntoDocument(url) {
  const md = (typeof getEditorMarkdown === 'function' ? getEditorMarkdown() : '') || '';
  const pageEl = document.getElementById('studio-doc-page');
  if (!pageEl) {
    showToast('Open the document editor first.', 'warn');
    return;
  }
  const insertionOffset = _docInsertionOffset(pageEl, md);
  const updated = md.slice(0, insertionOffset).replace(/\s+$/, '')
    + '\n\n![](' + url + ')\n\n'
    + md.slice(insertionOffset).replace(/^\s+/, '');
  loadMilkdownEditor(pageEl, updated).then(() => {
    requestAnimationFrame(() => _highlightInsertedImage(pageEl, url));
  });
  markDirty();
  showToast('Image inserted', 'success');
}

function _findFocusedBlockIndex(pageEl) {
  const pm = pageEl.querySelector('.ProseMirror');
  if (!pm) return -1;
  const sel = window.getSelection();
  if (!sel || !sel.anchorNode || !pm.contains(sel.anchorNode)) return -1;
  let node = sel.anchorNode;
  while (node && node.parentNode !== pm) node = node.parentNode;
  if (!node || node.parentNode !== pm) return -1;
  return Array.from(pm.children).indexOf(node);
}

function _docInsertionOffset(pageEl, md) {
  // No focused block (user clicked through the palette before placing a
  // cursor in the editor) → append at end. Matches the prior behavior so
  // existing flows aren't surprised.
  const focusedIndex = _findFocusedBlockIndex(pageEl);
  if (focusedIndex < 0) return md.length;

  // Walk markdown blocks (separated by blank lines, respecting code fences)
  // until we've passed the focused block, then return its end offset.
  const lines = md.split('\n');
  let blockIdx = -1;
  let inCode = false;
  let endOfCurrent = 0;
  let cursor = 0;
  let inBlock = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineLen = line.length + 1; // +1 for the consumed '\n'
    const trimmed = line.trim();
    if (trimmed.startsWith('```')) inCode = !inCode;

    if (!inCode && trimmed === '') {
      if (inBlock) {
        if (blockIdx === focusedIndex) return endOfCurrent;
        inBlock = false;
      }
    } else {
      if (!inBlock) { inBlock = true; blockIdx++; }
      endOfCurrent = cursor + lineLen;
    }
    cursor += lineLen;
  }
  if (inBlock && blockIdx === focusedIndex) return endOfCurrent;
  // Couldn't map (PM and source diverged for some reason) — append.
  return md.length;
}

function _highlightInsertedImage(pageEl, url) {
  const imgs = pageEl.querySelectorAll('img[src="' + CSS.escape(url) + '"]');
  const newImg = imgs[imgs.length - 1];
  if (!newImg) return;
  newImg.scrollIntoView({ behavior: 'smooth', block: 'center' });
  // Brief outline so the user immediately sees where the image landed.
  // The transition + cleanup keeps the visual debt minimal.
  newImg.style.transition = 'box-shadow 0.6s ease';
  newImg.style.boxShadow = '0 0 0 3px rgba(108, 138, 255, 0.6)';
  setTimeout(() => {
    if (newImg && newImg.style) newImg.style.boxShadow = '';
  }, 1400);
}

async function _mountStudioPalette(artifactType) {
  // Tear down any prior instance (editor switches mount a fresh palette).
  if (_activePalette) {
    try { _activePalette.destroy(); } catch { /* ignore */ }
    _activePalette = null;
    _activeImageTool = null;
  }
  const host = document.getElementById('studio-palette-mount');
  if (!host) return;

  let createPalette, createImageTool, createDesignTool,
      createLayoutTool, createStructureTool, createAiTool,
      createStudioImageApi;
  try {
    ({ createPalette } = await import('./studio/palette.js'));
    ({ createImageTool } = await import('./studio/tools/image.js'));
    ({ createDesignTool } = await import('./studio/tools/design.js'));
    ({ createLayoutTool } = await import('./studio/tools/layout.js'));
    ({ createStructureTool } = await import('./studio/tools/structure.js'));
    ({ createAiTool } = await import('./studio/tools/ai.js'));
    ({ createStudioImageApi } = await import('./studio/api.js'));
  } catch (err) {
    console.warn('Studio palette modules failed to load:', err);
    return;
  }

  const api = createStudioImageApi({ artifactId: state.artifactId });
  const themes = await _fetchThemesForPalette(artifactType);
  const ctx = {
    artifactId: state.artifactId,
    artifactType,
    supportsAppend: artifactType === 'presentation',
    api,
    themes,
    getFocusSlot: _focusSlotForArtifact(artifactType),
    onSlotChange: _applyImageToSlotForArtifact(artifactType),
    getDesign: _getStudioDesign,
    onDesignChange: _setStudioDesign,
    getLayoutOptions: () => _getLayoutOptionsForArtifact(artifactType),
    onLayoutChange: (groupId, value) => _applyLayoutChangeForArtifact(artifactType, groupId, value),
    getStructureItems: () => _getStructureItemsForArtifact(artifactType),
    getStructureActions: () => _getStructureActionsForArtifact(artifactType),
    onStructureJump: (id) => _onStructureJumpForArtifact(artifactType, id),
    onStructureAction: (actionId) => _onStructureActionForArtifact(artifactType, actionId),
    aiActionGroups: _aiActionGroupsForArtifact(artifactType),
    runAi: (action, extra) => studioAiAction(action, extra),
  };
  const palette = createPalette({ host, artifactType, ctx });
  const imageTool = createImageTool();
  const designTool = createDesignTool();
  const layoutTool = createLayoutTool();
  const structureTool = createStructureTool();
  const aiTool = createAiTool();
  palette.registerTool(imageTool);
  palette.registerTool(designTool);
  palette.registerTool(layoutTool);
  palette.registerTool(structureTool);
  palette.registerTool(aiTool);
  // Default tab opens to Image since it's the most common picker action.
  palette.activate('image');
  _activePalette = palette;
  _activeImageTool = imageTool;
}

// --- Design tool ctx wiring -------------------------------------------------
// The design tool is artifact-type-agnostic — it reads/writes a normalized
// design block on state.source. Renderer side honors design via studio_routes
// _resolve_design (lazy migration from source.theme / source.reading).

// Mirror of the Python normalize_design defaults. Kept lean — palette tool
// hydrates from this when state.source.design is missing.
const _DESIGN_DEFAULTS = {
  theme: '',
  font_family: 'system',
  font_size_scale: 1.0,
  line_height: 'comfortable',
  density: 'default',
  accent_override: null,
};

// Forward map: legacy `source.reading` → unified design fields. Mirrors
// _READING_*_TO_* maps in studio_routes.py.
const _READING_FONT_TO_FAMILY = { serif: 'serif', sans: 'sans', dyslexic: 'dyslexic' };
const _READING_SIZE_TO_SCALE = { xs: 0.85, sm: 0.92, md: 1.0, lg: 1.15, xl: 1.45 };
const _READING_LEADING_TO_LH = { compact: 'tight', normal: 'comfortable', relaxed: 'airy' };

function _getStudioDesign() {
  const src = state.source;
  if (!src) return { ..._DESIGN_DEFAULTS };
  if (src.design && typeof src.design === 'object') {
    return { ..._DESIGN_DEFAULTS, ...src.design };
  }
  // Synthesize from legacy fields so the tool reflects whatever the artifact
  // already had before design was canonical.
  const themeName = typeof src.theme === 'string'
    ? src.theme
    : (src.theme?.preset || state.theme || '');
  const synth = { ..._DESIGN_DEFAULTS, theme: themeName };
  const reading = src.reading;
  if (reading && typeof reading === 'object') {
    const f = (reading.font || '').toLowerCase();
    if (_READING_FONT_TO_FAMILY[f]) synth.font_family = _READING_FONT_TO_FAMILY[f];
    const sz = (reading.size || '').toLowerCase();
    if (sz in _READING_SIZE_TO_SCALE) synth.font_size_scale = _READING_SIZE_TO_SCALE[sz];
    const ld = (reading.leading || '').toLowerCase();
    if (ld in _READING_LEADING_TO_LH) synth.line_height = _READING_LEADING_TO_LH[ld];
  }
  return synth;
}

function _setStudioDesign(design) {
  if (!state.source || !design || typeof design !== 'object') return;
  state.source.design = { ..._DESIGN_DEFAULTS, ...design };
  // Keep legacy theme field in sync — older renderer paths and the existing
  // theme popover still read from it. _resolve_design on the server reads
  // design first, then theme as fallback, so writing both is belt-and-braces.
  if (design.theme) {
    state.theme = design.theme;
    if (typeof state.source.theme === 'string') state.source.theme = design.theme;
    else if (state.source.theme && typeof state.source.theme === 'object') state.source.theme.preset = design.theme;
    else state.source.theme = { preset: design.theme };
  }
  markDirty();
  _flushPendingSave().then(() => {
    if (state.viewMode === 'overview') _reloadOverviewIframe();
    if (state.previewOpen) refreshPreview();
  });
  // Bounce the legacy theme picker so its swatches show the new active theme.
  loadThemes();
  // Re-skin the editor surface so the WYSIWYG matches what the renderer will produce.
  _applyArtifactTheme();
}

async function _fetchThemesForPalette(artifactType) {
  // Ebook surfaces use their own EPUB-specific theme list; for the other
  // four formats the central /themes/list is the source of truth.
  if (artifactType === 'ebook') {
    return (typeof _EPUB_THEME_OPTIONS !== 'undefined' && _EPUB_THEME_OPTIONS) || [];
  }
  if (state._themesCache && Array.isArray(state._themesCache) && state._themesCache.length) {
    return state._themesCache;
  }
  try {
    const resp = await fetch('/api/studio/themes/list');
    if (!resp.ok) return [];
    const data = await resp.json();
    state._themesCache = data.themes || [];
    return state._themesCache;
  } catch {
    return [];
  }
}

// --- Layout tool ctx wiring -------------------------------------------------
// Each artifact type returns its own option groups. The Layout tool stays
// generic — these dispatchers turn artifact state into the {id,label,type,
// options,activeValue} contract the tool consumes.

function _getLayoutOptionsForArtifact(artifactType) {
  if (artifactType === 'presentation') {
    const slide = _currentSlide();
    const current = slide?.layout || 'content';
    return [{
      id: 'slide_layout',
      label: 'Slide layout',
      type: 'segmented',
      activeValue: current,
      options: _SLIDE_LAYOUTS.map(l => ({
        value: l.id,
        label: l.label,
        iconSvg: `<svg viewBox="0 0 24 24" width="24" height="16" fill="currentColor">${l.icon}</svg>`,
      })),
      help: 'Title / Section uses centered text. Content + Two column use the body fields.',
    }];
  }
  if (artifactType === 'chart') {
    const types = ['bar', 'line', 'pie', 'scatter', 'area', 'stacked_bar', 'stacked_area', 'horizontal_bar'];
    return [{
      id: 'chart_type',
      label: 'Chart type',
      type: 'segmented',
      activeValue: state.chartConfig?.chart_type || 'bar',
      options: types.map(t => ({ value: t, label: t.replace(/_/g, ' ') })),
    }, {
      id: 'show_values',
      label: 'Data values',
      type: 'toggle',
      toggleLabel: 'Show numeric labels on every datapoint',
      activeValue: !!state.chartConfig?.show_values,
    }];
  }
  if (artifactType === 'spreadsheet') {
    const sheet = state.gridSheets?.[state.currentSheet];
    if (!sheet) return [];
    return [{
      id: 'freeze_header',
      label: 'Header row',
      type: 'toggle',
      toggleLabel: 'Freeze header when scrolling',
      activeValue: sheet.freeze_header !== false,   // default true
    }, {
      id: 'summary_row',
      label: 'Summary row',
      type: 'select',
      activeValue: sheet.summary_row || 'none',
      options: [
        { value: 'none', label: 'None' },
        { value: 'sum', label: 'Sum' },
        { value: 'average', label: 'Average' },
        { value: 'count', label: 'Count' },
      ],
    }];
  }
  // Documents + ebooks defer per-element layout to v2 (needs focused-element
  // tracking the editor doesn't expose today). The tool will surface the
  // "no layout options" empty state.
  return [];
}

function _applyLayoutChangeForArtifact(artifactType, groupId, value) {
  if (artifactType === 'presentation' && groupId === 'slide_layout') {
    _applySlideLayout(value);
    return;
  }
  if (artifactType === 'chart') {
    if (!state.chartConfig) return;
    if (groupId === 'chart_type') {
      state.chartConfig.chart_type = value;
      // Mirror into source so /save re-renders with the new type
      if (state.source) state.source.chart_type = value;
      const sel = document.getElementById('studio-chart-type');
      if (sel) sel.value = value;
      markDirty();
      if (typeof renderChartPreview === 'function') renderChartPreview();
    } else if (groupId === 'show_values') {
      state.chartConfig.show_values = !!value;
      if (state.source) state.source.show_values = !!value;
      const cb = document.getElementById('studio-chart-show-values');
      if (cb) cb.checked = !!value;
      markDirty();
      if (typeof renderChartPreview === 'function') renderChartPreview();
    }
    return;
  }
  if (artifactType === 'spreadsheet') {
    const sheet = state.gridSheets?.[state.currentSheet];
    if (!sheet) return;
    if (groupId === 'freeze_header') {
      sheet.freeze_header = !!value;
    } else if (groupId === 'summary_row') {
      sheet.summary_row = value;
    }
    markDirty();
    _flushPendingSave();
    return;
  }
}

// --- Structure tool ctx wiring ---------------------------------------------

function _getStructureItemsForArtifact(artifactType) {
  if (artifactType === 'presentation') {
    return (state.slides || []).map((s, i) => ({
      id: String(i),
      label: (s.title || '').trim() || `Slide ${i + 1}`,
      kind: 'slide',
      badge: s.layout && s.layout !== 'content' ? s.layout : undefined,
      active: i === state.currentSlide,
    }));
  }
  if (artifactType === 'document') {
    const root = document.querySelector('#studio-doc-page .ProseMirror, #studio-doc-page [contenteditable]');
    if (!root) return [];
    const headings = root.querySelectorAll('h1, h2, h3, h4');
    return Array.from(headings).map((h, i) => {
      if (!h.id) h.id = `studio-h-${i}`;
      return {
        id: h.id,
        label: (h.textContent || '').trim() || '(empty)',
        kind: 'heading',
        level: parseInt(h.tagName.substring(1), 10),
      };
    });
  }
  if (artifactType === 'spreadsheet') {
    return (state.gridSheets || []).map((sh, i) => ({
      id: String(i),
      label: sh.name || `Sheet ${i + 1}`,
      kind: 'sheet',
      active: i === state.currentSheet,
    }));
  }
  if (artifactType === 'chart') {
    return (state.chartConfig?.datasets || []).map((d, i) => ({
      id: String(i),
      label: d.name || `Series ${i + 1}`,
      kind: 'dataset',
      badge: `${(d.values || []).length} pts`,
    }));
  }
  if (artifactType === 'ebook') {
    // Walk the live editor DOM so the list reflects in-progress edits.
    const editor = document.getElementById('studio-ebook-editor');
    if (!editor) return [];
    const chapters = editor.querySelectorAll('.studio-ebook-chapter');
    return Array.from(chapters).map((ch, i) => {
      const titleInput = ch.querySelector('input[data-ebook-chapter-field="heading"]');
      return {
        id: String(i),
        label: (titleInput?.value || '').trim() || `Chapter ${i + 1}`,
        kind: 'chapter',
      };
    });
  }
  return [];
}

function _getStructureActionsForArtifact(artifactType) {
  if (artifactType === 'presentation') {
    return [{ id: 'open_sorter', label: 'Open slide sorter' }];
  }
  if (artifactType === 'spreadsheet') {
    return [{ id: 'add_sheet', label: 'Add sheet' }];
  }
  return [];
}

function _onStructureJumpForArtifact(artifactType, id) {
  if (artifactType === 'presentation') {
    const idx = Number(id);
    if (Number.isFinite(idx) && idx >= 0 && idx < state.slides.length) {
      saveCurrentSlideState();
      state.currentSlide = idx;
      renderSlideThumbs();
      loadSlide(idx);
    }
    return;
  }
  if (artifactType === 'document') {
    const target = document.getElementById(id);
    target?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    return;
  }
  if (artifactType === 'spreadsheet') {
    const idx = Number(id);
    if (Number.isFinite(idx) && state.gridSheets?.[idx]) {
      state.currentSheet = idx;
      renderSheetTabs();
      renderGrid();
    }
    return;
  }
  if (artifactType === 'ebook') {
    const editor = document.getElementById('studio-ebook-editor');
    const target = editor?.querySelector(`.studio-ebook-chapter[data-chapter-index="${id}"]`);
    target?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    return;
  }
  // Chart datasets don't have a scroll-target — clicking is a no-op for v1.
}

function _onStructureActionForArtifact(artifactType, actionId) {
  if (artifactType === 'presentation' && actionId === 'open_sorter') {
    _enterSlideSorter();
    return;
  }
  if (artifactType === 'spreadsheet' && actionId === 'add_sheet') {
    const newSheet = {
      name: `Sheet ${(state.gridSheets?.length || 0) + 1}`,
      headers: ['Column 1', 'Column 2'],
      rows: [['', '']],
    };
    state.gridSheets.push(newSheet);
    state.currentSheet = state.gridSheets.length - 1;
    renderSheetTabs();
    renderGrid();
    markDirty();
  }
}

// --- AI tool ctx wiring ----------------------------------------------------

function _aiActionGroupsForArtifact(artifactType) {
  return AI_ACTIONS[artifactType] || AI_ACTIONS.document;
}

function _duplicateCurrentSlide() {
  saveCurrentSlideState();
  const src = state.slides[state.currentSlide];
  if (!src) return;
  const copy = JSON.parse(JSON.stringify(src));
  state.slides.splice(state.currentSlide + 1, 0, copy);
  state.currentSlide += 1;
  markDirty();
  renderSlideThumbs();
  loadSlide(state.currentSlide);
}

function _syncCanvasLayout() {
  const canvas = document.getElementById('studio-slide-canvas');
  const slide = state.slides[state.currentSlide];
  if (!canvas || !slide) return;
  canvas.dataset.layout = slide.layout || 'content';
  // Reflect the active layout in the picker buttons so users see which
  // slot the current slide is in.
  document.querySelectorAll('.studio-slide-layout-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.layout === (slide.layout || 'content'));
  });
}

function renderSlideThumbs() {
  const container = document.getElementById('studio-slide-thumbs');
  if (!container) return;

  // Each thumb carries its layout icon so the panel gives a real at-a-glance
  // preview rather than a title-only row. draggable=true enables HTML5 DnD
  // for reordering; we wire dragstart/dragover/drop below.
  container.innerHTML = state.slides.map((slide, i) => `
    <div class="studio-slide-thumb${i === state.currentSlide ? ' active' : ''}" data-index="${i}" draggable="true">
      <div class="studio-slide-thumb-num">${i + 1}</div>
      <div class="studio-slide-thumb-layout" aria-hidden="true">${_layoutIconSvg(slide.layout)}</div>
      <div class="studio-slide-thumb-title">${escapeHtml(slide.title || 'Untitled')}</div>
      ${state.slides.length > 1 ? `<button class="studio-slide-thumb-delete" data-delete="${i}" title="Delete slide">&times;</button>` : ''}
    </div>
  `).join('');

  // Wire thumb clicks
  container.querySelectorAll('.studio-slide-thumb').forEach(thumb => {
    thumb.addEventListener('click', (e) => {
      if (e.target.closest('.studio-slide-thumb-delete')) return;
      const idx = parseInt(thumb.dataset.index);
      if (!isNaN(idx)) selectSlide(idx);
    });
  });

  // Wire delete buttons
  container.querySelectorAll('.studio-slide-thumb-delete').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.delete);
      if (!isNaN(idx) && confirm('Delete this slide?')) {
        deleteSlide(idx);
      }
    });
  });

  _wireSlideReorder(container);

  // Update count
  const countEl = document.getElementById('studio-slide-count');
  if (countEl) countEl.textContent = `${state.slides.length} slides`;

  // Re-sync the layout picker with the active slide.
  _syncCanvasLayout();
}

// HTML5 DnD reorder. We track which index is being dragged on dragstart,
// show an insertion marker on dragover based on the cursor's midpoint,
// and on drop splice the array + re-render. drop-above/drop-below classes
// drive the CSS line that indicates where the slide will land.
function _wireSlideReorder(container) {
  let fromIndex = -1;
  container.querySelectorAll('.studio-slide-thumb').forEach(thumb => {
    thumb.addEventListener('dragstart', (e) => {
      fromIndex = parseInt(thumb.dataset.index);
      thumb.classList.add('is-dragging');
      try { e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', String(fromIndex)); } catch {}
    });
    thumb.addEventListener('dragend', () => {
      thumb.classList.remove('is-dragging');
      container.querySelectorAll('.drop-above, .drop-below').forEach(t => t.classList.remove('drop-above', 'drop-below'));
    });
    thumb.addEventListener('dragover', (e) => {
      if (fromIndex < 0) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      container.querySelectorAll('.drop-above, .drop-below').forEach(t => t.classList.remove('drop-above', 'drop-below'));
      const rect = thumb.getBoundingClientRect();
      const above = (e.clientY - rect.top) < rect.height / 2;
      thumb.classList.add(above ? 'drop-above' : 'drop-below');
    });
    thumb.addEventListener('drop', (e) => {
      e.preventDefault();
      if (fromIndex < 0) return;
      const targetIdx = parseInt(thumb.dataset.index);
      const rect = thumb.getBoundingClientRect();
      const above = (e.clientY - rect.top) < rect.height / 2;
      let toIndex = above ? targetIdx : targetIdx + 1;
      // Dropping onto your own position (or the slot immediately after) is
      // a no-op — bail before mutating state so dirty stays clean.
      if (toIndex === fromIndex || toIndex === fromIndex + 1) { fromIndex = -1; return; }
      saveCurrentSlideState();
      const [moved] = state.slides.splice(fromIndex, 1);
      if (toIndex > fromIndex) toIndex--;
      state.slides.splice(toIndex, 0, moved);
      // Keep the currently-selected slide selected even after the move by
      // following the reference — if we moved the active slide, it moved
      // to `toIndex`; otherwise compute the shift.
      if (state.currentSlide === fromIndex) state.currentSlide = toIndex;
      else if (fromIndex < state.currentSlide && toIndex >= state.currentSlide) state.currentSlide--;
      else if (fromIndex > state.currentSlide && toIndex <= state.currentSlide) state.currentSlide++;
      fromIndex = -1;
      markDirty();
      renderSlideThumbs();
      loadSlide(state.currentSlide);
    });
  });
}

function updateThumbTitle(index) {
  const thumb = document.querySelector(`.studio-slide-thumb[data-index="${index}"] .studio-slide-thumb-title`);
  if (thumb) thumb.textContent = state.slides[index]?.title || 'Untitled';
}

function selectSlide(index) {
  if (index < 0 || index >= state.slides.length) return;
  // Save current slide state
  saveCurrentSlideState();
  state.currentSlide = index;
  loadSlide(index);
  renderSlideThumbs();
}

function loadSlide(index) {
  const slide = state.slides[index];
  if (!slide) return;

  const titleEl = document.getElementById('studio-slide-title');
  const bodyEl = document.getElementById('studio-slide-body');
  const body2El = document.getElementById('studio-slide-body2');
  const notesEl = document.getElementById('studio-slide-notes');

  if (titleEl) titleEl.value = slide.title || '';
  if (bodyEl) bodyEl.innerText = slide.body || '';
  if (body2El) body2El.innerText = slide.body2 || '';
  if (notesEl) notesEl.value = slide.notes || '';

  _syncCanvasLayout();
  // Reset image focus to the slide's primary so the palette's suggested
  // query refreshes with the new slide's title.
  state.imageFocusSlot = { field: 'image_url' };
  _renderAdditionalImagesStrip();
  if (_activePalette?.activeId === 'image' && _activeImageTool?.onCtxChange) {
    _activeImageTool.onCtxChange({ getFocusSlot: _focusSlotForPalette });
  }
}

function saveCurrentSlideState() {
  const slide = state.slides[state.currentSlide];
  if (!slide) return;

  const titleEl = document.getElementById('studio-slide-title');
  const bodyEl = document.getElementById('studio-slide-body');
  const body2El = document.getElementById('studio-slide-body2');
  const notesEl = document.getElementById('studio-slide-notes');

  if (titleEl) slide.title = titleEl.value;
  if (bodyEl) slide.body = bodyEl.innerText;
  if (body2El) slide.body2 = body2El.innerText;
  if (notesEl) slide.notes = notesEl.value;
}

function addSlide(slideData) {
  const newSlide = slideData || {
    layout: 'content',
    title: '',
    body: '',
    notes: '',
    image_url: '',
    additional_images: [],
  };
  if (!Array.isArray(newSlide.additional_images)) newSlide.additional_images = [];
  // Insert after current slide
  state.slides.splice(state.currentSlide + 1, 0, newSlide);
  state.currentSlide = state.currentSlide + 1;
  markDirty();
  renderSlideThumbs();
  loadSlide(state.currentSlide);
  // Focus the title
  setTimeout(() => document.getElementById('studio-slide-title')?.focus(), 50);
}

function deleteSlide(index) {
  if (state.slides.length <= 1) return; // keep at least one slide
  state.slides.splice(index, 1);
  if (state.currentSlide >= state.slides.length) {
    state.currentSlide = state.slides.length - 1;
  }
  markDirty();
  renderSlideThumbs();
  loadSlide(state.currentSlide);
}

// ---------------------------------------------------------------------------
// Present Mode (fullscreen slideshow)
// ---------------------------------------------------------------------------
let _presentIdx = 0;

function startPresentation() {
  if (!state.slides || !state.slides.length) return;
  saveCurrentSlideState();
  _presentIdx = state.currentSlide;

  // Create overlay
  let overlay = document.getElementById('studio-present-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'studio-present-overlay';
    overlay.className = 'studio-present-overlay';
    overlay.innerHTML = `
      <div class="studio-present-slide" id="studio-present-slide">
        <h1 id="studio-present-title"></h1>
        <div class="present-body" id="studio-present-body"></div>
      </div>
      <div class="studio-present-counter" id="studio-present-counter"></div>
      <div class="studio-present-exit-hint" id="studio-present-hint">Press Esc to exit &bull; Arrow keys to navigate</div>
    `;
    document.body.appendChild(overlay);
  }

  overlay.classList.remove('hidden');
  renderPresentSlide();

  // Fade the hint after 3 seconds
  const hint = document.getElementById('studio-present-hint');
  setTimeout(() => hint?.classList.add('fade'), 3000);

  // Keyboard handler
  overlay._keyHandler = (e) => {
    if (e.key === 'Escape') {
      exitPresentation();
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
      e.preventDefault();
      if (_presentIdx < state.slides.length - 1) {
        _presentIdx++;
        renderPresentSlide();
      }
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (_presentIdx > 0) {
        _presentIdx--;
        renderPresentSlide();
      }
    }
  };
  document.addEventListener('keydown', overlay._keyHandler);

  // Click to advance
  overlay.addEventListener('click', () => {
    if (_presentIdx < state.slides.length - 1) {
      _presentIdx++;
      renderPresentSlide();
    } else {
      exitPresentation();
    }
  });

  // Show cursor on mouse move, hide after 2s
  let cursorTimer;
  overlay.addEventListener('mousemove', () => {
    overlay.style.cursor = 'default';
    clearTimeout(cursorTimer);
    cursorTimer = setTimeout(() => { overlay.style.cursor = 'none'; }, 2000);
  });
}

function renderPresentSlide() {
  const slide = state.slides[_presentIdx];
  if (!slide) return;

  const titleEl = document.getElementById('studio-present-title');
  const bodyEl = document.getElementById('studio-present-body');
  const counterEl = document.getElementById('studio-present-counter');

  if (titleEl) titleEl.textContent = slide.title || '';
  if (bodyEl) bodyEl.textContent = slide.body || '';
  if (counterEl) counterEl.textContent = `${_presentIdx + 1} / ${state.slides.length}`;
}

function exitPresentation() {
  const overlay = document.getElementById('studio-present-overlay');
  if (overlay) {
    overlay.classList.add('hidden');
    if (overlay._keyHandler) {
      document.removeEventListener('keydown', overlay._keyHandler);
      overlay._keyHandler = null;
    }
  }
  // Jump to the slide we were on in present mode
  selectSlide(_presentIdx);
}

// ---------------------------------------------------------------------------
// Slide Sorter (light table view — macro reorganize)
// ---------------------------------------------------------------------------
// Takes over the canvas area with a large grid of slide previews. Shares the
// underlying state.slides array with the edit view — reordering here is
// visible the instant the user exits back to edit mode. Selection lives on
// state only while sorter mode is active; exiting clears it.
function _enterSlideSorter() {
  // Flush any in-flight edits so the sorter snapshot matches what's typed.
  saveCurrentSlideState();
  const editor = document.getElementById('studio-slide-editor');
  const sorter = document.getElementById('studio-slide-sorter');
  if (!editor || !sorter) return;
  state._sorterActive = true;
  state._sorterSel = new Set([state.currentSlide]);
  editor.classList.add('sorter-mode');
  sorter.classList.remove('hidden');
  sorter.setAttribute('aria-hidden', 'false');
  _renderSlideSorter();
  // Key listener attached to document so Del / Esc fire even if the grid
  // loses focus (e.g., the user clicks the document chrome).
  state._sorterKeyHandler = (e) => {
    if (!state._sorterActive) return;
    if (e.key === 'Escape') { e.preventDefault(); _exitSlideSorter(); return; }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      _sorterDeleteSelection();
    } else if ((e.ctrlKey || e.metaKey) && (e.key === 'a' || e.key === 'A')) {
      e.preventDefault();
      state._sorterSel = new Set(state.slides.map((_, i) => i));
      _renderSlideSorter();
    }
  };
  document.addEventListener('keydown', state._sorterKeyHandler);
  document.getElementById('studio-slide-sorter-grid')?.focus();
}

function _exitSlideSorter() {
  if (!state._sorterActive) return;
  state._sorterActive = false;
  if (state._sorterKeyHandler) {
    document.removeEventListener('keydown', state._sorterKeyHandler);
    state._sorterKeyHandler = null;
  }
  const editor = document.getElementById('studio-slide-editor');
  const sorter = document.getElementById('studio-slide-sorter');
  editor?.classList.remove('sorter-mode');
  sorter?.classList.add('hidden');
  sorter?.setAttribute('aria-hidden', 'true');
  // Re-sync the active slide — currentSlide may have shifted due to deletes
  // or reorders, and the edit panel is stale. Cheapest is to just load it.
  if (state.currentSlide >= state.slides.length) state.currentSlide = Math.max(0, state.slides.length - 1);
  renderSlideThumbs();
  loadSlide(state.currentSlide);
  state._sorterSel = null;
}

// Body preview is truncated so tall slides don't balloon tile height.
const _SORTER_BODY_PREVIEW = 160;

function _renderSlideSorter() {
  const grid = document.getElementById('studio-slide-sorter-grid');
  const sel  = document.getElementById('studio-slide-sorter-sel');
  if (!grid) return;
  const selection = state._sorterSel || new Set();
  grid.innerHTML = state.slides.map((slide, i) => {
    const isSel = selection.has(i);
    const body = String(slide.body || '').slice(0, _SORTER_BODY_PREVIEW);
    const title = String(slide.title || '').slice(0, 120);
    return `
      <div class="studio-sorter-tile${isSel ? ' selected' : ''}${i === state.currentSlide ? ' current' : ''}"
           data-index="${i}" draggable="true" tabindex="-1">
        <div class="studio-sorter-tile-num">${i + 1}</div>
        <div class="studio-sorter-tile-preview" data-layout="${escapeHtml(slide.layout || 'content')}">
          <div class="studio-sorter-tile-title">${escapeHtml(title || 'Untitled')}</div>
          <div class="studio-sorter-tile-body">${escapeHtml(body)}</div>
        </div>
      </div>`;
  }).join('');
  if (sel) sel.textContent = selection.size
    ? `${selection.size} of ${state.slides.length} selected`
    : `${state.slides.length} slide${state.slides.length === 1 ? '' : 's'}`;
  _wireSlideSorter(grid);
}

function _wireSlideSorter(grid) {
  const tiles = grid.querySelectorAll('.studio-sorter-tile');

  // Selection — single click replaces; Ctrl/Cmd toggles one; Shift extends
  // a contiguous range from the most-recently-clicked anchor. Tracking the
  // anchor on state gives Shift-click the same behavior as File Explorer.
  tiles.forEach(tile => {
    tile.addEventListener('click', (e) => {
      if (e.target.closest('[data-sorter-no-select]')) return;
      const idx = Number(tile.dataset.index);
      const sel = state._sorterSel || new Set();
      if (e.shiftKey && state._sorterAnchor != null) {
        const lo = Math.min(state._sorterAnchor, idx);
        const hi = Math.max(state._sorterAnchor, idx);
        sel.clear();
        for (let i = lo; i <= hi; i++) sel.add(i);
      } else if (e.ctrlKey || e.metaKey) {
        if (sel.has(idx)) sel.delete(idx); else sel.add(idx);
        state._sorterAnchor = idx;
      } else {
        sel.clear(); sel.add(idx);
        state._sorterAnchor = idx;
      }
      state._sorterSel = sel;
      _renderSlideSorter();
    });
    tile.addEventListener('dblclick', () => {
      const idx = Number(tile.dataset.index);
      state.currentSlide = idx;
      _exitSlideSorter();
    });
  });

  // HTML5 DnD — supports moving a single or multi-selection. On drop we
  // rebuild state.slides with the moved block relocated. For a multi-drag
  // the dragged tiles keep their relative order.
  let fromIndex = -1;
  tiles.forEach(tile => {
    tile.addEventListener('dragstart', (e) => {
      fromIndex = Number(tile.dataset.index);
      // Ensure the dragged tile is part of the selection; if not, replace it.
      const sel = state._sorterSel || new Set();
      if (!sel.has(fromIndex)) { sel.clear(); sel.add(fromIndex); state._sorterSel = sel; _renderSlideSorter(); }
      tile.classList.add('is-dragging');
      try { e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', String(fromIndex)); } catch {}
    });
    tile.addEventListener('dragend', () => {
      tile.classList.remove('is-dragging');
      grid.querySelectorAll('.drop-before, .drop-after').forEach(t => t.classList.remove('drop-before', 'drop-after'));
    });
    tile.addEventListener('dragover', (e) => {
      if (fromIndex < 0) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      grid.querySelectorAll('.drop-before, .drop-after').forEach(t => t.classList.remove('drop-before', 'drop-after'));
      const rect = tile.getBoundingClientRect();
      const before = (e.clientX - rect.left) < rect.width / 2;
      tile.classList.add(before ? 'drop-before' : 'drop-after');
    });
    tile.addEventListener('drop', (e) => {
      e.preventDefault();
      if (fromIndex < 0) return;
      const targetIdx = Number(tile.dataset.index);
      const rect = tile.getBoundingClientRect();
      const before = (e.clientX - rect.left) < rect.width / 2;
      let insertAt = before ? targetIdx : targetIdx + 1;
      _sorterMoveSelection(insertAt);
      fromIndex = -1;
    });
  });
}

// Relocate the selection to `insertAt`. We extract the selected slides in
// their current order, adjust insertAt for slots removed before it, then
// splice the block into its new home. The anchor + current-slide pointer
// are mapped into the new indices so continuity is preserved.
function _sorterMoveSelection(insertAt) {
  const sel = Array.from(state._sorterSel || []).sort((a, b) => a - b);
  if (!sel.length) return;
  // Count how many selected slides sit before the insertion point so we
  // can adjust the target index after extraction.
  const removedBefore = sel.filter(i => i < insertAt).length;
  const block = sel.map(i => state.slides[i]);
  // Remove from highest to lowest so earlier indices stay valid during splice.
  for (let i = sel.length - 1; i >= 0; i--) state.slides.splice(sel[i], 1);
  const landing = insertAt - removedBefore;
  state.slides.splice(landing, 0, ...block);

  // Map currentSlide into the new layout.
  if (sel.includes(state.currentSlide)) {
    // The active slide moved with the block — land it at the matching offset.
    state.currentSlide = landing + sel.indexOf(state.currentSlide);
  } else {
    // Outside the moving block. First shift down by the count removed
    // before it, then shift back up if the block was inserted at-or-above
    // the resulting position.
    const before = sel.filter(i => i < state.currentSlide).length;
    let after = state.currentSlide - before;
    if (landing <= after) after += block.length;
    state.currentSlide = after;
  }

  // Update selection to the new contiguous block so the user sees what moved.
  state._sorterSel = new Set();
  for (let i = 0; i < block.length; i++) state._sorterSel.add(landing + i);
  state._sorterAnchor = landing;
  markDirty();
  _renderSlideSorter();
}

function _sorterDeleteSelection() {
  const sel = Array.from(state._sorterSel || []).sort((a, b) => a - b);
  if (!sel.length) return;
  // Always keep at least one slide — matches deleteSlide's contract.
  if (sel.length >= state.slides.length) {
    showToast('Cannot delete every slide — at least one must remain.', 'warning');
    return;
  }
  if (!confirm(`Delete ${sel.length} slide${sel.length === 1 ? '' : 's'}?`)) return;
  // Remove highest → lowest so indices stay valid.
  for (let i = sel.length - 1; i >= 0; i--) state.slides.splice(sel[i], 1);
  // Re-home currentSlide: if it was deleted, snap to the nearest surviving
  // index; otherwise shift down by however many earlier slides vanished.
  if (sel.includes(state.currentSlide)) {
    state.currentSlide = Math.min(sel[0], state.slides.length - 1);
  } else {
    const before = sel.filter(i => i < state.currentSlide).length;
    state.currentSlide = Math.max(0, state.currentSlide - before);
  }
  state._sorterSel = new Set();
  state._sorterAnchor = null;
  markDirty();
  _renderSlideSorter();
}

// ---------------------------------------------------------------------------
// Grid Editor (XLSX)
// ---------------------------------------------------------------------------
// Grid undo/redo — snapshot the sheets array as a deep-copied JSON string
// before each mutating operation. We keep 50 steps; deeper history eats
// memory on large sheets and the user can always reload from disk to
// start fresh. Snapshotting rows directly (instead of diffing) is O(N)
// in the cell count but trivially correct for any kind of mutation.
const _GRID_UNDO_MAX = 50;

function _gridSnapshot() {
  if (!state.gridSheets) return;
  state._gridUndo = state._gridUndo || [];
  state._gridRedo = state._gridRedo || [];
  state._gridUndo.push(JSON.stringify(state.gridSheets));
  while (state._gridUndo.length > _GRID_UNDO_MAX) state._gridUndo.shift();
  state._gridRedo.length = 0; // any new edit invalidates the redo branch
  _renderGridHistoryButtons();
}

function _gridUndo() {
  if (!state._gridUndo?.length) return;
  state._gridRedo = state._gridRedo || [];
  state._gridRedo.push(JSON.stringify(state.gridSheets));
  state.gridSheets = JSON.parse(state._gridUndo.pop());
  markDirty();
  renderGrid();
  _renderGridHistoryButtons();
}

function _gridRedo() {
  if (!state._gridRedo?.length) return;
  state._gridUndo = state._gridUndo || [];
  state._gridUndo.push(JSON.stringify(state.gridSheets));
  state.gridSheets = JSON.parse(state._gridRedo.pop());
  markDirty();
  renderGrid();
  _renderGridHistoryButtons();
}

function _renderGridHistoryButtons() {
  const u = document.getElementById('studio-grid-undo');
  const r = document.getElementById('studio-grid-redo');
  if (u) u.disabled = !state._gridUndo?.length;
  if (r) r.disabled = !state._gridRedo?.length;
}

function openGridEditor() {
  _hideLoading();
  dom.askBar.style.display = '';
  dom.saveBtn.disabled = false;

  // Load sheets from source. columnWidths is persisted alongside the data
  // so a user's hand-sized columns survive round-trips. Missing widths
  // default to DEFAULT_COL_WIDTH at render time.
  state.gridSheets = (state.source.sheets || []).map(s => ({
    name: s.name || 'Sheet1',
    headers: [...(s.headers || [])],
    rows: (s.rows || []).map(r => [...r]),
    freeze_header: s.freeze_header ?? true,
    columnWidths: Array.isArray(s.columnWidths) ? [...s.columnWidths] : [],
    // Preserve formats + summary so the editor honors column display rules
    // (currency / percentage) and round-trips them on save. Previously these
    // were silently dropped on the first edit, leaving the saved XLSX with
    // no column formats even when the source defined them.
    formats: (s.formats && typeof s.formats === 'object') ? { ...s.formats } : {},
    summary: (s.summary && typeof s.summary === 'object') ? { ...s.summary } : {},
    sortCol: -1,
    sortDir: null,
  }));
  state.currentSheet = 0;
  state._gridActive = { row: 0, col: 0 };
  state._gridUndo = [];
  state._gridRedo = [];

  // Build grid editor HTML — now with a formula bar docked above the grid.
  dom.body.innerHTML = `
    <div class="studio-grid-editor">
      <div class="studio-grid-tabs" id="studio-grid-tabs"></div>
      <div class="studio-grid-formula-bar">
        <span class="studio-grid-ref" id="studio-grid-ref">A1</span>
        <span class="studio-grid-fx" aria-hidden="true">ƒx</span>
        <input type="text" class="studio-grid-formula-input" id="studio-grid-formula-input"
               placeholder='Type a value or =SUM / IF / CONCAT / LEN / ROUND(…)' autocomplete="off" spellcheck="false">
      </div>
      <div class="studio-grid-table-wrap" id="studio-grid-table-wrap"></div>
      <div class="studio-ai-blocks" id="studio-ai-blocks" style="padding:var(--space-md)"></div>
    </div>
  `;

  // Toolbar
  dom.toolbar.innerHTML = `
    <span style="font-size:var(--text-xs);color:var(--text-muted)">Spreadsheet Editor</span>
    <span class="studio-toolbar-divider"></span>
    <button class="studio-toolbar-btn" id="studio-grid-undo" title="Undo (Ctrl+Z)" disabled>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6.7 3L3 13"/></svg>
    </button>
    <button class="studio-toolbar-btn" id="studio-grid-redo" title="Redo (Ctrl+Shift+Z)" disabled>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M21 7v6h-6"/><path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6.7 3L21 13"/></svg>
    </button>
    <span class="studio-toolbar-divider"></span>
    <button class="studio-toolbar-btn" id="studio-grid-add-row" title="Add row">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
    </button>
    <button class="studio-toolbar-btn" id="studio-grid-add-col" title="Add column">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/><rect x="3" y="3" width="18" height="18" rx="1" opacity="0.3"/></svg>
    </button>
    <span style="font-size:var(--text-xs);color:var(--text-muted)" id="studio-grid-info"></span>
  `;

  document.getElementById('studio-grid-undo')?.addEventListener('click', _gridUndo);
  document.getElementById('studio-grid-redo')?.addEventListener('click', _gridRedo);
  document.getElementById('studio-grid-add-row')?.addEventListener('click', () => {
    const sheet = state.gridSheets[state.currentSheet];
    if (!sheet) return;
    _gridSnapshot();
    sheet.rows.push(sheet.headers.map(() => ''));
    markDirty();
    renderGrid();
  });
  document.getElementById('studio-grid-add-col')?.addEventListener('click', () => {
    const sheet = state.gridSheets[state.currentSheet];
    if (!sheet) return;
    _gridSnapshot();
    sheet.headers.push(`Column ${sheet.headers.length + 1}`);
    sheet.rows.forEach(r => r.push(''));
    markDirty();
    renderGrid();
  });

  // Formula bar: commit on Enter, bail on Escape. Enter moves focus back
  // into the cell so continued typing updates both views in lock-step.
  const formulaEl = document.getElementById('studio-grid-formula-input');
  formulaEl?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      _commitFormulaBar(formulaEl.value);
      formulaEl.blur();
    } else if (e.key === 'Escape') {
      // Restore the active cell's current value so a typo doesn't overwrite it.
      _refreshFormulaBar();
      formulaEl.blur();
    }
  });

  buildAiPopover('spreadsheet');
  renderSheetTabs();
  renderGrid();
  _registerGridFindProvider();

  state.theme = state.source?.theme?.preset || state.source?.theme || 'slate';
  loadThemes();
}

// Grid search covers headers + every data cell of the active sheet. Jumping
// to a match focuses the underlying <td> so the existing highlight styling
// (and the formula bar) update in step. Replacements snapshot into the
// grid's undo history so Ctrl+Z reverses a mistaken replace-all.
async function _registerGridFindProvider() {
  const mod = await import('./studio-find.js');
  mod.registerProvider('grid', {
    getMatches(re) {
      const sheet = state.gridSheets?.[state.currentSheet];
      if (!sheet) return [];
      const out = [];
      const push = (row, col, text) => {
        let m;
        while ((m = re.exec(text)) !== null) {
          if (m[0].length === 0) { re.lastIndex++; continue; }
          out.push({ row, col, start: m.index, end: m.index + m[0].length });
        }
      };
      sheet.headers.forEach((h, c) => push(-1, c, String(h ?? '')));
      sheet.rows.forEach((r, ri) => r.forEach((v, c) => push(ri, c, String(v ?? ''))));
      return out;
    },
    focusMatch(match) {
      const sel = `.studio-grid-table td[contenteditable][data-row="${match.row}"][data-col="${match.col}"]`;
      const cell = document.querySelector(sel);
      if (cell instanceof HTMLElement) {
        cell.focus();
        cell.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    },
    applyReplace(match, replacement) {
      const sheet = state.gridSheets?.[state.currentSheet];
      if (!sheet) return;
      _gridSnapshot();
      if (match.row === -1) {
        const v = String(sheet.headers[match.col] ?? '');
        sheet.headers[match.col] = v.slice(0, match.start) + replacement + v.slice(match.end);
      } else {
        if (!sheet.rows[match.row]) sheet.rows[match.row] = [];
        const v = String(sheet.rows[match.row][match.col] ?? '');
        sheet.rows[match.row][match.col] = v.slice(0, match.start) + replacement + v.slice(match.end);
      }
      markDirty();
      renderGrid();
    },
  });
}

// A2 / AA13 → {row, col}. Returns null on malformed input. Case-insensitive
// on the letter portion, rows are 1-based in A1 notation.
function _parseCellRef(ref) {
  const m = /^([A-Z]+)(\d+)$/i.exec(ref.trim());
  if (!m) return null;
  let col = 0;
  const letters = m[1].toUpperCase();
  for (let i = 0; i < letters.length; i++) col = col * 26 + (letters.charCodeAt(i) - 64);
  return { col: col - 1, row: parseInt(m[2], 10) - 1 };
}
function _refToA1(row, col) {
  let s = '';
  let c = col + 1;
  while (c > 0) {
    const rem = (c - 1) % 26;
    s = String.fromCharCode(65 + rem) + s;
    c = Math.floor((c - 1) / 26);
  }
  return `${s}${row + 1}`;
}

// Formula evaluator — non-recursive, no cell-reference chaining. Cell refs
// resolve to their literal stored string; functions fold over their args.
// Supported:
//   Range aggregates (1 range arg):
//     SUM, AVG/AVERAGE, COUNT, COUNTA, MIN, MAX, UNIQUE
//   Cell-level scalars:
//     IF(cond,a,b) AND(a,b,...) OR(...) NOT(x) IFERROR(x,fallback)
//     ROUND(n,digits?) ABS(n) SQRT(n) POW(n,exp) MOD(a,b)
//     LEN(s) UPPER(s) LOWER(s) TRIM(s) CONCAT(a,b,...)
//     LEFT(s,n) RIGHT(s,n) MID(s,start,len)
//     TODAY() NOW()
// Args may be cell refs (A1), numbers, or quoted strings "hello". No
// nested function calls — one function per formula. Returns a string
// result, '#ERR' for malformed formulas, or null for non-formula input.
function _evalFormula(input, sheet) {
  const s = String(input || '').trim();
  if (!s.startsWith('=')) return null;
  const m = /^=\s*([A-Z]+)\s*\((.*)\)\s*$/i.exec(s);
  if (!m) return '#ERR';
  const fn = m[1].toUpperCase();
  const argsRaw = m[2];

  // TODAY / NOW take no args. Fast path before we try to split.
  if (fn === 'TODAY') return new Date().toISOString().slice(0, 10);
  if (fn === 'NOW')   return new Date().toISOString().replace('T', ' ').slice(0, 16);

  // Aggregate-over-range family — a single A1:B5 argument.
  if (['SUM','AVG','AVERAGE','COUNT','COUNTA','MIN','MAX','UNIQUE'].includes(fn)) {
    const rm = /^\s*([A-Z]+\d+)\s*(?::\s*([A-Z]+\d+))?\s*$/i.exec(argsRaw);
    if (!rm) return '#ERR';
    const a = _parseCellRef(rm[1]);
    const b = rm[2] ? _parseCellRef(rm[2]) : a;
    if (!a || !b) return '#ERR';
    const r1 = Math.min(a.row, b.row), r2 = Math.max(a.row, b.row);
    const c1 = Math.min(a.col, b.col), c2 = Math.max(a.col, b.col);
    const vals = [];
    for (let r = r1; r <= r2; r++)
      for (let c = c1; c <= c2; c++) vals.push(sheet.rows[r]?.[c]);
    const nums = vals.map(v => (v === '' || v == null) ? null : Number(v))
                     .filter(v => v != null && Number.isFinite(v));
    if (fn === 'COUNT')  return String(nums.length);
    if (fn === 'COUNTA') return String(vals.filter(v => v !== '' && v != null).length);
    if (fn === 'UNIQUE') return [...new Set(vals.filter(v => v !== '' && v != null).map(String))].join(', ');
    if (!nums.length)    return fn === 'SUM' ? '0' : '';
    if (fn === 'SUM')             return String(nums.reduce((a,b) => a+b, 0));
    if (fn === 'AVG' || fn === 'AVERAGE') return String(nums.reduce((a,b) => a+b, 0) / nums.length);
    if (fn === 'MIN')             return String(Math.min(...nums));
    if (fn === 'MAX')             return String(Math.max(...nums));
  }

  // Scalar family — comma-separated arg list. Args are resolved one by one
  // to either a number (if they parse) or a string. Strings keep their
  // "quotes" off and cell refs dereference to sheet contents.
  const args = _splitFormulaArgs(argsRaw).map(a => _resolveArg(a, sheet));
  const num  = (v) => { const n = Number(v); return Number.isFinite(n) ? n : NaN; };
  const str  = (v) => (v == null ? '' : String(v));
  const truthy = (v) => {
    if (v === '' || v == null) return false;
    const n = Number(v);
    if (Number.isFinite(n)) return n !== 0;
    const s = String(v).trim().toLowerCase();
    return !(s === 'false' || s === 'no' || s === '0');
  };

  try {
    switch (fn) {
      case 'IF':       return args.length < 2 ? '#ERR' : truthy(args[0]) ? str(args[1]) : str(args[2] ?? '');
      case 'AND':      return String(args.every(truthy));
      case 'OR':       return String(args.some(truthy));
      case 'NOT':      return String(!truthy(args[0]));
      case 'IFERROR': {
        const v = args[0];
        return (v === '#ERR' || v === '' || v == null) ? str(args[1] ?? '') : str(v);
      }
      case 'ROUND':    { const n = num(args[0]); const d = Number(args[1] ?? 0) | 0;
                         if (!Number.isFinite(n)) return '#ERR';
                         const mul = Math.pow(10, d); return String(Math.round(n * mul) / mul); }
      case 'ABS':      { const n = num(args[0]); return Number.isFinite(n) ? String(Math.abs(n)) : '#ERR'; }
      case 'SQRT':     { const n = num(args[0]); return Number.isFinite(n) && n >= 0 ? String(Math.sqrt(n)) : '#ERR'; }
      case 'POW':      { const a = num(args[0]), b = num(args[1]); return Number.isFinite(a) && Number.isFinite(b) ? String(Math.pow(a, b)) : '#ERR'; }
      case 'MOD':      { const a = num(args[0]), b = num(args[1]); return Number.isFinite(a) && Number.isFinite(b) && b !== 0 ? String(a % b) : '#ERR'; }
      case 'LEN':      return String(str(args[0]).length);
      case 'UPPER':    return str(args[0]).toUpperCase();
      case 'LOWER':    return str(args[0]).toLowerCase();
      case 'TRIM':     return str(args[0]).trim();
      case 'CONCAT':   return args.map(str).join('');
      case 'LEFT':     { const n = Number(args[1] ?? 1) | 0; return str(args[0]).slice(0, Math.max(0, n)); }
      case 'RIGHT':    { const n = Number(args[1] ?? 1) | 0; return n <= 0 ? '' : str(args[0]).slice(-n); }
      case 'MID':      { const start = (Number(args[1]) | 0) - 1; const len = Number(args[2] ?? 1) | 0;
                         return str(args[0]).substr(Math.max(0, start), Math.max(0, len)); }
    }
  } catch { return '#ERR'; }
  return '#ERR';
}

// Split comma-separated args respecting double-quoted strings so commas
// inside "hello, world" stay together. No escape handling — simple cases
// only; anyone doing anything sophisticated can reference a cell.
function _splitFormulaArgs(src) {
  const out = [];
  let cur = '';
  let inQ = false;
  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if (ch === '"') { inQ = !inQ; cur += ch; continue; }
    if (ch === ',' && !inQ) { out.push(cur.trim()); cur = ''; continue; }
    cur += ch;
  }
  if (cur.trim() !== '' || out.length) out.push(cur.trim());
  return out;
}

// Resolve one argument to a value. Cell refs deref; quoted strings unquote;
// numbers stay numeric; bare strings pass through (so TRUE / FALSE / a
// column letter typed as text works).
function _resolveArg(raw, sheet) {
  const a = raw.trim();
  if (!a) return '';
  if (a.startsWith('"') && a.endsWith('"')) return a.slice(1, -1);
  if (/^[A-Z]+\d+$/i.test(a)) {
    const ref = _parseCellRef(a);
    return ref ? (sheet.rows[ref.row]?.[ref.col] ?? '') : '#ERR';
  }
  return a;
}

function _activeCell() {
  const sheet = state.gridSheets?.[state.currentSheet];
  const pos = state._gridActive || { row: 0, col: 0 };
  return { sheet, ...pos };
}

function _refreshFormulaBar() {
  const refEl = document.getElementById('studio-grid-ref');
  const input = document.getElementById('studio-grid-formula-input');
  const { sheet, row, col } = _activeCell();
  if (!refEl || !input || !sheet) return;
  refEl.textContent = row === -1 ? `${String.fromCharCode(65 + (col % 26))} (header)` : _refToA1(row, col);
  const val = row === -1 ? sheet.headers[col] : sheet.rows[row]?.[col];
  input.value = val ?? '';
}

function _commitFormulaBar(raw) {
  const { sheet, row, col } = _activeCell();
  if (!sheet) return;
  let next = raw;
  // Evaluate formulas against the current sheet snapshot. Non-formula input
  // lands verbatim so strings containing a leading quote or equals-less
  // text aren't touched.
  if (typeof raw === 'string' && raw.startsWith('=')) {
    const result = _evalFormula(raw, sheet);
    if (result != null) next = result;
  }
  if (row === -1) {
    sheet.headers[col] = next;
  } else {
    if (!sheet.rows[row]) sheet.rows[row] = [];
    sheet.rows[row][col] = next;
  }
  markDirty();
  renderGrid();
  // Keep focus on the active cell so the user can tab into the next one.
  _focusActiveCell();
}

function _focusActiveCell() {
  const { row, col } = _activeCell();
  const cell = document.querySelector(
    `.studio-grid-table td[contenteditable][data-row="${row}"][data-col="${col}"]`,
  );
  if (cell instanceof HTMLElement) cell.focus();
}

function renderSheetTabs() {
  const container = document.getElementById('studio-grid-tabs');
  if (!container) return;

  container.innerHTML = state.gridSheets.map((sheet, i) =>
    `<button class="studio-grid-tab${i === state.currentSheet ? ' active' : ''}" data-sheet="${i}">${escapeHtml(sheet.name)}</button>`
  ).join('');

  container.querySelectorAll('.studio-grid-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      saveGridState();
      state.currentSheet = parseInt(tab.dataset.sheet);
      renderSheetTabs();
      renderGrid();
    });
  });
}

const _DEFAULT_COL_WIDTH = 120;
const _MIN_COL_WIDTH = 48;

// Sorted-row indices for the active sheet. Numeric detection mirrors the
// CSV editor: every non-empty cell in the column must parse as a finite
// number, otherwise we fall back to localeCompare.
function _sortedGridRowIndices(sheet) {
  if (!sheet || sheet.sortCol < 0 || !sheet.sortDir) {
    return sheet ? sheet.rows.map((_, i) => i) : [];
  }
  const col = sheet.sortCol;
  const dir = sheet.sortDir === 'asc' ? 1 : -1;
  const isNumeric = sheet.rows.every(r => {
    const v = String(r[col] ?? '').trim();
    return v === '' || Number.isFinite(Number(v));
  });
  const idx = sheet.rows.map((_, i) => i);
  idx.sort((a, b) => {
    const va = sheet.rows[a][col] ?? '';
    const vb = sheet.rows[b][col] ?? '';
    if (isNumeric) {
      const na = va === '' ? Infinity * dir : Number(va);
      const nb = vb === '' ? Infinity * dir : Number(vb);
      return (na - nb) * dir;
    }
    return String(va).localeCompare(String(vb)) * dir;
  });
  return idx;
}

// Format a raw cell value for display per the column's format rule
// (matching the XLSX renderer's currency / percentage column styles).
// Returns the raw value untouched when no format applies or the value
// isn't numeric, so non-numeric "Jul" / formulas / blanks render as-is.
function _formatCellDisplay(value, format) {
  if (value === '' || value == null) return '';
  const fmt = String(format || '').toLowerCase();
  if (!fmt || (fmt !== 'currency' && fmt !== 'percentage' && fmt !== 'percent')) {
    return String(value);
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!isFinite(n)) return String(value);
  if (fmt === 'currency') {
    return n.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
  }
  // percentage: 0.08 → "8.0%"
  return (n * 100).toFixed(1) + '%';
}

function _columnFormat(sheet, c) {
  const header = sheet.headers?.[c];
  if (!header || !sheet.formats) return '';
  return sheet.formats[header] || '';
}

function renderGrid() {
  const wrap = document.getElementById('studio-grid-table-wrap');
  if (!wrap) return;

  const sheet = state.gridSheets[state.currentSheet];
  if (!sheet) { wrap.innerHTML = ''; return; }

  const colLetters = sheet.headers.map((_, i) => String.fromCharCode(65 + (i % 26)));
  const widthFor = (c) => Number(sheet.columnWidths?.[c]) || _DEFAULT_COL_WIDTH;
  const order = _sortedGridRowIndices(sheet);

  const sortArrow = (c) => {
    if (sheet.sortCol !== c || !sheet.sortDir) return '';
    return sheet.sortDir === 'asc'
      ? '<svg width="8" height="8" viewBox="0 0 24 24" fill="currentColor"><path d="M7 14l5-5 5 5z"/></svg>'
      : '<svg width="8" height="8" viewBox="0 0 24 24" fill="currentColor"><path d="M7 10l5 5 5-5z"/></svg>';
  };

  // colgroup lets us set widths once per column and have every cell
  // inherit — cheaper than writing width inline on each <td>, and the
  // right hook for sticky positioning interactions.
  let html = '<table class="studio-grid-table"><colgroup>';
  html += '<col class="studio-grid-rownum-col">';
  for (let c = 0; c < sheet.headers.length; c++) {
    html += `<col data-col="${c}" style="width:${widthFor(c)}px">`;
  }
  html += '</colgroup><thead><tr>';
  html += '<th class="studio-grid-row-num"></th>';
  for (let c = 0; c < sheet.headers.length; c++) {
    const sortedCls = sheet.sortCol === c && sheet.sortDir ? ' is-sorted' : '';
    html += `<th class="studio-grid-colhead${sortedCls}" data-col="${c}" data-sortable="1">
      <span class="studio-grid-col-letter">${escapeHtml(colLetters[c])}</span>
      <span class="studio-grid-col-arrow">${sortArrow(c)}</span>
      <span class="studio-grid-col-resize" data-resize="${c}" title="Drag to resize"></span>
    </th>`;
  }
  html += '</tr><tr>';
  html += '<th class="studio-grid-row-num">H</th>';
  for (let c = 0; c < sheet.headers.length; c++) {
    html += `<td contenteditable="true" data-row="-1" data-col="${c}" class="studio-grid-header-cell">${escapeHtml(String(sheet.headers[c] || ''))}</td>`;
  }
  html += '</tr></thead><tbody>';

  for (let visible = 0; visible < order.length; visible++) {
    const r = order[visible];
    html += '<tr>';
    html += `<td class="studio-grid-row-num" data-row="${r}">${visible + 1}</td>`;
    for (let c = 0; c < sheet.headers.length; c++) {
      const val = sheet.rows[r]?.[c] ?? '';
      // Apply column format for display. Raw value lives in sheet.rows; on
      // focus the cell flips back to raw so the user edits the underlying
      // number, on blur the cell re-formats from the new raw value.
      const colFormat = _columnFormat(sheet, c);
      const display = _formatCellDisplay(val, colFormat);
      const numericCls = (colFormat === 'currency' || colFormat === 'percentage' || colFormat === 'percent') ? ' studio-grid-numeric' : '';
      html += `<td contenteditable="true" data-row="${r}" data-col="${c}" class="${numericCls.trim()}">${escapeHtml(display)}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';

  wrap.innerHTML = html;

  // Update info
  const info = document.getElementById('studio-grid-info');
  if (info) info.textContent = `${sheet.rows.length} rows, ${sheet.headers.length} cols`;

  _wireGridColumnResize(wrap, sheet);

  // Wire cell editing
  wrap.querySelectorAll('td[contenteditable]').forEach(cell => {
    cell.addEventListener('focus', () => {
      state._gridActive = { row: parseInt(cell.dataset.row), col: parseInt(cell.dataset.col) };
      _refreshFormulaBar();
      // Highlight the active row/col header for spatial feedback.
      wrap.querySelectorAll('.studio-grid-colhead.active, .studio-grid-row-num.active').forEach(el => el.classList.remove('active'));
      wrap.querySelector(`.studio-grid-colhead[data-col="${cell.dataset.col}"]`)?.classList.add('active');
      if (cell.parentElement) cell.parentElement.querySelector('.studio-grid-row-num')?.classList.add('active');
      // For formatted columns ($ / %), flip to the underlying raw value so
      // the user edits "142000" not "$142,000.00" — the latter would round-
      // trip as garbage on save. Header row (data-row="-1") and unformatted
      // cells stay as-is.
      const row = parseInt(cell.dataset.row);
      if (row >= 0) {
        const col = parseInt(cell.dataset.col);
        const fmt = _columnFormat(sheet, col);
        if (fmt) {
          const raw = sheet.rows[row]?.[col];
          if (raw !== undefined && raw !== null && raw !== '') {
            cell.textContent = String(raw);
          }
        }
      }
    });
    cell.addEventListener('blur', () => {
      const row = parseInt(cell.dataset.row);
      const col = parseInt(cell.dataset.col);
      const raw = cell.textContent;
      const val = raw.trim();

      // Evaluate formulas entered directly into a cell the same way the
      // formula bar does — so typing "=SUM(A1:A5)" + Tab commits the
      // numeric result into the cell.
      let toStore = val;
      if (val.startsWith('=')) {
        const result = _evalFormula(val, sheet);
        if (result != null) toStore = result;
      }

      if (row === -1) {
        if (sheet.headers[col] !== toStore) {
          _gridSnapshot();
          sheet.headers[col] = toStore;
          markDirty();
          if (toStore !== val) cell.textContent = toStore;
        }
      } else {
        if (!sheet.rows[row]) sheet.rows[row] = [];
        const prev = String(sheet.rows[row][col] ?? '');
        if (prev !== toStore) {
          _gridSnapshot();
          sheet.rows[row][col] = toStore;
          markDirty();
        }
        // Always re-format on blur — even when the user didn't change the
        // value, the focus handler swapped to raw and we need to restore the
        // formatted display ("142000" → "$142,000.00").
        const fmt = _columnFormat(sheet, col);
        if (fmt) {
          cell.textContent = _formatCellDisplay(toStore, fmt);
        } else if (toStore !== val) {
          cell.textContent = toStore;
        }
      }
    });

    // Keyboard navigation
    cell.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const next = e.shiftKey ? cell.previousElementSibling : cell.nextElementSibling;
        if (next?.contentEditable === 'true') next.focus();
      } else if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const row = cell.parentElement;
        const nextRow = row?.nextElementSibling;
        if (nextRow) {
          const colIdx = Array.from(row.children).indexOf(cell);
          const nextCell = nextRow.children[colIdx];
          if (nextCell?.contentEditable === 'true') nextCell.focus();
        }
      }
    });
  });

  // Click a column letter (letter-cell, not the resize handle or editable
  // header) to cycle sort asc → desc → none. Delegated so one listener
  // covers every header without re-wiring on render.
  wrap.querySelector('thead tr')?.addEventListener('click', (e) => {
    if (e.target.closest('.studio-grid-col-resize')) return; // resize handle owns clicks
    const th = e.target.closest('.studio-grid-colhead');
    if (!th) return;
    const col = Number(th.dataset.col);
    if (sheet.sortCol !== col) { sheet.sortCol = col; sheet.sortDir = 'asc'; }
    else if (sheet.sortDir === 'asc')  sheet.sortDir = 'desc';
    else { sheet.sortCol = -1; sheet.sortDir = null; }
    renderGrid();
  });

  // Right-click row/column for insert + delete. Blocks the browser
  // context menu only when we have a target — clicking empty space still
  // gets the default menu so "Inspect element" works where we haven't
  // claimed it.
  wrap.addEventListener('contextmenu', (e) => {
    const rowNumEl = e.target.closest('.studio-grid-row-num[data-row]');
    const colHead  = e.target.closest('.studio-grid-colhead');
    const cell     = e.target.closest('td[contenteditable]');
    if (!rowNumEl && !colHead && !cell) return;
    e.preventDefault();
    const rowIdx = rowNumEl ? Number(rowNumEl.dataset.row)
                 : cell && cell.dataset.row !== '-1' ? Number(cell.dataset.row)
                 : null;
    const colIdx = colHead ? Number(colHead.dataset.col)
                 : cell ? Number(cell.dataset.col)
                 : null;
    _openGridContextMenu(e.clientX, e.clientY, sheet, rowIdx, colIdx);
  });

  // Restore focus highlight + formula bar when re-rendering after edits
  // so the active cell halo doesn't vanish on every commit.
  _refreshFormulaBar();
  _renderGridHistoryButtons();
}

// Small context-menu utility tailored to the grid. We render inline into
// body so it escapes any editor overflow:hidden container, and close on
// any outside click. Actions operate on the sheet passed in, snapshotting
// undo state before each mutation so Ctrl+Z reverses them.
function _openGridContextMenu(x, y, sheet, rowIdx, colIdx) {
  _closeGridContextMenu();
  const menu = document.createElement('div');
  menu.className = 'studio-grid-menu';
  const items = [];
  if (rowIdx != null) {
    items.push({ id: 'insert-row-above', label: 'Insert row above' });
    items.push({ id: 'insert-row-below', label: 'Insert row below' });
    items.push({ id: 'delete-row',       label: 'Delete row', danger: true });
  }
  if (colIdx != null) {
    if (items.length) items.push({ sep: true });
    items.push({ id: 'insert-col-before', label: 'Insert column left' });
    items.push({ id: 'insert-col-after',  label: 'Insert column right' });
    items.push({ id: 'delete-col',        label: 'Delete column', danger: true });
  }
  if (sheet.sortCol >= 0) {
    items.push({ sep: true });
    items.push({ id: 'clear-sort', label: 'Clear sort' });
  }
  if (!items.length) return;
  menu.innerHTML = items.map(it =>
    it.sep ? `<div class="studio-grid-menu-sep"></div>`
           : `<button class="studio-grid-menu-item${it.danger ? ' danger' : ''}" data-id="${it.id}">${escapeHtml(it.label)}</button>`,
  ).join('');
  document.body.appendChild(menu);
  const rect = menu.getBoundingClientRect();
  menu.style.left = `${Math.min(x, window.innerWidth  - rect.width  - 8)}px`;
  menu.style.top  = `${Math.min(y, window.innerHeight - rect.height - 8)}px`;

  menu.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-id]');
    if (!btn) return;
    const id = btn.dataset.id;
    _gridSnapshot();
    if (id === 'insert-row-above' && rowIdx != null) {
      sheet.rows.splice(rowIdx, 0, sheet.headers.map(() => ''));
    } else if (id === 'insert-row-below' && rowIdx != null) {
      sheet.rows.splice(rowIdx + 1, 0, sheet.headers.map(() => ''));
    } else if (id === 'delete-row' && rowIdx != null) {
      sheet.rows.splice(rowIdx, 1);
    } else if (id === 'insert-col-before' && colIdx != null) {
      sheet.headers.splice(colIdx, 0, `Column ${sheet.headers.length + 1}`);
      sheet.rows.forEach(r => r.splice(colIdx, 0, ''));
      sheet.columnWidths?.splice?.(colIdx, 0, _DEFAULT_COL_WIDTH);
    } else if (id === 'insert-col-after' && colIdx != null) {
      sheet.headers.splice(colIdx + 1, 0, `Column ${sheet.headers.length + 1}`);
      sheet.rows.forEach(r => r.splice(colIdx + 1, 0, ''));
      sheet.columnWidths?.splice?.(colIdx + 1, 0, _DEFAULT_COL_WIDTH);
    } else if (id === 'delete-col' && colIdx != null) {
      sheet.headers.splice(colIdx, 1);
      sheet.rows.forEach(r => r.splice(colIdx, 1));
      sheet.columnWidths?.splice?.(colIdx, 1);
      // If the sorted column was deleted, collapse the sort state.
      if (sheet.sortCol === colIdx) { sheet.sortCol = -1; sheet.sortDir = null; }
      else if (sheet.sortCol > colIdx) sheet.sortCol--;
    } else if (id === 'clear-sort') {
      sheet.sortCol = -1; sheet.sortDir = null;
    }
    markDirty();
    renderGrid();
    _closeGridContextMenu();
  });
  setTimeout(() => document.addEventListener('click', _closeGridContextMenu, { once: true }), 0);
}

function _closeGridContextMenu() {
  document.querySelectorAll('.studio-grid-menu').forEach(m => m.remove());
}

// Drag-resize handles live inside each column header. On pointerdown we
// capture the pointer, stream width updates into the <col> element so
// the grid reflows in real time, then commit the final width into the
// sheet model so it persists across renders and saves.
function _wireGridColumnResize(wrap, sheet) {
  wrap.querySelectorAll('.studio-grid-col-resize').forEach(handle => {
    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const col = Number(handle.dataset.resize);
      const colEl = wrap.querySelector(`col[data-col="${col}"]`);
      if (!colEl) return;
      const startX = e.clientX;
      const startW = colEl.offsetWidth;
      handle.setPointerCapture(e.pointerId);
      handle.classList.add('is-dragging');
      document.body.classList.add('studio-grid-resizing');
      const onMove = (ev) => {
        const next = Math.max(_MIN_COL_WIDTH, startW + (ev.clientX - startX));
        colEl.style.width = `${next}px`;
      };
      const onUp = () => {
        handle.removeEventListener('pointermove', onMove);
        handle.removeEventListener('pointerup', onUp);
        handle.removeEventListener('pointercancel', onUp);
        handle.classList.remove('is-dragging');
        document.body.classList.remove('studio-grid-resizing');
        const w = colEl.offsetWidth;
        sheet.columnWidths = sheet.columnWidths || [];
        if (sheet.columnWidths[col] !== w) {
          sheet.columnWidths[col] = w;
          markDirty();
        }
      };
      handle.addEventListener('pointermove', onMove);
      handle.addEventListener('pointerup', onUp);
      handle.addEventListener('pointercancel', onUp);
    });
    // Double-click resets to default width.
    handle.addEventListener('dblclick', (e) => {
      e.stopPropagation();
      const col = Number(handle.dataset.resize);
      sheet.columnWidths = sheet.columnWidths || [];
      sheet.columnWidths[col] = _DEFAULT_COL_WIDTH;
      markDirty();
      renderGrid();
    });
  });
}

function saveGridState() {
  // Grid state is saved on blur of each cell, nothing extra needed here
}

function getGridSource() {
  saveGridState();
  return {
    ...state.source,
    sheets: state.gridSheets.map(s => ({
      name: s.name,
      headers: s.headers,
      rows: s.rows,
      freeze_header: s.freeze_header,
      columnWidths: s.columnWidths || [],
      // Round-trip column formats + summary so a re-render through the XLSX
      // engine preserves $ / % column styling and sum/avg rows.
      formats: s.formats && Object.keys(s.formats).length ? s.formats : undefined,
      summary: s.summary && Object.keys(s.summary).length ? s.summary : undefined,
    })),
  };
}

// ---------------------------------------------------------------------------
// Chart.js CDN Loading
// ---------------------------------------------------------------------------
let ChartJS = null;
let _chartLoaded = false;

let ChartDataLabels = null;

async function ensureChartJsLoaded() {
  if (_chartLoaded) return true;
  try {
    // Pinned to exact version — prevents silent breaking changes from CDN.
    // Update version manually when upgrading.
    const mod = await import('https://cdn.jsdelivr.net/npm/chart.js@4.5.1/+esm');
    ChartJS = mod.Chart;
    // Register all components
    const { CategoryScale, LinearScale, BarElement, LineElement, PointElement,
            ArcElement, BarController, LineController, PieController, ScatterController,
            DoughnutController, RadarController, PolarAreaController,
            Tooltip, Legend, Title } = mod;
    ChartJS.register(
      CategoryScale, LinearScale, BarElement, LineElement, PointElement,
      ArcElement, BarController, LineController, PieController, ScatterController,
      DoughnutController, RadarController, PolarAreaController,
      Tooltip, Legend, Title
    );
    // C1 fix: lazy-load chartjs-plugin-datalabels so the "Show data values"
    // toggle actually paints labels above bars / inside slices. Without it
    // the checkbox was a no-op. Plugin is registered per-chart in
    // renderChartPreview, not globally, so other future Chart.js usages
    // don't get surprise labels.
    try {
      const dl = await import('https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/+esm');
      ChartDataLabels = dl.default || dl;
    } catch (dlErr) {
      console.warn('Failed to load chartjs-plugin-datalabels:', dlErr);
    }
    _chartLoaded = true;
  } catch (err) {
    console.error('Failed to load Chart.js:', err);
  }
  return _chartLoaded;
}

// ---------------------------------------------------------------------------
// Chart Editor
// ---------------------------------------------------------------------------
async function openChartEditor() {
  _hideLoading();
  dom.askBar.style.display = '';
  dom.saveBtn.disabled = false;

  // Load chart config from source
  state.chartConfig = {
    title: state.source.title || 'Chart',
    subtitle: state.source.subtitle || '',
    chart_type: state.source.chart_type || 'bar',
    x_label: state.source.x_label || '',
    y_label: state.source.y_label || '',
    labels: [...(state.source.labels || [])],
    datasets: (state.source.datasets || []).map(d => ({
      name: d.name || '',
      values: [...(d.values || [])],
    })),
    show_values: state.source.show_values || false,
    value_format: state.source.value_format || 'auto',
    sort: state.source.sort || 'none',
    caption: state.source.caption || '',
  };

  const chartTypes = ['bar', 'line', 'pie', 'scatter', 'area', 'stacked_bar', 'stacked_area', 'horizontal_bar'];
  const valueFormats = ['auto', 'number', 'currency', 'percent', 'abbreviated'];
  const sortModes = ['none', 'desc', 'asc'];

  dom.body.innerHTML = `
    <div class="studio-chart-editor">
      <div class="studio-chart-config" id="studio-chart-config">
        <label>Chart Type</label>
        <select id="studio-chart-type">
          ${chartTypes.map(t => `<option value="${t}"${t === state.chartConfig.chart_type ? ' selected' : ''}>${escapeHtml(t.replace(/_/g, ' '))}</option>`).join('')}
        </select>
        <label>Title</label>
        <input id="studio-chart-title" type="text" value="${escapeHtml(state.chartConfig.title)}">
        <label>Subtitle</label>
        <input id="studio-chart-subtitle" type="text" value="${escapeHtml(state.chartConfig.subtitle)}" placeholder="Optional context / timeframe">
        <label>X-Axis Label</label>
        <input id="studio-chart-x-label" type="text" value="${escapeHtml(state.chartConfig.x_label)}">
        <label>Y-Axis Label</label>
        <input id="studio-chart-y-label" type="text" value="${escapeHtml(state.chartConfig.y_label)}" placeholder="Include units e.g. Revenue ($M)">
        <label>Number format</label>
        <select id="studio-chart-value-format">
          ${valueFormats.map(f => `<option value="${f}"${f === state.chartConfig.value_format ? ' selected' : ''}>${escapeHtml(f)}</option>`).join('')}
        </select>
        <label>Sort</label>
        <select id="studio-chart-sort">
          ${sortModes.map(s => `<option value="${s}"${s === state.chartConfig.sort ? ' selected' : ''}>${escapeHtml(s)}</option>`).join('')}
        </select>
        <label><input id="studio-chart-show-values" type="checkbox" ${state.chartConfig.show_values ? 'checked' : ''}> Show data values</label>
        <div style="margin-top:var(--space-lg);border-top:1px solid var(--border);padding-top:var(--space-md)">
          <label style="margin-top:0">Data</label>
          <div id="studio-chart-data-table"></div>
          <div class="studio-chart-data-actions">
            <button class="studio-slide-add-btn" id="studio-chart-add-row">+ Add Row</button>
            <button class="studio-slide-add-btn" id="studio-chart-add-series">+ Add Series</button>
          </div>
        </div>
      </div>
      <div class="studio-chart-preview-area">
        <div class="studio-chart-canvas-wrap">
          <canvas id="studio-chart-canvas"></canvas>
        </div>
        <div class="studio-ai-blocks" id="studio-ai-blocks" style="padding:var(--space-md)"></div>
      </div>
    </div>
  `;

  dom.toolbar.innerHTML = `<span style="font-size:var(--text-xs);color:var(--text-muted)">Chart Editor</span>`;
  buildAiPopover('chart');

  // Load Chart.js
  const loaded = await ensureChartJsLoaded();

  // Wire config change handlers
  const debounceRender = debounce(() => { updateChartConfig(); renderChartPreview(); }, 300);

  ['studio-chart-type', 'studio-chart-title', 'studio-chart-subtitle', 'studio-chart-x-label', 'studio-chart-y-label'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', debounceRender);
  });
  ['studio-chart-show-values', 'studio-chart-value-format', 'studio-chart-sort'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', debounceRender);
  });
  document.getElementById('studio-chart-add-row')?.addEventListener('click', () => {
    state.chartConfig.labels.push('');
    state.chartConfig.datasets.forEach(d => d.values.push(0));
    markDirty();
    renderChartDataTable();
    renderChartPreview();
  });
  document.getElementById('studio-chart-add-series')?.addEventListener('click', _chartAddSeries);

  // Render initial state
  renderChartDataTable();
  if (loaded) renderChartPreview();

  state.theme = state.source?.theme?.preset || state.source?.theme || 'slate';
  loadThemes();
}

function debounce(fn, ms) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

function updateChartConfig() {
  const cfg = state.chartConfig;
  cfg.chart_type = document.getElementById('studio-chart-type')?.value || 'bar';
  cfg.title = document.getElementById('studio-chart-title')?.value || '';
  cfg.subtitle = document.getElementById('studio-chart-subtitle')?.value || '';
  cfg.x_label = document.getElementById('studio-chart-x-label')?.value || '';
  cfg.y_label = document.getElementById('studio-chart-y-label')?.value || '';
  cfg.show_values = document.getElementById('studio-chart-show-values')?.checked || false;
  cfg.value_format = document.getElementById('studio-chart-value-format')?.value || 'auto';
  cfg.sort = document.getElementById('studio-chart-sort')?.value || 'none';
  markDirty();
}

function renderChartDataTable() {
  const container = document.getElementById('studio-chart-data-table');
  if (!container) return;

  const cfg = state.chartConfig;
  let html = '<table class="studio-grid-table studio-chart-data-table" style="font-size:var(--text-xs)">';

  // Header: Label | Series1 | Series2 ... Each series header is editable
  // (rename) and carries an × affordance when there's more than one series —
  // we never let the user delete the final series since a chart with zero
  // data series is indistinguishable from an empty canvas.
  html += '<thead><tr><th>Label</th>';
  cfg.datasets.forEach((ds, d) => {
    const canDelete = cfg.datasets.length > 1;
    html += `<th class="studio-chart-series-head" data-ds="${d}">
      <span class="studio-chart-series-name" contenteditable="true" data-field="series-name" data-ds="${d}" spellcheck="false">${escapeHtml(ds.name || `Series ${d + 1}`)}</span>
      ${canDelete ? `<button class="studio-chart-series-del" data-ds="${d}" title="Delete series" aria-label="Delete series">&times;</button>` : ''}
    </th>`;
  });
  html += '</tr></thead><tbody>';

  for (let r = 0; r < cfg.labels.length; r++) {
    html += '<tr>';
    html += `<td contenteditable="true" data-field="label" data-row="${r}">${escapeHtml(String(cfg.labels[r] || ''))}</td>`;
    for (let d = 0; d < cfg.datasets.length; d++) {
      const val = cfg.datasets[d].values[r] ?? 0;
      html += `<td contenteditable="true" data-field="value" data-row="${r}" data-ds="${d}">${escapeHtml(String(val))}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';

  container.innerHTML = html;

  // Wire label / value cells
  container.querySelectorAll('td[contenteditable]').forEach(cell => {
    cell.addEventListener('blur', () => {
      const row = parseInt(cell.dataset.row);
      const field = cell.dataset.field;
      const val = cell.textContent.trim();

      if (field === 'label') {
        cfg.labels[row] = val;
      } else if (field === 'value') {
        const ds = parseInt(cell.dataset.ds);
        const num = parseFloat(val);
        cfg.datasets[ds].values[row] = isNaN(num) ? 0 : num;
      }
      markDirty();
      renderChartPreview();
    });
  });

  // Series header inline rename — commit on blur. Empty names fall back to
  // "Series N" so the chart legend never renders a blank entry.
  container.querySelectorAll('.studio-chart-series-name').forEach(span => {
    span.addEventListener('blur', () => {
      const d = parseInt(span.dataset.ds);
      const name = span.textContent.trim();
      cfg.datasets[d].name = name || `Series ${d + 1}`;
      if (!name) span.textContent = cfg.datasets[d].name;
      markDirty();
      renderChartPreview();
    });
    span.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); span.blur(); }
    });
  });

  // Series delete buttons. Confirm before dropping — values are gone once
  // the dataset is spliced out of the config array.
  container.querySelectorAll('.studio-chart-series-del').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const d = parseInt(btn.dataset.ds);
      _chartRemoveSeries(d);
    });
  });
}

function _chartAddSeries() {
  const cfg = state.chartConfig;
  if (!cfg) return;
  const name = `Series ${cfg.datasets.length + 1}`;
  // Initialize with zeros so the table renders uniformly. Users typically
  // paste real numbers in immediately after adding.
  cfg.datasets.push({ name, values: cfg.labels.map(() => 0) });
  markDirty();
  renderChartDataTable();
  renderChartPreview();
}

function _chartRemoveSeries(d) {
  const cfg = state.chartConfig;
  if (!cfg || cfg.datasets.length <= 1) return;
  const ds = cfg.datasets[d];
  if (!ds) return;
  const name = ds.name || `Series ${d + 1}`;
  if (!confirm(`Delete series "${name}"? Its data will be lost.`)) return;
  cfg.datasets.splice(d, 1);
  markDirty();
  renderChartDataTable();
  renderChartPreview();
}

function _hexToRgba(hex, alpha) {
  hex = hex.trim().replace('#', '');
  if (hex.length !== 6) return `rgba(99,102,241,${alpha})`;
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// Build a 6-color palette anchored on the accent. Accent stays first so
// single-series charts read as "the app's color"; the rest are HSL hue
// rotations (+45°, +90°, ...) tuned to land on visually distinct hues
// without going garish. Used by _getChartColors when no theme picker is
// populated (the chart editor hides it).
function _paletteFromAccent(accent) {
  const hex = String(accent).trim().replace('#', '');
  if (hex.length !== 6) return [_hexToRgba(accent || '#6c8aff', 0.7)];
  const r = parseInt(hex.substring(0, 2), 16) / 255;
  const g = parseInt(hex.substring(2, 4), 16) / 255;
  const b = parseInt(hex.substring(4, 6), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  let h = 0, s = 0;
  if (d > 0) {
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)) * 60; break;
      case g: h = ((b - r) / d + 2) * 60; break;
      case b: h = ((r - g) / d + 4) * 60; break;
    }
  }
  const offsets = [0, 45, 90, 200, 270, 330];
  return offsets.map(off => {
    const hh = (h + off) % 360;
    return `hsla(${hh.toFixed(0)},${(s * 100).toFixed(0)}%,${(l * 100).toFixed(0)}%,0.78)`;
  });
}

function _getChartColors() {
  // Default colors if no theme loaded
  const defaults = [
    'rgba(99,102,241,0.7)', 'rgba(239,68,68,0.7)', 'rgba(34,197,94,0.7)',
    'rgba(249,115,22,0.7)', 'rgba(168,85,247,0.7)', 'rgba(59,130,246,0.7)',
  ];

  // C2 fix: derive the palette from the document's --accent CSS variable
  // so the chart matches the app theme even before the themes picker has
  // populated. The artifact theme picker (slate/corporate/emerald/...) is
  // hidden in the chart editor; without this read, every chart rendered
  // a generic Chart.js purple regardless of theme.
  try {
    const root = document.documentElement;
    const cs = getComputedStyle(root);
    const accent = (cs.getPropertyValue('--accent') || '').trim();
    if (accent) {
      return _paletteFromAccent(accent);
    }
  } catch { /* fall through to defaults */ }

  // Try to find the current theme from the loaded themes picker
  const activeTheme = dom.themePicker?.querySelector('.studio-theme-option.active');
  if (!activeTheme) return defaults;

  // Get the accent color swatches
  const swatches = activeTheme.querySelectorAll('.studio-theme-swatch-color');
  if (swatches.length < 3) return defaults;

  // Extract colors from swatch backgrounds
  const accent = swatches[0]?.style.background || '';
  const dark = swatches[1]?.style.background || '';
  const light = swatches[2]?.style.background || '';

  if (!accent) return defaults;

  // Build a palette: accent as primary, then generate variants
  // Use the accent + complementary colors
  return [
    _hexToRgba(accent, 0.7),
    _hexToRgba(dark, 0.7),
    'rgba(239,68,68,0.7)',    // red
    'rgba(34,197,94,0.7)',    // green
    'rgba(249,115,22,0.7)',   // orange
    _hexToRgba(light, 0.5),
  ];
}

// Mirror of the backend _make_value_formatter (artifact_chart.py) so the
// live Chart.js preview formats numbers identically to the rendered PNG:
// $142K / 45% / 1.2M / 1,234. Auto-detects from the y/x axis labels.
function _studioChartFormatter(cfg) {
  const flat = [];
  (cfg.datasets || []).forEach(ds => (ds.values || []).forEach(v => {
    const n = Number(v); if (isFinite(n)) flat.push(n);
  }));
  const maxAbs = flat.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  let fmt = cfg.value_format || 'auto';
  if (fmt === 'auto') {
    const text = `${cfg.y_label || ''} ${cfg.x_label || ''}`.toLowerCase();
    const pct = ['%', 'percent', 'rate', 'share', 'ratio', 'growth', 'margin', 'ctr'];
    const cur = ['$', '£', '€', 'usd', 'eur', 'gbp', 'revenue', 'cost', 'price', 'sales', 'budget', 'profit', 'income', 'expense', 'dollar'];
    if (pct.some(h => text.includes(h))) fmt = 'percent';
    else if (cur.some(h => text.includes(h))) fmt = 'currency';
    else if (maxAbs >= 10000) fmt = 'abbreviated';
    else fmt = 'number';
  }
  const pctScale = (flat.length && maxAbs <= 1.5) ? 100 : 1;
  const abbrev = (v) => {
    const a = Math.abs(v);
    for (const [d, s] of [[1e9, 'B'], [1e6, 'M'], [1e3, 'K']]) {
      if (a >= d) return `${(+(v / d).toFixed(1))}${s}`;
    }
    return Number.isInteger(v) ? v.toLocaleString('en-US') : `${+v.toFixed(1)}`;
  };
  return (v) => {
    const n = Number(v);
    if (!isFinite(n)) return String(v ?? '');
    if (fmt === 'percent') { const p = n * pctScale; return Number.isInteger(p) ? `${p}%` : `${p.toFixed(1)}%`; }
    if (fmt === 'abbreviated') return abbrev(n);
    if (fmt === 'currency') {
      if (Math.abs(n) >= 10000) return (n < 0 ? '-$' : '$') + abbrev(Math.abs(n));
      return (n < 0 ? '-$' : '$') + Math.abs(n).toLocaleString('en-US', { maximumFractionDigits: 2 });
    }
    return n.toLocaleString('en-US', { maximumFractionDigits: 2 });
  };
}

// Mirror of the backend sort + pie "Other"-grouping so the editable preview
// matches the PNG. Returns reordered {labels, datasets}.
function _studioChartDisplayData(cfg) {
  let labels = [...(cfg.labels || [])];
  let datasets = (cfg.datasets || []).map(ds => ({ name: ds.name, values: [...(ds.values || [])] }));
  const single = datasets.length === 1;
  if (single && (cfg.sort === 'asc' || cfg.sort === 'desc') &&
      ['bar', 'horizontal_bar', 'pie'].includes(cfg.chart_type)) {
    const vals = datasets[0].values;
    const order = labels.map((_, i) => i).sort((a, b) =>
      cfg.sort === 'desc' ? (Number(vals[b]) || 0) - (Number(vals[a]) || 0)
        : (Number(vals[a]) || 0) - (Number(vals[b]) || 0));
    labels = order.map(i => labels[i]);
    datasets = [{ name: datasets[0].name, values: order.map(i => vals[i]) }];
  }
  if (cfg.chart_type === 'pie' && single) {
    let pairs = labels.map((l, i) => [l, Number(datasets[0].values[i]) || 0]).filter(p => p[1] > 0);
    pairs.sort((a, b) => b[1] - a[1]);
    const MAX = 6;
    if (pairs.length > MAX) {
      const head = pairs.slice(0, MAX - 1);
      const other = pairs.slice(MAX - 1).reduce((s, p) => s + p[1], 0);
      pairs = [...head, ['Other', other]];
    }
    labels = pairs.map(p => p[0]);
    datasets = [{ name: datasets[0].name, values: pairs.map(p => p[1]) }];
  }
  return { labels, datasets };
}

function renderChartPreview() {
  if (!ChartJS) return;
  const canvas = document.getElementById('studio-chart-canvas');
  if (!canvas) return;

  // Destroy previous instance
  if (state.chartInstance) {
    state.chartInstance.destroy();
    state.chartInstance = null;
  }

  const cfg = state.chartConfig;

  // Empty-state: a chart with no labels or no numeric values draws a blank
  // canvas with no explanation (the "shows empty to the user" bug). Swap in
  // a placeholder that points at the data table instead.
  const _hasData = (cfg.labels || []).length > 0 && (cfg.datasets || []).some(
    ds => (ds.values || []).some(v => typeof v === 'number' && isFinite(v)),
  );
  const _wrap = canvas.parentElement;
  let _emptyEl = _wrap?.querySelector('.studio-chart-empty');
  if (!_hasData) {
    canvas.style.display = 'none';
    if (!_emptyEl && _wrap) {
      _emptyEl = document.createElement('div');
      _emptyEl.className = 'studio-chart-empty';
      _emptyEl.style.cssText = 'padding:48px;text-align:center;color:var(--text-muted);font-size:var(--text-sm)';
      _wrap.appendChild(_emptyEl);
    }
    if (_emptyEl) _emptyEl.textContent = 'No data to chart yet — add labels and values in the table on the left.';
    return;
  }
  canvas.style.display = '';
  if (_emptyEl) _emptyEl.remove();

  const chartTypeMap = {
    bar: 'bar', line: 'line', pie: 'pie', scatter: 'scatter',
    area: 'line', stacked_bar: 'bar', stacked_area: 'line', horizontal_bar: 'bar',
  };
  const type = chartTypeMap[cfg.chart_type] || 'bar';

  // Generate theme-aware colors from the accent
  const themeColors = _getChartColors();

  // Apply the same sort / pie "Other"-grouping the backend does, and build a
  // matching number formatter, so the preview reads like the rendered PNG.
  const disp = _studioChartDisplayData(cfg);
  const fmtValue = _studioChartFormatter(cfg);

  const datasets = disp.datasets.map((ds, i) => ({
    label: ds.name || `Series ${i + 1}`,
    data: ds.values,
    backgroundColor: type === 'pie' ? themeColors : themeColors[i % themeColors.length],
    borderColor: type === 'line' ? themeColors[i % themeColors.length] : undefined,
    fill: cfg.chart_type === 'area' || cfg.chart_type === 'stacked_area',
  }));

  // When "Show data values" is on, paint formatted values above bars / on
  // points / inside slices (matches the PNG: $142K, 45%, 1.2M).
  const showValues = !!cfg.show_values;
  const datalabelsOpts = showValues ? {
    display: true,
    color: '#1f2937',
    font: { weight: '600', size: 11 },
    anchor: type === 'pie' ? 'center' : 'end',
    align: type === 'pie' ? 'center' : 'top',
    offset: type === 'pie' ? 0 : 4,
    formatter: (v) => {
      if (v == null || v === '') return '';
      const n = Number(v);
      return isFinite(n) ? fmtValue(n) : String(v);
    },
  } : { display: false };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      title: { display: !!cfg.title, text: cfg.title },
      subtitle: { display: !!cfg.subtitle, text: cfg.subtitle, color: '#6b7280', font: { size: 12 }, padding: { bottom: 8 } },
      legend: { display: disp.datasets.length > 1 },
      datalabels: datalabelsOpts,
    },
    scales: type === 'pie' ? {} : {
      x: {
        title: { display: !!cfg.x_label, text: cfg.x_label },
        stacked: cfg.chart_type.startsWith('stacked'),
        // Value axis is x only for horizontal bars; format ticks there.
        ...(cfg.chart_type === 'horizontal_bar' ? { ticks: { callback: (val) => fmtValue(val) } } : {}),
      },
      y: {
        title: { display: !!cfg.y_label, text: cfg.y_label },
        stacked: cfg.chart_type.startsWith('stacked'),
        ...(cfg.chart_type !== 'horizontal_bar' ? { ticks: { callback: (val) => fmtValue(val) } } : {}),
      },
    },
    indexAxis: cfg.chart_type === 'horizontal_bar' ? 'y' : 'x',
  };

  const plugins = (showValues && ChartDataLabels) ? [ChartDataLabels] : [];
  state.chartInstance = new ChartJS(canvas, {
    type,
    data: { labels: disp.labels, datasets },
    options,
    plugins,
  });
}

function getChartSource() {
  updateChartConfig();
  return {
    ...state.source,
    ...state.chartConfig,
  };
}

async function getChartCanvasBlob() {
  const canvas = document.getElementById('studio-chart-canvas');
  if (!canvas) return null;
  return new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
}

// ---------------------------------------------------------------------------
// Milkdown Editor (shared with notes — same CDN load pattern)
// ---------------------------------------------------------------------------
let CrepeClass = null;
let _milkdownLoaded = false;
let _milkdownLoading = false;

async function ensureMilkdownLoaded() {
  if (_milkdownLoaded) return true;
  if (_milkdownLoading) {
    while (_milkdownLoading) await new Promise(r => setTimeout(r, 50));
    return _milkdownLoaded;
  }
  _milkdownLoading = true;
  try {
    if (!document.getElementById('milkdown-css-common')) {
      const c1 = document.createElement('link');
      c1.id = 'milkdown-css-common';
      c1.rel = 'stylesheet';
      c1.href = '/ui/lib/milkdown/common-bundle.css';
      document.head.appendChild(c1);
      const c2 = document.createElement('link');
      c2.id = 'milkdown-css-theme';
      c2.rel = 'stylesheet';
      c2.href = '/ui/lib/milkdown/frame.css';
      document.head.appendChild(c2);
      const c3 = document.createElement('link');
      c3.id = 'milkdown-css-prosemirror';
      c3.rel = 'stylesheet';
      c3.href = '/ui/lib/milkdown/prosemirror.css';
      document.head.appendChild(c3);
    }
    // Pinned to exact version — prevents silent breaking changes from CDN.
    // CSS is vendored locally (ui/lib/milkdown/). Use esm.sh, NOT
    // jsdelivr's `+esm` — the latter ships duplicated prosemirror-state
    // instances across transitive deps, triggering a keyed-plugin
    // collision in EditorState.create. See browse.js ensureMilkdownLoaded
    // for the full reasoning. Must match the URL there so both surfaces
    // hit the same module-cache entry.
    const mod = await import('https://esm.sh/@milkdown/crepe@7.19.2?bundle-deps');
    CrepeClass = mod.Crepe;
    _milkdownLoaded = true;
  } catch (err) {
    console.error('Failed to load Milkdown:', err);
  }
  _milkdownLoading = false;
  return _milkdownLoaded;
}

async function _destroyEditor() {
  if (!state.editor) return;
  const ed = state.editor;
  state.editor = null;
  try {
    if (typeof ed.destroy === 'function') await ed.destroy();
  } catch { /* ignore */ }
}

function _fallbackTextarea(container, markdown) {
  container.innerHTML = '';
  const ta = document.createElement('textarea');
  ta.style.cssText = 'width:100%;height:100%;border:none;background:transparent;color:var(--text-primary);font-family:"Crimson Pro",Georgia,serif;font-size:1.0625rem;line-height:1.75;resize:none;outline:none;';
  ta.value = markdown;
  ta.addEventListener('input', markDirty);
  container.appendChild(ta);
  state.editor = { _textarea: ta };
}

async function loadMilkdownEditor(container, markdown) {
  const loaded = await ensureMilkdownLoaded();
  if (!loaded) {
    _fallbackTextarea(container, markdown);
    return;
  }

  await _destroyEditor();
  container.innerHTML = '';

  try {
    const root = document.createElement('div');
    container.appendChild(root);
    const crepe = new CrepeClass({
      root,
      defaultValue: markdown || '',
      features: {
        [CrepeClass.Feature?.CodeMirror ?? 'code-mirror']: false,
        [CrepeClass.Feature?.Latex ?? 'latex']: false,
      },
    });
    crepe.on((listener) => {
      listener.markdownUpdated(() => markDirty());
    });
    await crepe.create();
    state.editor = crepe;
  } catch (err) {
    console.error('Milkdown init failed:', err);
    _fallbackTextarea(container, markdown);
  }
}

function getEditorMarkdown() {
  if (!state.editor) return '';
  if (state.editor._textarea) return state.editor._textarea.value;
  try {
    if (typeof state.editor.getMarkdown === 'function') return state.editor.getMarkdown();
  } catch { /* ignore */ }
  const el = document.querySelector('#studio-doc-page .editor, #studio-doc-page .ProseMirror, #studio-doc-page [contenteditable]');
  return el?.textContent || '';
}

// ---------------------------------------------------------------------------
// Theme Picker
// ---------------------------------------------------------------------------
// EPUB reading themes. These are *book* themes (real page background +
// typography) — distinct from the white-paper business palettes served by
// /api/studio/themes/list. Names MUST match _EPUB_THEMES in
// augmentum/tools/artifact_ebook.py, which does the actual render; the
// swatch trio here is just (accent, page-bg, edge) for the picker preview.
const _EPUB_THEME_OPTIONS = [
  { name: 'storybook', accent: '#8b7355', accent_dark: '#faf9f6', accent_light: '#d4c5a9' },
  { name: 'paper',     accent: '#2563eb', accent_dark: '#ffffff', accent_light: '#e5e7eb' },
  { name: 'sepia',     accent: '#9a6b3f', accent_dark: '#f4ecd8', accent_light: '#ddccab' },
  { name: 'slate',     accent: '#3b6ea5', accent_dark: '#f5f6f8', accent_light: '#d7dce2' },
  { name: 'night',     accent: '#7ea6d8', accent_dark: '#16181c', accent_light: '#2c3036' },
  { name: 'midnight',  accent: '#8aa6d6', accent_dark: '#0e1320', accent_light: '#232b3d' },
];

// Per-book reading-comfort controls layered on top of the theme. Keys MUST
// match _EPUB_FONT_STACKS / _EPUB_SIZE_EM / _EPUB_LEADING in
// augmentum/tools/artifact_ebook.py — those do the actual render. ''/'md'/
// 'normal' are the "leave it as the theme intended" defaults.
const _EPUB_READING_ROWS = [
  { group: 'font', label: 'Font', def: '', options: [
    { key: '', label: 'Auto' }, { key: 'serif', label: 'Serif' },
    { key: 'sans', label: 'Sans' }, { key: 'dyslexic', label: 'Reader' },
  ] },
  { group: 'size', label: 'Size', def: 'md', options: [
    { key: 'xs', label: 'A', style: 'font-size:10px' },
    { key: 'sm', label: 'A', style: 'font-size:12px' },
    { key: 'md', label: 'A', style: 'font-size:14px' },
    { key: 'lg', label: 'A', style: 'font-size:16px' },
    { key: 'xl', label: 'A', style: 'font-size:18px' },
  ] },
  { group: 'leading', label: 'Spacing', def: 'normal', options: [
    { key: 'compact', label: 'Tight' }, { key: 'normal', label: 'Normal' },
    { key: 'relaxed', label: 'Roomy' },
  ] },
];

// Current reading settings off the open ebook source, with defaults filled.
function _ebookReading() {
  const r = (state.source && typeof state.source.reading === 'object' && state.source.reading) || {};
  return { font: r.font || '', size: r.size || 'md', leading: r.leading || 'normal' };
}

function _readingRowHtml(row, active) {
  const segs = row.options.map((o) =>
    `<button type="button" class="studio-reading-seg${o.key === active ? ' active' : ''}"`
    + ` data-reading-group="${escapeHtml(row.group)}" data-reading-value="${escapeHtml(o.key)}"`
    + (o.style ? ` style="${escapeHtml(o.style)}"` : '')
    + `>${escapeHtml(o.label)}</button>`,
  ).join('');
  return `<div class="studio-reading-row"><span class="studio-reading-label">${escapeHtml(row.label)}</span>`
    + `<div class="studio-reading-segs">${segs}</div></div>`;
}

// Theme cache populated by loadThemes(). Keyed by name → full theme palette
// from /api/studio/themes/list. Used by _applyArtifactTheme() so the editor
// surface mirrors what the PDF/DOCX renderer will produce.
let _themeCache = null;

const _DOC_LH_MULT = { tight: 0.85, comfortable: 1.0, airy: 1.2 };
const _DOC_DENSITY_MULT = { compact: 0.8, default: 1.0, spacious: 1.25 };
const _DOC_FONT_STACKS = {
  system: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  sans: '"Helvetica Neue", Helvetica, Arial, sans-serif',
  serif: 'Georgia, "Times New Roman", Times, serif',
  mono: '"SF Mono", Consolas, "Liberation Mono", monospace',
  dyslexic: '"OpenDyslexic", "Comic Sans MS", system-ui, sans-serif',
};

// Mirror the renderer's design application onto the editor's .studio-doc-page
// so the WYSIWYG surface matches the PDF/DOCX preview. No-op when the page
// DOM isn't mounted yet OR themes haven't loaded yet — both call sites
// (page setup, theme fetch) re-trigger this so it settles after both land.
function _applyArtifactTheme() {
  const pageEl = document.getElementById('studio-doc-page');
  if (!pageEl) return;
  if (!_themeCache) return;
  const theme = _themeCache[state.theme] || _themeCache.slate || _themeCache[Object.keys(_themeCache)[0]];
  if (!theme) return;
  const design = _getStudioDesign();
  const accent = design.accent_override || theme.accent;
  const fontFam = _DOC_FONT_STACKS[design.font_family] || _DOC_FONT_STACKS.system;
  const sizeScale = Number(design.font_size_scale) || 1;
  const lhMult = _DOC_LH_MULT[design.line_height] ?? 1;
  const densityMult = _DOC_DENSITY_MULT[design.density] ?? 1;
  const s = pageEl.style;
  s.setProperty('--doc-bg', theme.background);
  s.setProperty('--doc-surface', theme.surface);
  s.setProperty('--doc-border', theme.border);
  s.setProperty('--doc-text', theme.text);
  s.setProperty('--doc-text-secondary', theme.text_secondary);
  s.setProperty('--doc-text-muted', theme.text_muted);
  s.setProperty('--doc-accent', accent);
  s.setProperty('--doc-accent-light', theme.accent_light);
  s.setProperty('--doc-accent-dark', theme.accent_dark);
  s.setProperty('--doc-font', fontFam);
  s.setProperty('--doc-font-size', `${(1.05 * sizeScale).toFixed(3)}rem`);
  s.setProperty('--doc-line-height', (1.75 * lhMult).toFixed(3));
  s.setProperty('--doc-page-pad-block', `${Math.round(48 * densityMult)}px`);
  s.setProperty('--doc-page-pad-inline', `${Math.round(64 * densityMult)}px`);
}

async function loadThemes() {
  // Ebooks get their own reading-theme list (no server round-trip needed).
  if (state.sourceType === 'ebook') { renderThemePicker(_EPUB_THEME_OPTIONS); return; }
  try {
    const resp = await fetch('/api/studio/themes/list');
    if (!resp.ok) return;
    const data = await resp.json();
    const themes = data.themes || [];
    _themeCache = {};
    for (const t of themes) _themeCache[t.name] = t;
    renderThemePicker(themes);
    _applyArtifactTheme();
  } catch { /* ignore */ }
}

function renderThemePicker(themes) {
  if (!dom.themePicker) return;

  const label = state.sourceType === 'ebook' ? 'EPUB Theme'
    : state.sourceType === 'presentation' ? 'Deck Theme'
    : state.sourceType === 'spreadsheet' ? 'Sheet Theme'
    : state.sourceType === 'chart' ? 'Chart Theme'
    : 'Document Theme';
  let html = `<div class="studio-theme-picker-label">${escapeHtml(label)}</div>`;
  for (const t of themes) {
    const isActive = t.name === state.theme;
    html += `<button class="studio-theme-option${isActive ? ' active' : ''}" data-theme="${escapeHtml(t.name)}">
      <div class="studio-theme-swatch">
        <div class="studio-theme-swatch-color" style="background:${escapeHtml(t.accent)}"></div>
        <div class="studio-theme-swatch-color" style="background:${escapeHtml(t.accent_dark)}"></div>
        <div class="studio-theme-swatch-color" style="background:${escapeHtml(t.accent_light)}"></div>
      </div>
      <span class="studio-theme-name">${escapeHtml(t.name)}</span>
    </button>`;
  }

  // Ebooks also get reading-comfort controls (font / size / line spacing)
  // right in the same popover — they layer on top of whichever theme is
  // active and re-render the EPUB the same way the theme does.
  if (state.sourceType === 'ebook') {
    const r = _ebookReading();
    html += `<div class="studio-theme-picker-label studio-reading-heading">Reading</div>`;
    html += _readingRowHtml(_EPUB_READING_ROWS[0], r.font);
    html += _readingRowHtml(_EPUB_READING_ROWS[1], r.size);
    html += _readingRowHtml(_EPUB_READING_ROWS[2], r.leading);
  }

  dom.themePicker.innerHTML = html;

  // Wire clicks
  dom.themePicker.querySelectorAll('.studio-theme-option').forEach(btn => {
    btn.addEventListener('click', () => {
      const name = btn.dataset.theme;
      if (name) selectTheme(name);
      dom.themePicker.classList.add('hidden');
    });
  });
  // Reading segments: don't close the popover — the user often nudges size
  // then spacing back-to-back.
  dom.themePicker.querySelectorAll('.studio-reading-seg').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      selectReading(btn.dataset.readingGroup, btn.dataset.readingValue);
    });
  });
}

function selectReading(group, value) {
  if (!group || !state.source) return;
  if (!state.source.reading || typeof state.source.reading !== 'object') state.source.reading = {};
  state.source.reading[group] = value;
  markDirty();
  loadThemes();  // re-render so the active segment updates
  _applyArtifactTheme();
  _flushPendingSave().then(() => {
    if (state.viewMode === 'overview') _reloadOverviewIframe();
  });
}

function selectTheme(name) {
  state.theme = name;
  // Update source
  if (state.source) {
    if (!state.source.theme || typeof state.source.theme === 'string') {
      state.source.theme = { preset: name };
    } else {
      state.source.theme.preset = name;
    }
    // Keep the canonical design block in sync — the new Design tool reads
    // from source.design first, so writing through here means the popover
    // and the palette stay aligned.
    if (!state.source.design || typeof state.source.design !== 'object') {
      state.source.design = { ..._DESIGN_DEFAULTS };
    }
    state.source.design.theme = name;
  }
  markDirty();
  // Re-render theme picker to show active state
  loadThemes();
  // Mirror the new theme onto the editor surface immediately — loadThemes()
  // is async, so this synchronous apply keeps the editor in sync even before
  // the (already-cached) theme list returns.
  _applyArtifactTheme();
  // Notify the active design tool (if any) so its swatches reflect the change.
  if (_activePalette?.activeId === 'design') {
    _activePalette.setCtx({ getDesign: _getStudioDesign });
  }
  // Refresh preview if open
  if (state.previewOpen) refreshPreview();
  // Persist immediately rather than waiting on the debounced autosave —
  // /api/studio/{id}/save re-renders the artifact binary from source, so
  // the new theme is on disk the moment it's picked. When the read-only
  // Overview is up (ebook), reload its iframe afterwards so the user sees
  // the re-render in place without bouncing back to the editor.
  _flushPendingSave().then(() => {
    if (state.viewMode === 'overview') _reloadOverviewIframe();
  });
}

// Bust the Overview preview iframe so it re-fetches the (just re-rendered)
// artifact. No-op when the Overview isn't mounted.
function _reloadOverviewIframe() {
  const iframe = dom.body?.querySelector('.studio-overview-iframe');
  if (!iframe) return;
  const base = (iframe.getAttribute('src') || '').split('?')[0];
  if (base) iframe.src = `${base}?v=${Date.now()}`;
}

// ---------------------------------------------------------------------------
// Live Preview
// ---------------------------------------------------------------------------
// Word count + reading-time badge. Shared across any editor that wants it.
// _wordCountInstall(getText) wires a debounced recompute; callers fire it
// manually from their input listener or on every render to keep it current.
// Reading time uses 220 wpm — a tick above the common 200-250 range that
// reads as "brisk but realistic" and matches Medium's heuristic.
function _wordCountInstall(getText) {
  const el = document.getElementById('studio-word-count');
  if (!el) return;
  const compute = () => {
    const text = (getText() || '').trim();
    const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
    const mins = words > 0 ? Math.max(1, Math.round(words / 220)) : 0;
    el.textContent = words === 0
      ? ''
      : `${words.toLocaleString()} word${words === 1 ? '' : 's'} · ${mins} min read`;
  };
  compute();
  state._wordCountCompute = compute;
  clearInterval(state._wordCountTicker);
  state._wordCountTicker = setInterval(compute, 1500);
}

// Outline panel — auto-generated from the Milkdown document's live H1-H5
// elements. Clicking a row scrolls the matching heading into view. We
// re-render on a 1.5s tick while the editor is active so typing surfaces
// new sections without a manual refresh.
function _toggleOutline() {
  const panel = document.getElementById('studio-outline-panel');
  if (!panel) return;
  panel.classList.toggle('hidden');
  const visible = !panel.classList.contains('hidden');
  panel.setAttribute('aria-hidden', visible ? 'false' : 'true');
  // Call the renderer in both directions: opening builds the list + arms
  // the ticker, closing falls through the early-return that clears it.
  _renderOutline();
}

// Read-side counterpart of _toggleSessionSidebar: apply the persisted
// collapse preference when an editor mounts. Called from openDocumentEditor
// (and any other session-shell editor). Missing this definition threw a
// ReferenceError that aborted openDocumentEditor before Milkdown loaded.
function _restoreSessionSidebar() {
  const { layout } = _getSessionLayout();
  if (!layout) return;
  let collapsed = false;
  try { collapsed = localStorage.getItem('studio.session-panel-collapsed') === '1'; } catch (_) {}
  layout.dataset.panelCollapsed = collapsed ? 'true' : 'false';
  const btn = document.getElementById('studio-sidebar-toggle-btn');
  if (btn) btn.classList.toggle('active', collapsed);
}

function _toggleSessionSidebar() {
  const { layout } = _getSessionLayout();
  if (!layout) return;
  const collapsed = layout.dataset.panelCollapsed === 'true';
  layout.dataset.panelCollapsed = collapsed ? 'false' : 'true';
  // Persist so reopening the editor retains the preference.
  try { localStorage.setItem('studio.session-panel-collapsed', collapsed ? '0' : '1'); } catch (_) {}
  // Update the button active state.
  const btn = document.getElementById('studio-sidebar-toggle-btn');
  if (btn) btn.classList.toggle('active', !collapsed);
}

function _renderOutline() {
  const list = document.getElementById('studio-outline-list');
  const panel = document.getElementById('studio-outline-panel');
  if (!list || !panel) return;
  // Hidden panel — tear down the ticker so we don't refresh work nobody sees.
  if (panel.classList.contains('hidden')) {
    clearInterval(state._outlineTicker);
    state._outlineTicker = null;
    return;
  }
  const root = document.querySelector('#studio-doc-page .ProseMirror, #studio-doc-page [contenteditable]');
  if (!root) { list.innerHTML = '<div class="studio-outline-empty">Document not ready</div>'; return; }
  const headings = root.querySelectorAll('h1, h2, h3, h4, h5');
  if (!headings.length) {
    list.innerHTML = '<div class="studio-outline-empty">No headings yet. Use Heading 1-4 to build an outline.</div>';
  } else {
    const items = [];
    // Always rewrite ids so insertions/deletions can't leave duplicate
    // studio-h-N values on separate headings (which would break
    // getElementById lookups inside click handlers).
    headings.forEach((h, i) => {
      const level = parseInt(h.tagName.substring(1), 10);
      const text = (h.textContent || '').trim() || '(empty)';
      h.id = `studio-h-${i}`;
      items.push({ level, text, id: h.id });
    });
    list.innerHTML = items.map(i =>
      `<a class="studio-outline-item" data-level="${i.level}" href="#${i.id}" data-target="${i.id}">${escapeHtml(i.text)}</a>`,
    ).join('');
    list.querySelectorAll('.studio-outline-item').forEach(a => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.getElementById(a.dataset.target);
        target?.scrollIntoView({ block: 'start', behavior: 'smooth' });
      });
    });
  }
  // Arm a single ticker while the panel is visible — idempotent since we
  // clear first, so repeated calls don't stack intervals.
  if (!state._outlineTicker) {
    state._outlineTicker = setInterval(() => {
      const p = document.getElementById('studio-outline-panel');
      if (!p || p.classList.contains('hidden')) {
        clearInterval(state._outlineTicker);
        state._outlineTicker = null;
        return;
      }
      _renderOutline();
    }, 1500);
  }
}

// Heading style picker — apply the chosen level to the current line by
// driving Milkdown's command pipeline when available, falling back to a
// document.execCommand('formatBlock') for the textarea fallback path.
function _applyHeadingStyle(level) {
  const root = document.querySelector('#studio-doc-page .ProseMirror, #studio-doc-page [contenteditable]');
  if (!root) return;
  root.focus();
  try {
    document.execCommand('formatBlock', false, level.toUpperCase());
  } catch { /* Best-effort — Milkdown's own headings should still be usable. */ }
  _renderOutline();
  markDirty();
}

// Scope key for the Find modal, resolved from state.sourceType. Returning a
// falsy value lets the browser's native Find-in-page take over instead.
function _findScopeForCurrentEditor() {
  const t = state.sourceType;
  if (t === 'document' || t === 'imported_text') return 'doc';
  if (t === 'presentation') return 'slides';
  if (t === 'spreadsheet')  return 'grid';
  if (t === 'csv')          return 'csv';
  return null;
}

async function _onListenClick(e) {
  const btn = e.currentTarget;
  const { readAloud, stopReadAloud } = await import('./read-aloud.js');
  // Toggle off if already playing from this button.
  if (btn.classList.contains('playing')) { stopReadAloud(); return; }
  // Prefer the live editor state so unsaved edits get read. Fall back to
  // the cached source_json for sourceType='document', or the raw text for
  // imported plain-text/markdown artifacts.
  let md = getEditorMarkdown();
  if (!md && state.sourceType === 'document') {
    md = sectionsToMarkdown(state.source);
  }
  if (!md || !md.trim()) {
    showToast('Nothing to read in this artifact.', 'warning');
    return;
  }
  const { ttsCleanText } = await import('./chat/tts.js');
  const text = ttsCleanText(md).trim();
  if (!text) {
    showToast('No readable text in this artifact.', 'warning');
    return;
  }
  await readAloud(text, btn);
}

function togglePreview() {
  state.previewOpen = !state.previewOpen;
  dom.previewBtn?.classList.toggle('active', state.previewOpen);

  const { layout } = _getSessionLayout();
  const host = _getSessionMainHost();
  const editor = host?.querySelector('.studio-doc-editor') || dom.body.querySelector('.studio-doc-editor');
  let previewPane = host?.querySelector('.studio-preview-pane') || dom.body.querySelector('.studio-preview-pane');

  // Hook the layout for L2: the workspace panel + tabs are CSS-hidden when
  // preview is open so the editor + preview share the full width instead of
  // competing with a third column.
  if (layout) layout.dataset.previewOpen = state.previewOpen ? 'true' : 'false';

  if (state.previewOpen) {
    // Create preview pane if it doesn't exist
    if (!previewPane) {
      previewPane = document.createElement('div');
      previewPane.className = 'studio-preview-pane';
      previewPane.innerHTML = `
        <div class="studio-preview-header">
          <span>Preview</span>
          <button class="studio-icon-btn" data-studio-close-preview="true" title="Close preview">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
        <div class="studio-preview-loading" id="studio-preview-loading">Rendering preview...</div>
        <iframe class="studio-preview-iframe" id="studio-preview-iframe" style="display:none"></iframe>
      `;
      previewPane.querySelector('[data-studio-close-preview="true"]')?.addEventListener('click', () => {
        state.previewOpen = false;
        dom.previewBtn?.classList.remove('active');
        previewPane?.classList.add('hidden');
        editor?.classList.remove('with-preview');
        host?.classList.remove('has-preview');
        if (layout) layout.dataset.previewOpen = 'false';
      });
      host.appendChild(previewPane);
    }

    previewPane.classList.remove('hidden');
    editor?.classList.add('with-preview');
    host?.classList.add('has-preview');

    refreshPreview();
  } else {
    previewPane?.classList.add('hidden');
    editor?.classList.remove('with-preview');
    host?.classList.remove('has-preview');
  }
}

async function refreshPreview() {
  if (!state.previewOpen || !state.artifactId || !state.source) return;

  const loading = document.getElementById('studio-preview-loading');
  const iframe = document.getElementById('studio-preview-iframe');
  if (loading) loading.style.display = '';
  if (iframe) iframe.style.display = 'none';

  // Build current source with theme
  let currentSource;
  if (state.sourceType === 'document') {
    const md = getEditorMarkdown();
    currentSource = markdownToSource(md);
    if (state.theme) {
      currentSource.theme = { preset: state.theme };
    }
  } else if (state.sourceType === 'presentation') {
    saveCurrentSlideState();
    currentSource = { ...state.source, slides: state.slides };
    if (state.theme) currentSource.theme = { preset: state.theme };
  } else {
    currentSource = state.source;
  }

  try {
    const resp = await fetch(`/api/studio/${state.artifactId}/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: currentSource }),
    });

    if (!resp.ok) {
      if (loading) loading.textContent = 'Preview failed';
      return;
    }

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);

    if (iframe) {
      iframe.src = url;
      iframe.style.display = '';
      iframe.onload = () => URL.revokeObjectURL(url);
    }
    if (loading) loading.style.display = 'none';
  } catch (err) {
    if (loading) loading.textContent = 'Preview error';
  }
}

// ---------------------------------------------------------------------------
// Dirty state + Save
// ---------------------------------------------------------------------------
// Force any pending debounced autosave to run now and wait for it. Safe to
// call when nothing is dirty (no-op). Used before re-entering openStudio for
// a mode/artifact switch so the just-made edit isn't lost when state resets.
async function _flushPendingSave() {
  clearTimeout(state._autosaveTimer);
  if (!state.dirty || state.saving) return;
  state._autosaving = true;
  try { await saveArtifact(); }
  catch { /* saveArtifact surfaces its own error path */ }
  finally { state._autosaving = false; }
}

function markDirty() {
  if (!state.dirty) {
    _updateDirtyState(true);
  }
  // Debounced preview refresh
  if (state.previewOpen) {
    clearTimeout(state.previewDebounce);
    state.previewDebounce = setTimeout(refreshPreview, 2000);
  }
  // Schedule an autosave 2s after the last edit. Each markDirty resets
  // the timer, so a burst of typing collapses into one save. PDF visual
  // editor handles its own save path and opts out.
  if (state.sourceType && state.sourceType !== 'pdf_visual') {
    clearTimeout(state._autosaveTimer);
    _setSaveStatus('dirty');
    state._autosaveTimer = setTimeout(async () => {
      if (!state.dirty || state.saving) return;
      // Flag so saveArtifact suppresses its success toast — the header chip
      // is the only confirmation we want for a silent debounced save.
      state._autosaving = true;
      try { await saveArtifact(); }
      catch { /* saveArtifact surfaces its own error path */ }
      finally { state._autosaving = false; }
      // In Overview the only editable thing is the theme — refresh the
      // preview iframe so a late-landing autosave is reflected too.
      if (state.viewMode === 'overview') _reloadOverviewIframe();
    }, 2000);
  }
}

// Saved-indicator state machine. States: idle, dirty, saving, saved.
// _savedAt drives the "Saved Xs ago" tick so the badge stays current
// without re-running on every keystroke.
function _setSaveStatus(status, timestamp) {
  state._saveStatus = status;
  if (status === 'saved') state._savedAt = timestamp || Date.now();
  _renderSaveStatus();
  // Kick the ticker: when saved, re-render every 5s so the "Xs ago"
  // text stays fresh. Clear on any state change.
  clearInterval(state._saveTicker);
  if (status === 'saved') {
    state._saveTicker = setInterval(_renderSaveStatus, 5000);
  }
}

function _renderSaveStatus() {
  const el = document.getElementById('studio-save-status');
  if (!el) return;
  const status = state._saveStatus || 'idle';
  if (status === 'idle') { el.textContent = ''; el.className = 'studio-save-status'; return; }
  if (status === 'dirty') {
    el.textContent = 'Unsaved';
    el.className = 'studio-save-status is-dirty';
    return;
  }
  if (status === 'saving') {
    el.textContent = 'Saving…';
    el.className = 'studio-save-status is-saving';
    return;
  }
  // saved — relative time
  const delta = Date.now() - (state._savedAt || Date.now());
  let label;
  if (delta < 5000)        label = 'Saved just now';
  else if (delta < 60_000) label = `Saved ${Math.round(delta / 1000)}s ago`;
  else if (delta < 3_600_000) label = `Saved ${Math.round(delta / 60_000)}m ago`;
  else                     label = `Saved ${Math.round(delta / 3_600_000)}h ago`;
  el.textContent = label;
  el.className = 'studio-save-status is-saved';
}

async function saveArtifact() {
  _setSaveStatus('saving');
  // CSV edits are serialized back to a full-file upload (no source_json).
  if (state.sourceType === 'csv') {
    state.saving = true;
    dom.saveBtn.textContent = 'Saving...';
    dom.saveBtn.disabled = true;
    try {
      const ok = await _saveCsvArtifact();
      if (ok) _setSaveStatus('saved');
      else    _setSaveStatus('dirty');
    }
    finally {
      state.saving = false;
      dom.saveBtn.textContent = 'Save';
      dom.saveBtn.disabled = false;
    }
    return;
  }
  // Visual PDF editor has its own save path
  if (state.sourceType === 'pdf_visual' && state._pdfGetBytes) {
    state.saving = true;
    dom.saveBtn.textContent = 'Saving...';
    dom.saveBtn.disabled = true;
    let pdfSaveOk = false;
    try {
      const bytes = await state._pdfGetBytes();
      pdfSaveOk = await _savePdfBytes(state.artifactId, bytes);
    } finally {
      state.saving = false;
      dom.saveBtn.textContent = 'Save';
      dom.saveBtn.disabled = false;
      // Honest save state — without this the chip stays stuck on "Saving…"
      // until the next edit. Matches the main path's success/dirty branching.
      _setSaveStatus(pdfSaveOk ? 'saved' : 'dirty');
    }
    return;
  }
  if (!state.artifactId || !state.source || state.saving) return;
  state.saving = true;
  dom.saveBtn.textContent = 'Saving...';
  dom.saveBtn.disabled = true;

  try {
    // Get current content and convert back to source
    let updatedSource;
    if (state.sourceType === 'document') {
      const md = getEditorMarkdown();
      updatedSource = markdownToSource(md);
    } else if (state.sourceType === 'presentation') {
      saveCurrentSlideState();
      updatedSource = { ...state.source, slides: state.slides };
    } else if (state.sourceType === 'spreadsheet') {
      updatedSource = getGridSource();
    } else if (state.sourceType === 'chart') {
      updatedSource = getChartSource();
    } else if (state.sourceType === 'ebook') {
      updatedSource = getEbookSource();
    } else {
      updatedSource = state.source;
    }

    // is_autosave gates server-side version snapshotting — autosaves are
    // every 2s while typing and would balloon the history table, so only
    // manual saves (user clicked Save / Ctrl+S) leave a version behind.
    const savePayload = { source: updatedSource, is_autosave: state._autosaving === true };
    if (state.sourceType === 'chart') {
      const blob = await getChartCanvasBlob();
      if (blob) {
        const base64 = await new Promise(resolve => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result.split(',')[1]);
          reader.readAsDataURL(blob);
        });
        savePayload.rendered_png = base64;
      }
    }

    const resp = await fetch(`/api/studio/${state.artifactId}/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(savePayload),
    });

    if (resp.ok) {
      const data = await resp.json();
      state.source = updatedSource;
      _updateDirtyState(false);
      _setSaveStatus('saved');
      // Keep the toast for manual saves; autosaves are silent — the header
      // chip is the only UI confirmation needed for the debounced path.
      if (!state._autosaving) showToast(`Saved (${(data.size_bytes / 1024).toFixed(0)} KB)`, 'success');
      // R2: surface non-fatal renderer warnings (e.g. PDF Unicode fallback)
      // even on autosave so a user editing emoji into a PDF doesn't ship a
      // silently-degraded file. The toast is informational (not 'error') so
      // it doesn't gate the user.
      const warnings = Array.isArray(data.warnings) ? data.warnings : [];
      for (const w of warnings) showToast(w, 'warning');
    } else {
      const err = await resp.json().catch(() => ({}));
      _setSaveStatus('dirty');
      showToast(err.error || 'Save failed', 'error');
    }
  } catch (err) {
    _setSaveStatus('dirty');
    showToast(`Save error: ${err.message}`, 'error');
  }

  state.saving = false;
  dom.saveBtn.textContent = 'Save';
  dom.saveBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Formatting Toolbar (document mode)
// ---------------------------------------------------------------------------
function buildDocToolbar() {
  if (!dom.toolbar) return;
  // Milkdown Crepe supplies the inline formatting toolbar, so our job here
  // is to surface ambient signals (word count, reading time) and hooks for
  // heading-level picker + focus mode. Delegated to the toolbar root.
  dom.toolbar.innerHTML = `
    <span style="font-size:var(--text-xs);color:var(--text-muted)">Document Editor</span>
    <span class="studio-toolbar-divider"></span>
    <select class="studio-heading-picker" id="studio-heading-picker" title="Paragraph style">
      <option value="p">Body</option>
      <option value="h1">Heading 1</option>
      <option value="h2">Heading 2</option>
      <option value="h3">Heading 3</option>
      <option value="h4">Heading 4</option>
    </select>
    <button class="studio-toolbar-btn" id="studio-focus-mode-btn" title="Focus mode (distraction-free)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/></svg>
    </button>
    <button class="studio-toolbar-btn" id="studio-outline-btn" title="Toggle outline">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
    </button>
    <button class="studio-toolbar-btn" id="studio-sidebar-toggle-btn" title="Toggle sidebar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="15" y1="3" x2="15" y2="21"/><line x1="9" y1="7" x2="12" y2="7"/><line x1="9" y1="11" x2="12" y2="11"/></svg>
    </button>
    <span class="studio-toolbar-divider"></span>
    <span class="studio-word-count" id="studio-word-count" aria-live="polite"></span>
  `;

  document.getElementById('studio-heading-picker')?.addEventListener('change', (e) => {
    _applyHeadingStyle(e.target.value);
    e.target.selectedIndex = 0; // reset so the picker acts as a stateless action
  });
  document.getElementById('studio-focus-mode-btn')?.addEventListener('click', () => {
    const overlay = dom.overlay;
    if (!overlay) return;
    overlay.classList.toggle('studio-focus-mode');
  });
  document.getElementById('studio-outline-btn')?.addEventListener('click', _toggleOutline);
  document.getElementById('studio-sidebar-toggle-btn')?.addEventListener('click', _toggleSessionSidebar);
  // Delegated close on the outline panel — built when the editor renders.
  // We defer the lookup to a microtask so the DOM is ready after innerHTML.
  queueMicrotask(() => {
    document.getElementById('studio-outline-close')?.addEventListener('click', _toggleOutline);
  });
}

// ---------------------------------------------------------------------------
// AI Popover
// ---------------------------------------------------------------------------
const AI_ACTIONS = {
  document: [
    { group: 'Edit', items: [
      { action: 'fix', name: 'Fix & Polish', desc: 'Grammar, spelling, flow' },
      { action: 'expand', name: 'Expand', desc: 'Flesh out with detail' },
      { action: 'formalize', name: 'Formalize', desc: 'Professional prose' },
      { action: 'translate', name: 'Translate', desc: 'To another language' },
    ]},
    { group: 'Analyze', items: [
      { action: 'summarize', name: 'Summarize', desc: 'Concise key points' },
      { action: 'keypoints', name: 'Key Points', desc: 'Numbered takeaways' },
      { action: 'outline', name: 'Outline', desc: 'Structured sections' },
    ]},
    { group: 'Create', items: [
      { action: 'add_section', name: 'Add Section', desc: 'AI writes a new section' },
    ]},
  ],
  presentation: [
    { group: 'Edit', items: [
      { action: 'expand', name: 'Expand Slide', desc: 'More detail on this slide' },
      { action: 'formalize', name: 'Formalize', desc: 'Professional language' },
      { action: 'fix', name: 'Fix & Polish', desc: 'Grammar and flow' },
      { action: 'translate', name: 'Translate', desc: 'To another language' },
    ]},
    { group: 'Create', items: [
      { action: 'add_slide', name: 'Add Slide', desc: 'AI generates a new slide' },
      { action: 'speaker_notes', name: 'Speaker Notes', desc: 'AI writes notes for this slide' },
    ]},
    { group: 'Analyze', items: [
      { action: 'summarize', name: 'Summarize Deck', desc: 'Overview of all slides' },
    ]},
  ],
  spreadsheet: [
    { group: 'Edit', items: [
      { action: 'fill_data', name: 'Fill Data', desc: 'AI suggests values for empty cells' },
      { action: 'add_column', name: 'Add Column', desc: 'AI suggests a new column' },
      { action: 'fix', name: 'Fix & Polish', desc: 'Clean up labels and formatting' },
    ]},
    { group: 'Analyze', items: [
      { action: 'summarize', name: 'Summarize', desc: 'Describe this data' },
      { action: 'keypoints', name: 'Key Insights', desc: 'Find patterns and trends' },
    ]},
  ],
  chart: [
    { group: 'Edit', items: [
      { action: 'suggest_type', name: 'Suggest Chart Type', desc: 'AI recommends a better type' },
      { action: 'add_series', name: 'Add Series', desc: 'AI suggests additional data' },
    ]},
    { group: 'Analyze', items: [
      { action: 'summarize', name: 'Describe Chart', desc: 'AI analyzes the data' },
    ]},
  ],
};

function buildAiPopover(editorType) {
  const groups = AI_ACTIONS[editorType] || AI_ACTIONS.document;
  const existing = dom.headerActions?.querySelector('.studio-ai-popover');
  if (existing) existing.remove();

  const popover = document.createElement('div');
  popover.className = 'studio-ai-popover hidden';

  let html = '';
  for (const group of groups) {
    html += `<div class="studio-ai-popover-section">`;
    html += `<div class="studio-ai-popover-label">${escapeHtml(group.group)}</div>`;
    for (const item of group.items) {
      html += `<button class="studio-ai-popover-item" data-action="${escapeHtml(item.action)}">
        <div><span class="studio-ai-popover-name">${escapeHtml(item.name)}</span>
        <span class="studio-ai-popover-desc">${escapeHtml(item.desc)}</span></div>
      </button>`;
    }
    html += `</div>`;
  }
  popover.innerHTML = html;

  // Wire clicks
  popover.addEventListener('click', (e) => {
    const item = e.target.closest('.studio-ai-popover-item');
    if (!item) return;
    const action = item.dataset.action;
    popover.classList.add('hidden');
    if (action === 'translate') {
      const lang = prompt('Translate to which language?', 'Spanish');
      if (lang) studioAiAction(action, lang);
    } else if (action) {
      studioAiAction(action);
    }
  });

  dom.headerActions?.appendChild(popover);
}

// ---------------------------------------------------------------------------
// AI Actions — Inline Blocks
// ---------------------------------------------------------------------------
const ACTION_LABELS = {
  summarize: 'AI Summary', keypoints: 'AI Key Points', expand: 'AI Expanded',
  formalize: 'AI Formalized', fix: 'AI Polished', explain: 'AI Explanation',
  extract_tasks: 'AI Tasks', outline: 'AI Outline', translate: 'AI Translation',
  ask: 'AI Answer', add_section: 'AI New Section',
  add_slide: 'AI New Slide', speaker_notes: 'AI Speaker Notes',
  fill_data: 'AI Fill Data', add_column: 'AI New Column',
  suggest_type: 'AI Chart Type', add_series: 'AI New Series',
};

async function studioAiAction(action, question) {
  let content;
  if (state.sourceType === 'presentation') {
    const slide = state.slides[state.currentSlide];
    content = slide ? `Title: ${slide.title}\n\n${slide.body}` : '';
    // For deck-wide actions, send all slides
    if (['summarize'].includes(action)) {
      content = state.slides.map((s, i) => `Slide ${i+1}: ${s.title}\n${s.body}`).join('\n\n---\n\n');
    }
  } else if (state.sourceType === 'spreadsheet') {
    const sheet = state.gridSheets[state.currentSheet];
    if (sheet) {
      content = `Headers: ${sheet.headers.join(', ')}\n`;
      content += sheet.rows.map((r, i) => `Row ${i+1}: ${r.join(', ')}`).join('\n');
    }
  } else if (state.sourceType === 'chart') {
    const cfg = state.chartConfig;
    content = `Chart: ${cfg.title} (${cfg.chart_type})\nLabels: ${cfg.labels.join(', ')}\n`;
    content += cfg.datasets.map(d => `${d.name}: ${d.values.join(', ')}`).join('\n');
  } else if (state.sourceType === 'ebook') {
    const source = getEbookSource();
    content = `Title: ${source.title}\nAuthor: ${source.author}\n\n`;
    content += source.chapters.map((ch, i) => (
      `Chapter ${i + 1}: ${ch.heading}\n${ch.body}`
    )).join('\n\n---\n\n');
  } else {
    content = getEditorMarkdown();
  }
  if (!content && action !== 'add_section' && action !== 'add_slide') {
    showToast('No content to work with', 'warning');
    return;
  }

  if (state.aiAbort) state.aiAbort.abort();
  state.aiAbort = new AbortController();

  const blocksContainer = dom.body.querySelector('#studio-ai-blocks') || dom.body;
  const label = ACTION_LABELS[action] || 'AI Result';
  const isInsertable = ['expand', 'formalize', 'fix', 'translate', 'add_section', 'outline', 'extract_tasks', 'add_slide', 'speaker_notes'].includes(action);

  const block = document.createElement('div');
  block.className = 'studio-ai-block studio-ai-block-streaming';
  block.innerHTML = `
    <div class="studio-ai-block-header">
      <span class="studio-ai-block-label">${escapeHtml(label)}</span>
      <div class="studio-ai-block-actions">
        ${isInsertable ? '<button class="studio-ai-block-btn primary" data-action="insert">Insert</button>' : ''}
        <button class="studio-ai-block-btn" data-action="copy">Copy</button>
        <button class="studio-ai-block-btn remove" data-action="remove">&times;</button>
      </div>
    </div>
    <div class="studio-ai-block-content"></div>
  `;
  blocksContainer.appendChild(block);

  // Scroll the block into view inside whichever editor owns the scroll
  // viewport, then keep its (height-capped) content pinned to the bottom as
  // the response streams in.
  const scrollEl = dom.body.querySelector('.studio-doc-editor')
    || dom.body.querySelector('.studio-slide-canvas-area')
    || dom.body.querySelector('.studio-ebook-editor');
  if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
  else block.scrollIntoView({ block: 'nearest' });

  const contentEl = block.querySelector('.studio-ai-block-content');

  const aiAction = (action === 'add_section' || action === 'add_slide' || action === 'speaker_notes') ? 'ask' : action;
  const body = { action: aiAction, content: content || 'Empty document', model: app.state.currentModel || '' };
  if (question) body.question = question;
  if (action === 'add_section') {
    body.content = content;
    body.question = 'Write a new section that would logically follow the existing content. Output as markdown with a ## heading.';
    body.action = 'ask';
  } else if (action === 'add_slide') {
    body.content = state.slides.map((s, i) => `Slide ${i+1}: ${s.title}\n${s.body}`).join('\n\n');
    body.question = 'Write content for a new presentation slide that logically continues this deck. Output format: first line is the slide title, then bullet points using "- " prefix.';
    body.action = 'ask';
  } else if (action === 'speaker_notes') {
    body.question = 'Write speaker notes for this slide. The notes should guide the presenter on what to say, key points to emphasize, and transitions. Output plain text, 3-5 sentences.';
    body.action = 'ask';
  } else if (action === 'fill_data') {
    body.question = 'Look at the data and suggest values for any empty cells. Output as a list of "Row X, Column Y: value" entries.';
    body.action = 'ask';
  } else if (action === 'add_column') {
    body.question = 'Suggest a useful new column for this data. Output the column header name on the first line, then one value per row.';
    body.action = 'ask';
  } else if (action === 'suggest_type') {
    body.question = 'Look at this data and suggest the best chart type. Explain why. Valid types: bar, line, pie, scatter, area, stacked_bar, stacked_area, horizontal_bar.';
    body.action = 'ask';
  } else if (action === 'add_series') {
    body.question = 'Suggest an additional data series that would complement this chart. Output the series name on the first line, then one number per row matching the existing labels.';
    body.action = 'ask';
  }

  let fullText = '';

  try {
    const resp = await fetch('/api/browse/ai', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: state.aiAbort.signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: 'AI failed' }));
      block.classList.remove('studio-ai-block-streaming');
      contentEl.innerHTML = `<p style="color:var(--text-muted)">${escapeHtml(err.error)}</p>`;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Shared incremental renderer (chat/stream-render.js) — coalesced + split
    // so a fast stream doesn't re-parse the whole answer per delta. compact =
    // full chat markdown minus the chat-only code toolbar.
    const aiRender = makeStreamRenderer(contentEl, {
      compact: true,
      onFlush: () => {
        if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
        contentEl.scrollTop = contentEl.scrollHeight;
      },
    });

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') break;
        try {
          const data = JSON.parse(payload);
          if (data.error) {
            block.classList.remove('studio-ai-block-streaming');
            contentEl.innerHTML = `<p style="color:var(--text-muted)">${escapeHtml(data.error)}</p>`;
            return;
          }
          if (data.delta) {
            fullText += data.delta;
            aiRender.render(fullText);
          }
        } catch { /* skip */ }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') { block.remove(); return; }
    contentEl.innerHTML = `<p style="color:var(--text-muted)">Error: ${escapeHtml(err.message)}</p>`;
  }

  block.classList.remove('studio-ai-block-streaming');
  if (fullText) {
    contentEl.innerHTML = renderMarkdown(fullText, { compact: true });
    highlightCodeDeferred(contentEl);
    block.dataset.markdown = fullText;
  }
}

function insertAiBlock(markdown) {
  if (!markdown) return;

  if (state.sourceType === 'presentation') {
    // Try to parse as a slide (title on first line, bullets after)
    const lines = markdown.trim().split('\n');
    const title = lines[0].replace(/^#+\s*/, '').replace(/^\*\*(.+)\*\*$/, '$1').trim();
    const body = lines.slice(1).join('\n').trim();
    addSlide({ layout: 'content', title, body, notes: '', image_url: '', additional_images: [] });
    showToast('Slide added', 'success');
    return;
  }

  // Document mode: append to editor
  const current = getEditorMarkdown();
  const updated = current ? current.trim() + '\n\n' + markdown : markdown;
  const pageEl = document.getElementById('studio-doc-page');
  if (pageEl) loadMilkdownEditor(pageEl, updated);
  markDirty();
  showToast('Inserted into document', 'success');
}
