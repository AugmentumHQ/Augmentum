/**
 * Library Workspace — immersive play + code editing for app builder projects.
 * Replaces the old library play view.
 */
import { escapeHtml } from './app.js';
import { assembleProject, getLastSourceMap, buildPreviewSrcdoc } from './assemble.js';
import { getSettings } from './settings.js';
import * as CodeMind from './codemind.js';
import * as CMEditor from './cm-editor.js';
import { ViewStack } from './view-stack.js';

// --- Helpers ---
/** Get the current model reliably — app.state first, localStorage fallback, empty last resort. */
function _getCurrentModel() {
  const fromState = window.app?.state?.currentModel;
  if (fromState && fromState !== 'default') return fromState;
  const fromStorage = localStorage.getItem('augmentum-selected-model');
  if (fromStorage) return fromStorage;
  return '';
}

/** Get current editor content (CM6 primary, falls back to _files). */
function _getEditorContent() {
  if (_cmEditorId && _cmReady) return CMEditor.getContent(_cmEditorId);
  const f = _files[_activeFile];
  return f?.content || '';
}

/** Set current editor content. */
function _setEditorContent(content) {
  if (_cmEditorId && _cmReady) CMEditor.setContent(_cmEditorId, content);
  const f = _files[_activeFile];
  if (f) f.content = content;
}

/** Focus the editor. */
function _focusEditor() {
  if (_cmEditorId && _cmReady) CMEditor.focus(_cmEditorId);
}

/** Check if the editor is focused. */
function _isEditorFocused() {
  const cmContainer = _el.cmContainer;
  return cmContainer ? cmContainer.contains(document.activeElement) : false;
}

/** Get the editor area DOM element (for positioning overlays). */
function _getEditorArea() {
  return _el.cmContainer?.closest('.workspace-editor-area') || _el.cmContainer;
}

// --- State ---
let _open = false;
let _mode = 'play'; // 'play' | 'work'
let _artifact = null;
let _files = [];        // current working files
let _snapshot = null;    // pre-edit snapshot for revert
let _sourceMap = [];
let _findBar = null;
let _findMatches = [];
let _findIdx = -1;
let _cmEditorId = null;   // CM6 editor instance ID
let _cmReady = false;     // CM6 loaded and ready
let _activeFile = 0;     // index into _files
let _modified = new Set(); // paths with unsaved changes
let _consoleEntries = [];
let _headerTimer = null;
let _pendingHunks = [];    // Array<{id, file, search, replace, status, diffLines, matchStart}>
let _rawResponse = '';
let _patchedFiles = [];
let _focusedHunkIdx = -1;
let _animating = false;
let _animPreviewTimer = null;
let _activeOperation = null; // 'quick' | 'rebuild'
let _activePromptAbort = null;
let _activeBuildId = '';
let _lastCheckpointSignature = '';
let _generationStartedAt = 0;
let _generationTimer = null;
let _generationLastStage = '';

// --- DOM Cache ---
const _el = {};
function _cacheDom() {
  const ids = [
    'workspace', 'workspace-header', 'workspace-title', 'workspace-back',
    'workspace-close', 'workspace-mode-toggle', 'workspace-accept-bar',
    'workspace-body', 'workspace-editor', 'workspace-file-tabs',
    'workspace-cm-container', 'workspace-resize',
    'workspace-preview', 'workspace-iframe', 'workspace-console',
    'workspace-console-count', 'workspace-console-clear',
    'workspace-console-toggle', 'workspace-console-log',
    'workspace-prompt', 'workspace-prompt-input', 'workspace-prompt-send',
    'workspace-learning-style', 'workspace-learn-diff',
  ];
  for (const id of ids) {
    const key = id.replace('workspace-', '').replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    _el[key] = document.getElementById(id);
  }
  // Promote to body so the three-pane Library can dispatch into us
  // without inheriting display:none from a hidden ancestor. The element
  // was authored as a child of the legacy library overlay in index.html;
  // that shell can stay display:none while the new Library surface is
  // active, and a position:fixed descendant inside a display:none parent
  // is still hidden. Reparenting is idempotent.
  if (_el.workspace && _el.workspace.parentElement !== document.body) {
    document.body.appendChild(_el.workspace);
  }
}

// --- Open / Close ---
export async function openWorkspace(artifact, mode = 'play') {
  if (!_el.workspace) _cacheDom();
  _wireEvents();
  _artifact = artifact;
  _mode = mode;

  // Fetch full artifact data (source_json) if not cached
  if (!artifact.source_json) {
    try {
      const resp = await fetch(`/api/artifacts/${artifact.id}`);
      if (resp.ok) {
        const full = await resp.json();
        artifact.source_json = full.source_json;
      }
    } catch { /* fallback to preview URL */ }
  }

  // Parse files from source_json
  if (artifact.source_json) {
    try {
      const source = typeof artifact.source_json === 'string'
        ? JSON.parse(artifact.source_json) : artifact.source_json;
      _files = source?.files || [];
    } catch { _files = []; }
  } else {
    _files = [];
  }

  // Show workspace with entrance animation
  _el.workspace?.removeAttribute('hidden');
  _el.workspace?.classList.remove('ws-leaving');
  _el.workspace?.classList.add('ws-entering');
  _el.workspace?.addEventListener('animationend', () => _el.workspace?.classList.remove('ws-entering'), { once: true });
  _setMode(mode);
  _loadPreview();
  _updateTitle();
  _open = true;

  // Initialize CM6 editor (lazy — non-blocking)
  _initCM6();

  // Initialize CodeMind AST engine (lazy — non-blocking)
  _initCodeMind();

  // Persist state for refresh recovery
  try { localStorage.setItem('augmentum_workspace', JSON.stringify({ artifactId: artifact.id, mode })); } catch {}

  // Track in ViewStack so a mode change (or library close) cleanly tears us
  // down instead of leaving the artifact editor visible over empty chat.
  // onClose skips the unsaved-changes confirm — a mode change is
  // authoritative user intent, and a blocking confirm during a programmatic
  // pop would desync the stack.
  ViewStack.pushOverlay('workspace', { onClose: () => _teardown({ skipConfirm: true }) });
}

// Re-entry guard: when _teardown pops the stack below, popOverlay calls our
// onClose which calls _teardown again. The _open check catches that, but an
// explicit flag makes intent clearer and costs nothing.
let _closeViaStack = false;

function _teardown({ skipConfirm = false } = {}) {
  if (_closeViaStack || !_open) return;
  // Unsaved changes guard — skipped for programmatic closes (e.g. mode change)
  if (!skipConfirm && _modified.size > 0) {
    if (!confirm('You have unsaved changes. Discard and close?')) return;
  }
  // Animate out, then hide
  if (_el.workspace) {
    _el.workspace.classList.add('ws-leaving');
    _el.workspace.addEventListener('animationend', () => {
      _el.workspace?.classList.remove('ws-leaving');
      _el.workspace?.setAttribute('hidden', '');
    }, { once: true });
  }
  if (_el.iframe) { _el.iframe.srcdoc = ''; _el.iframe.removeAttribute('src'); }
  // Destroy CM6 editor
  if (_cmEditorId) { CMEditor.destroy(_cmEditorId); _cmEditorId = null; }
  _artifact = null;
  _files = [];
  _snapshot = null;
  _patchedFiles = [];
  _pendingHunks = [];
  _rawResponse = '';
  _focusedHunkIdx = -1;
  _modified.clear();
  _consoleEntries = [];
  _open = false;
  _stopBuildStatusFeed();
  if (_activePromptAbort) {
    try { _activePromptAbort.abort(); } catch {}
  }
  _activePromptAbort = null;
  _activeOperation = null;
  _activeBuildId = '';
  _lastCheckpointSignature = '';
  _clearHeaderTimer();
  _dismissCoachTray();
  _dismissGenerationPanel();
  _el.workspace?.classList.remove('console-open');
  try { localStorage.removeItem('augmentum_workspace'); } catch {}

  // Sync ViewStack — pop after teardown so onClose re-entry hits the !_open
  // guard and short-circuits. Skip if we were invoked via onClose (flag set
  // by the caller).
  if (!skipConfirm && ViewStack.hasOverlay('workspace')) {
    _closeViaStack = true;
    try { ViewStack.popOverlay('workspace'); }
    finally { _closeViaStack = false; }
  }
}

export function closeWorkspace() {
  _teardown();
}

/**
 * Recover workspace state after page refresh.
 * Called from app.js init — if a workspace was open, reopen it.
 */
export async function recoverWorkspace() {
  try {
    const saved = localStorage.getItem('augmentum_workspace');
    if (!saved) return false;
    const { artifactId, mode } = JSON.parse(saved);
    if (!artifactId) return false;
    // Fetch artifact data
    const resp = await fetch(`/api/artifacts/${artifactId}`);
    if (!resp.ok) { localStorage.removeItem('augmentum_workspace'); return false; }
    const artifact = await resp.json();
    // Open library first (workspace lives inside library overlay)
    const lib = await import('./library.js');
    await lib.openLibrary();
    // Then open workspace
    await openWorkspace(artifact, mode || 'play');
    return true;
  } catch {
    localStorage.removeItem('augmentum_workspace');
    return false;
  }
}

// --- Mode Switching ---
function _setMode(mode) {
  _mode = mode;
  const ws = _el.workspace;
  if (!ws) return;
  ws.classList.toggle('play-mode', mode === 'play');
  ws.classList.toggle('work-mode', mode === 'work');
  // Toggle panels with animation
  _el.editor?.classList.toggle('hidden', mode === 'play');
  _el.resize?.classList.toggle('hidden', mode === 'play');
  _el.console?.classList.toggle('hidden', mode === 'play');
  _syncConsoleState();
  // Animate editor panel in when switching to work mode
  if (mode === 'work' && _el.editor) {
    _el.editor.classList.add('ws-panel-enter');
    _el.editor.addEventListener('animationend', () => _el.editor?.classList.remove('ws-panel-enter'), { once: true });
  }
  // Update mode toggle buttons
  _el.modeToggle?.querySelectorAll('.workspace-mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  // In play mode: start header auto-hide
  if (mode === 'play') _startHeaderAutoHide();
  else _clearHeaderTimer();
  // In work mode: load editor
  if (mode === 'work') _loadEditor();
}

// --- Preview ---
function _loadPreview() {
  if (!_el.iframe || _files.length === 0) return;
  const html = assembleProject(_files);
  _sourceMap = getLastSourceMap();
  if (!html) return;
  _clearConsoleEntries();
  // Inject console capture script
  const wrapped = _injectConsoleCapture(buildPreviewSrcdoc ? buildPreviewSrcdoc(html) : html);
  _sourceMap = _remapSourceMapForSrcdoc(_sourceMap, wrapped);
  _el.iframe.srcdoc = wrapped;
}

function _remapSourceMapForSrcdoc(sourceMap, srcdoc) {
  if (!Array.isArray(sourceMap) || !sourceMap.length || !srcdoc) return sourceMap || [];
  const lines = srcdoc.split('\n');
  return sourceMap.map(entry => {
    const marker = `/* ${entry.file} */`;
    const idx = lines.findIndex(line => line.includes(marker));
    return idx >= 0 ? { ...entry, assembledLineStart: idx + 2 } : entry;
  });
}

function _injectConsoleCapture(html) {
  // Skip if buildPreviewSrcdoc already injected console capture
  if (html.includes('code-console')) return html;
  const script = `<script>
window.onerror=function(m,s,l,c,e){parent.postMessage({type:'code-console',level:'error',message:m,source:s,line:l,column:c,stack:e?.stack||''},'*')};
window.addEventListener('unhandledrejection',function(e){parent.postMessage({type:'code-console',level:'error',message:'Unhandled: '+(e.reason?.message||e.reason||'unknown'),stack:e.reason?.stack||''},'*')});
['log','warn','error'].forEach(function(lv){var o=console[lv];console[lv]=function(){o.apply(console,arguments);try{parent.postMessage({type:'code-console',level:lv,message:Array.from(arguments).map(function(a){try{return typeof a==='object'?JSON.stringify(a):String(a)}catch(e){return String(a)}}).join(' ')},'*')}catch(e){}}});
<\/script>`;
  return html.replace('</body>', script + '</body>');
}

// --- Header Auto-hide (play mode) ---
function _startHeaderAutoHide() {
  _el.header?.classList.remove('auto-hidden');
  _clearHeaderTimer();
  _headerTimer = setTimeout(() => {
    if (_mode === 'play') _el.header?.classList.add('auto-hidden');
  }, 3000);
}

function _clearHeaderTimer() {
  if (_headerTimer) { clearTimeout(_headerTimer); _headerTimer = null; }
  _el.header?.classList.remove('auto-hidden');
}

// --- Editor ---
function _loadEditor() {
  _renderFileTabs();
  _loadFileIntoEditor(_activeFile);
}

function _renderFileTabs() {
  if (!_el.fileTabs) return;
  _el.fileTabs.innerHTML = _files.map((f, i) => {
    const mod = _modified.has(f.path) ? ' modified' : '';
    const active = i === _activeFile ? ' active' : '';
    return `<button class="workspace-file-tab${active}${mod}" data-idx="${i}">${_getFileIcon(f.path)}${escapeHtml(f.path)}</button>`;
  }).join('');
  _el.fileTabs.querySelectorAll('.workspace-file-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      _activeFile = parseInt(btn.dataset.idx, 10);
      _loadEditor();
    });
  });
  // Add new file button
  const addBtn = document.createElement('button');
  addBtn.className = 'workspace-file-tab workspace-add-file';
  addBtn.textContent = '+';
  addBtn.title = 'Add new file';
  addBtn.addEventListener('click', _addNewFile);
  _el.fileTabs.appendChild(addBtn);
  // Per-file hunk status indicators
  if (_pendingHunks.length > 0) {
    _el.fileTabs.querySelectorAll('.workspace-file-tab').forEach(btn => {
      const idx = parseInt(btn.dataset.idx, 10);
      const file = _files[idx] || _snapshot?.[idx];
      if (!file) return;
      const status = _getFileStatus(file.path);
      btn.classList.toggle('has-changes', status !== 'unchanged');
      btn.classList.toggle('all-accepted', status === 'accepted');
      btn.classList.toggle('all-rejected', status === 'rejected');
      // Show hunk count badge
      const count = _pendingHunks.filter(h => h.file === file.path).length;
      if (count > 0 && !btn.querySelector('.file-tab-count')) {
        const badge = document.createElement('span');
        badge.className = 'file-tab-count';
        badge.textContent = count;
        btn.appendChild(badge);
      }
    });
  }
}

function _loadFileIntoEditor(idx) {
  if (_animating) _animating = false;
  const file = _files[idx];
  if (!file) return;
  _hideDiffView();

  const container = _el.cmContainer;
  const lang = _getLangClass(file.path);

  if (_cmReady && container) {
    if (_cmEditorId) {
      // Editor exists — update content and language
      CMEditor.setContent(_cmEditorId, file.content || '');
      CMEditor.setLanguage(_cmEditorId, lang);
    } else {
      // Create new CM6 editor
      _cmEditorId = CMEditor.create(container, {
        content: file.content || '',
        language: lang,
        onChange: (content) => {
          const f = _files[_activeFile];
          if (f) {
            f.content = content;
            _modified.add(f.path);
            _renderFileTabs();
          }
          // Debounced preview reload
          clearTimeout(_previewReloadTimer);
          _previewReloadTimer = setTimeout(_loadPreview, 500);
          // CodeMind diagnostics
          _onCodeMindEdit();
        },
        onSave: () => {
          // Format + save
          _formatAndSave();
        },
      });
    }
    // Run CodeMind diagnostics on the new file
    _runCodeMindDiagnostics();
  }

  _updateStatusBar();
  // If hunks are pending, show hunk diff instead of editor
  if (_pendingHunks.length > 0) {
    _renderHunkDiff(file.path);
  }
}

async function _formatAndSave() {
  // Format with Prettier if loaded
  if (_prettier && _cmEditorId) {
    const content = CMEditor.getContent(_cmEditorId);
    const file = _files[_activeFile];
    const lang = file ? _getLangClass(file.path) : '';
    const parser = _PRETTIER_PARSERS[lang];
    if (parser) {
      try {
        const formatted = await _prettier.format(content, {
          parser, plugins: Object.values(_prettierPlugins),
          printWidth: 100, tabWidth: 2, useTabs: false, semi: true,
          singleQuote: true, trailingComma: 'es5', bracketSpacing: true,
        });
        if (formatted !== content) CMEditor.setContent(_cmEditorId, formatted);
      } catch { /* format failed */ }
    }
  }
  _saveToArtifact();
}


function _getLangClass(path) {
  if (path.endsWith('.html') || path.endsWith('.htm')) return 'markup';
  if (path.endsWith('.css') || path.endsWith('.scss')) return 'css';
  if (path.endsWith('.json')) return 'json';
  return 'javascript';
}

// --- File Type Icons ---
function _getFileIcon(path) {
  if (path.endsWith('.html') || path.endsWith('.htm')) return '<span class="file-icon file-icon-html">H</span>';
  if (path.endsWith('.css') || path.endsWith('.scss')) return '<span class="file-icon file-icon-css">C</span>';
  if (path.endsWith('.js') || path.endsWith('.ts')) return '<span class="file-icon file-icon-js">J</span>';
  if (path.endsWith('.json')) return '<span class="file-icon file-icon-json">{}</span>';
  return '<span class="file-icon file-icon-other">F</span>';
}

// --- Status Bar ---
function _updateStatusBar() {
  const posEl = document.getElementById('workspace-status-pos');
  const langEl = document.getElementById('workspace-status-lang');
  if (!posEl) return;

  // Get cursor position from CM6 or fallback textarea
  if (_cmEditorId && _cmReady) {
    const cursor = CMEditor.getCursor(_cmEditorId);
    posEl.textContent = `Ln ${cursor.line}, Col ${cursor.col}`;
  }

  if (langEl) {
    const file = _files[_activeFile];
    if (file) {
      if (file.path.endsWith('.html') || file.path.endsWith('.htm')) langEl.textContent = 'HTML';
      else if (file.path.endsWith('.css')) langEl.textContent = 'CSS';
      else if (file.path.endsWith('.json')) langEl.textContent = 'JSON';
      else langEl.textContent = 'JavaScript';
    }
  }
}

// ---------------------------------------------------------------------------
// CM6 Initialization
// ---------------------------------------------------------------------------

async function _initCM6() {
  try {
    const ok = await CMEditor.load();
    if (!ok) {
      console.warn('[Workspace] CM6 unavailable (CDN offline) — editor disabled');
      return;
    }
    _cmReady = true;
    // If files already loaded, create the editor now
    if (_files.length > 0 && _el.cmContainer) {
      _loadFileIntoEditor(_activeFile);
    }
  } catch (err) {
    console.warn('[Workspace] CM6 init failed:', err.message);
  }
}

// ---------------------------------------------------------------------------
// File Tree — collapsible sidebar showing project files
// ---------------------------------------------------------------------------

function _toggleFileTree() {
  const tree = document.getElementById('workspace-file-tree');
  if (!tree) return;
  tree.classList.toggle('hidden');
  if (!tree.classList.contains('hidden')) _renderFileTree();
}

function _renderFileTree() {
  const list = document.getElementById('file-tree-list');
  if (!list) return;

  list.innerHTML = _files.map((f, i) => {
    const ext = f.path.split('.').pop()?.toLowerCase() || '';
    const iconClass = ext === 'html' || ext === 'htm' ? 'ft-html' :
                      ext === 'css' || ext === 'scss' ? 'ft-css' :
                      ext === 'js' || ext === 'jsx' ? 'ft-js' :
                      ext === 'json' ? 'ft-json' :
                      ext === 'py' ? 'ft-py' : 'ft-other';
    const active = i === _activeFile ? ' active' : '';
    const modified = _modified.has(f.path) ? ' modified' : '';
    const lines = (f.content || '').split('\n').length;
    return `<div class="file-tree-item${active}${modified}" data-idx="${i}">
      <span class="file-tree-icon ${iconClass}"></span>
      <span class="file-tree-name">${escapeHtml(f.path)}</span>
      <span class="file-tree-meta">${lines}L</span>
    </div>`;
  }).join('');

  // Click to switch file
  list.addEventListener('click', (e) => {
    const item = e.target.closest('.file-tree-item');
    if (!item) return;
    const idx = parseInt(item.dataset.idx, 10);
    if (isNaN(idx) || !_files[idx]) return;
    _activeFile = idx;
    _loadFileIntoEditor(idx);
    _renderFileTabs();
    _renderFileTree();
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert tree-sitter row/col to string offset. */
function _posToOffset(code, row, col) {
  let offset = 0;
  const lines = code.split('\n');
  for (let i = 0; i < row && i < lines.length; i++) {
    offset += lines[i].length + 1; // +1 for \n
  }
  return offset + Math.min(col, (lines[row] || '').length);
}

// ---------------------------------------------------------------------------
// CodeMind Integration — AST-powered code intelligence
// ---------------------------------------------------------------------------

let _codeMindReady = false;
let _codeMindDiagnostics = [];
let _codeMindDebounce = null;

async function _initCodeMind() {
  try {
    const ok = await CodeMind.init();
    if (!ok) { console.debug('[Workspace] CodeMind unavailable (CDN offline)'); return; }
    _codeMindReady = true;

    // Pre-load grammars for current files
    for (const file of _files) {
      const lang = _getLangClass(file.path);
      if (CodeMind.resolveLanguage(lang)) {
        CodeMind.parse(file.content || '', lang, file.path).catch(() => {});
      }
    }
    // Run initial diagnostics on active file
    _runCodeMindDiagnostics();
  } catch { /* CodeMind init is best-effort */ }
}

/**
 * Run AST diagnostics on the active file. Called on edit (debounced)
 * and on file switch. Updates the error indicator in the status bar
 * and stores diagnostics for the error gutter.
 */
function _runCodeMindDiagnostics() {
  if (!_codeMindReady) return;

  const file = _files[_activeFile];
  if (!file) return;

  // Get content from CM6 editor or fallback textarea
  const content = _cmEditorId ? CMEditor.getContent(_cmEditorId) : _getEditorContent();
  const lang = _getLangClass(file.path);
  const result = CodeMind.parseSync(content, lang, file.path);
  _codeMindDiagnostics = result ? result.errors : [];

  // Update status bar error count
  const statusEl = document.getElementById('workspace-status-errors');
  if (statusEl) {
    if (_codeMindDiagnostics.length > 0) {
      statusEl.textContent = `\u26A0 ${_codeMindDiagnostics.length} error${_codeMindDiagnostics.length > 1 ? 's' : ''}`;
      statusEl.style.color = 'var(--error)';
      statusEl.title = _codeMindDiagnostics.map(e =>
        `Ln ${e.startRow + 1}: ${e.message}`
      ).join('\n');
    } else {
      statusEl.textContent = '\u2713';
      statusEl.style.color = 'var(--success, #22c55e)';
      statusEl.title = 'No syntax errors';
    }
  }

  // Push diagnostics to CM6 lint gutter (red squiggly underlines)
  if (_cmEditorId && _cmReady) {
    CMEditor.setDiagnosticsFromCodeMind(_cmEditorId, _codeMindDiagnostics);
  }

  // Update symbol outline
  _updateSymbolOutline();
}

/**
 * Debounced CodeMind update — called on every edit.
 */
function _onCodeMindEdit() {
  if (!_codeMindReady) return;
  clearTimeout(_codeMindDebounce);
  _codeMindDebounce = setTimeout(_runCodeMindDiagnostics, 150);
}

/**
 * Update the symbol outline (function/class list) in the status bar.
 * Click to show a dropdown, click a symbol to jump to its line.
 */
function _updateSymbolOutline() {
  const symbolEl = document.getElementById('workspace-status-symbols');
  if (!symbolEl || !_codeMindReady) return;

  const file = _files[_activeFile];
  if (!file) { symbolEl.textContent = ''; return; }

  const lang = _getLangClass(file.path);
  const decls = CodeMind.getDeclarations(_getEditorContent(), lang, file.path);

  if (decls.length === 0) {
    symbolEl.textContent = '';
    return;
  }

  // Show symbol count as clickable label
  symbolEl.textContent = `\u{1D4AE} ${decls.length} symbols`;
  symbolEl.style.cursor = 'pointer';

  // Remove old handler and re-attach
  symbolEl.onclick = (e) => {
    e.stopPropagation();
    // Toggle dropdown
    let existing = document.querySelector('.workspace-symbol-dropdown');
    if (existing) { existing.remove(); return; }

    const dropdown = document.createElement('div');
    dropdown.className = 'workspace-symbol-dropdown';
    dropdown.innerHTML = decls.map(d => {
      const icon = d.type === 'function' ? 'fn' : d.type === 'class' ? 'cls' : 'var';
      return `<div class="symbol-item" data-line="${d.line}"><span class="symbol-icon symbol-${icon}">${icon}</span>${escapeHtml(d.name)}<span class="symbol-line">:${d.line}</span></div>`;
    }).join('');

    // Position above the status bar
    const rect = symbolEl.getBoundingClientRect();
    dropdown.style.bottom = `${window.innerHeight - rect.top + 4}px`;
    dropdown.style.left = `${rect.left}px`;
    document.body.appendChild(dropdown);

    dropdown.addEventListener('click', (ev) => {
      const item = ev.target.closest('.symbol-item');
      if (!item) return;
      const line = parseInt(item.dataset.line, 10);
      if (isNaN(line)) return;

      // Jump to line via CM6 or fallback
      if (_cmEditorId && _cmReady) {
        CMEditor.setCursor(_cmEditorId, line, 1);
      }
      _updateStatusBar();
      dropdown.remove();
    });

    // Dismiss on outside click
    const dismiss = (ev) => {
      if (!dropdown.contains(ev.target) && ev.target !== symbolEl) {
        dropdown.remove();
        document.removeEventListener('click', dismiss);
      }
    };
    setTimeout(() => document.addEventListener('click', dismiss), 0);
  };
}

/**
 * Get compressed scope context for LLM calls.
 * Returns a smaller code context focused on the current cursor position.
 */
export function getCodeMindScope() {
  if (!_codeMindReady) return null;
  const file = _files[_activeFile];
  if (!file) return null;

  const val = _getEditorContent();
  const cursor = _cmEditorId ? CMEditor.getCursor(_cmEditorId) : { line: 1 };
  const row = cursor.line - 1;
  const lang = _getLangClass(file.path);

  return CodeMind.getScopeAt(val, row, lang, file.path);
}

/**
 * Get file declarations for multi-file context compression.
 */
export function getFileDeclarations(filePath, content, lang) {
  if (!_codeMindReady) return [];
  return CodeMind.getDeclarations(content, lang, filePath);
}

/**
 * Validate LLM-generated code before showing to user.
 */
export async function validateGenerated(code, lang) {
  if (!_codeMindReady) return { valid: true, errors: [] };
  return CodeMind.validate(code, lang);
}

// ---------------------------------------------------------------------------
// Prettier — Code Formatting (Shift+Alt+F or status bar button)
// ---------------------------------------------------------------------------
// Lazy-loads Prettier + plugins from CDN on first use (~185KB compressed).

let _prettier = null;
let _prettierPlugins = {};
let _prettierLoading = false;

const PRETTIER_VERSION = '3.5.3';
const PRETTIER_CDN = `https://cdn.jsdelivr.net/npm/prettier@${PRETTIER_VERSION}`;

async function _loadPrettier() {
  if (_prettier) return true;
  if (_prettierLoading) return false;
  _prettierLoading = true;

  try {
    const [prettierMod, babelMod, estreeMod, htmlMod, cssMod, tsMod] = await Promise.all([
      import(`${PRETTIER_CDN}/standalone.mjs`),
      import(`${PRETTIER_CDN}/plugins/babel.mjs`),
      import(`${PRETTIER_CDN}/plugins/estree.mjs`),
      import(`${PRETTIER_CDN}/plugins/html.mjs`),
      import(`${PRETTIER_CDN}/plugins/postcss.mjs`),
      import(`${PRETTIER_CDN}/plugins/typescript.mjs`),
    ]);
    _prettier = prettierMod.default || prettierMod;
    _prettierPlugins = {
      babel: babelMod.default || babelMod,
      estree: estreeMod.default || estreeMod,
      html: htmlMod.default || htmlMod,
      postcss: cssMod.default || cssMod,
      typescript: tsMod.default || tsMod,
    };
    _prettierLoading = false;
    return true;
  } catch (err) {
    console.warn('[Prettier] Failed to load from CDN:', err.message);
    _prettierLoading = false;
    return false;
  }
}

const _PRETTIER_PARSERS = {
  javascript: 'babel',
  jsx: 'babel',
  typescript: 'typescript',
  tsx: 'typescript',
  html: 'html',
  htm: 'html',
  markup: 'html',
  css: 'css',
  scss: 'css',
  json: 'json',
};

// ---------------------------------------------------------------------------
// Ghost Text — LLM-powered inline autocomplete
// ---------------------------------------------------------------------------
// Shows semi-transparent suggestion text after the cursor. Tab to accept.
// Uses the user's selected model via /api/chat with a focused completion prompt.

let _ghostText = '';           // The current suggestion
let _ghostAbort = null;        // AbortController for pending request
let _ghostDebounce = null;     // Debounce timer
let _ghostPos = -1;            // Cursor position when suggestion was generated
let _ghostEl = null;           // The overlay element showing ghost text
let _ghostEnabled = true;      // Toggle via status bar

const GHOST_DEBOUNCE_MS = 600; // Wait after last keystroke before requesting
const GHOST_MAX_TOKENS = 80;   // Keep completions short for speed
const GHOST_CONTEXT_LINES = 30; // Lines of context before/after cursor

/**
 * Trigger ghost text generation after a typing pause.
 * Called from the input handler. Checks server-persisted setting.
 */
function _triggerGhostText() {
  // Check the persisted setting from settings.js
  const s = getSettings();
  _ghostEnabled = s.ghostTextEnabled === true;
  if (!_ghostEnabled || !_cmEditorId) return;
  _dismissGhost();

  clearTimeout(_ghostDebounce);
  _ghostDebounce = setTimeout(_requestGhostText, GHOST_DEBOUNCE_MS);
}

/**
 * Request a completion from the LLM.
 */
async function _requestGhostText() {
  // Get content and cursor from CM6 or textarea
  let val, cursorRow, cursorCol;
  if (_cmEditorId && _cmReady) {
    val = CMEditor.getContent(_cmEditorId);
    const cursor = CMEditor.getCursor(_cmEditorId);
    cursorRow = cursor.line - 1;
    cursorCol = cursor.col - 1;
  } else return;

  if (!val || val.length === 0) return;

  // Build context window around cursor
  const lines = val.split('\n');

  const startLine = Math.max(0, cursorRow - GHOST_CONTEXT_LINES);
  const endLine = Math.min(lines.length, cursorRow + GHOST_CONTEXT_LINES);

  const codeBefore = lines.slice(startLine, cursorRow).join('\n') +
    (cursorRow > startLine ? '\n' : '') +
    lines[cursorRow].slice(0, cursorCol);
  const codeAfter = lines[cursorRow].slice(cursorCol) +
    (cursorRow < endLine - 1 ? '\n' : '') +
    lines.slice(cursorRow + 1, endLine).join('\n');

  // Skip if cursor is on an empty line with nothing meaningful before it
  const lineBeforeCursor = lines[cursorRow].slice(0, cursorCol).trim();
  if (!lineBeforeCursor && cursorRow > 0) {
    // Empty line — only suggest if the previous line ended mid-statement
    const prevLine = lines[cursorRow - 1]?.trimEnd() || '';
    if (prevLine.endsWith(';') || prevLine.endsWith('}') || prevLine.endsWith(':') ||
        prevLine === '' || prevLine.startsWith('//') || prevLine.startsWith('#')) {
      return; // Natural stopping point — don't suggest
    }
  }

  // Cancel any pending request
  if (_ghostAbort) _ghostAbort.abort();
  _ghostAbort = new AbortController();

  const file = _files[_activeFile];
  const lang = file ? _getLangClass(file.path) : 'javascript';

  try {
    // Use dedicated ghost text model if configured, else current chat model
    const ghostModel = getSettings().ghostTextModel || _getCurrentModel();

    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: ghostModel,
        messages: [
          {
            role: 'system',
            content: `You are a code autocomplete engine. Complete the ${lang} code at the <CURSOR> position. Output ONLY the completion text — no markdown, no explanation, no backticks. If the completion is a full line, include the newline. Keep it short (1-3 lines max). If nothing natural to complete, output nothing.`
          },
          {
            role: 'user',
            content: `${codeBefore}<CURSOR>${codeAfter}`
          }
        ],
        stream: false,
        think: false,
        max_tokens: GHOST_MAX_TOKENS,
        temperature: 0.15,
      }),
      signal: _ghostAbort.signal,
    });

    if (!resp.ok) return;
    const data = await resp.json();
    let completion = data.choices?.[0]?.message?.content || data.content || '';

    // Clean up: strip markdown fences, leading/trailing whitespace artifacts
    completion = completion
      .replace(/^```[\w]*\n?/, '').replace(/\n?```$/, '') // strip fences
      .replace(/^<CURSOR>/, '')  // some models echo the marker
      .replace(/^\n/, '');       // strip leading blank line

    if (!completion || completion.length < 2) return;

    // Only show if cursor hasn't moved since we started
    if (ta.selectionStart !== pos) return;

    _ghostText = completion;
    _ghostPos = pos;
    _showGhost();
  } catch (err) {
    if (err.name !== 'AbortError') console.debug('[Ghost] request failed:', err.message);
  }
}

/**
 * Show ghost text via CM6 decoration widget.
 */
function _showGhost() {
  if (!_ghostText || !_cmEditorId) return;
  CMEditor.showGhostText(_cmEditorId, _ghostText);
}

/**
 * Accept the current ghost text — CM6 inserts it at cursor.
 */
function _acceptGhost() {
  if (!_ghostText || !_cmEditorId) return;
  CMEditor.acceptGhostText(_cmEditorId);
  _ghostText = '';
  _ghostPos = -1;
}

/**
 * Dismiss ghost text.
 */
function _dismissGhost() {
  _ghostText = '';
  _ghostPos = -1;
  clearTimeout(_ghostDebounce);
  if (_ghostAbort) { _ghostAbort.abort(); _ghostAbort = null; }
  if (_cmEditorId) CMEditor.dismissGhostText(_cmEditorId);
}

// --- In-Editor Find & Replace ---
function _showFindBar(withReplace = false) {
  if (_findBar) {
    // If already open, toggle replace row visibility or just focus
    const replaceRow = _findBar.querySelector('.workspace-replace-row');
    if (withReplace && replaceRow) replaceRow.style.display = 'flex';
    _findBar.querySelector('input')?.focus();
    return;
  }

  let _findUseRegex = false;
  let _findAllFiles = false;

  _findBar = document.createElement('div');
  _findBar.className = 'workspace-find-bar';
  _findBar.innerHTML = `
    <div class="workspace-find-row">
      <input type="text" class="workspace-find-input" placeholder="Find\u2026" spellcheck="false" />
      <span class="workspace-find-count"></span>
      <button class="workspace-find-btn find-toggle-regex" data-action="toggle-regex" title="Use Regular Expression">.*</button>
      <button class="workspace-find-btn find-toggle-all" data-action="toggle-all-files" title="Search all files">All</button>
      <button class="workspace-find-btn" data-action="prev" title="Previous (Shift+Enter)">&#9650;</button>
      <button class="workspace-find-btn" data-action="next" title="Next (Enter)">&#9660;</button>
      <button class="workspace-find-btn" data-action="toggle-replace" title="Toggle Replace (Ctrl+H)">&#8644;</button>
      <button class="workspace-find-btn" data-action="close" title="Close (Escape)">&times;</button>
    </div>
    <div class="workspace-replace-row" style="display:${withReplace ? 'flex' : 'none'}">
      <input type="text" class="workspace-find-input workspace-replace-input" placeholder="Replace\u2026" spellcheck="false" />
      <button class="workspace-find-btn" data-action="replace" title="Replace current">Replace</button>
      <button class="workspace-find-btn" data-action="replace-all" title="Replace all">All</button>
    </div>
    <div class="workspace-find-all-results" style="display:none"></div>
  `;

  const editorArea = _getEditorArea();
  if (editorArea) editorArea.insertBefore(_findBar, editorArea.firstChild);

  const input = _findBar.querySelector('.workspace-find-row input');
  const replaceInput = _findBar.querySelector('.workspace-replace-input');
  const countEl = _findBar.querySelector('.workspace-find-count');
  const replaceRow = _findBar.querySelector('.workspace-replace-row');

  // Pre-fill with current selection
  // Pre-fill from CM6 selection
  const cmView = _cmEditorId ? CMEditor.getView(_cmEditorId) : null;
  const cmSel = cmView?.state?.selection?.main;
  if (cmSel && cmSel.from !== cmSel.to) {
    const sel = cmView.state.doc.sliceString(cmSel.from, cmSel.to);
    if (sel.length < 200 && !sel.includes('\n')) input.value = sel;
  }
  input.focus();
  if (input.value) input.dispatchEvent(new Event('input'));

  const allResultsEl = _findBar.querySelector('.workspace-find-all-results');

  function _runSearch() {
    const query = input.value;
    if (!query) { _findMatches = []; _findIdx = -1; countEl.textContent = ''; if (allResultsEl) allResultsEl.style.display = 'none'; return; }

    // Search in current file
    _findMatches = [];
    const val = _getEditorContent();

    if (_findUseRegex) {
      // Regex search
      try {
        const re = new RegExp(query, 'gi');
        let m;
        while ((m = re.exec(val)) !== null) {
          _findMatches.push(m.index);
          if (m.index === re.lastIndex) re.lastIndex++; // prevent infinite loop on zero-length matches
        }
      } catch {
        countEl.textContent = 'Invalid regex';
        return;
      }
    } else {
      // Plain text search (case-insensitive)
      const lower = val.toLowerCase();
      const q = query.toLowerCase();
      let idx = lower.indexOf(q);
      while (idx !== -1) {
        _findMatches.push(idx);
        idx = lower.indexOf(q, idx + 1);
      }
    }

    _findIdx = _findMatches.length > 0 ? 0 : -1;
    countEl.textContent = _findMatches.length > 0 ? `${_findIdx + 1}/${_findMatches.length}` : 'No results';
    if (_findIdx >= 0) _jumpToMatch(_findUseRegex ? query.length : query.length);

    // Cross-file search results
    if (_findAllFiles && allResultsEl) {
      _runCrossFileSearch(query, allResultsEl);
    } else if (allResultsEl) {
      allResultsEl.style.display = 'none';
    }
  }

  function _runCrossFileSearch(query, resultsEl) {
    const results = [];
    for (let fi = 0; fi < _files.length; fi++) {
      if (fi === _activeFile) continue; // skip current file (already shown in main results)
      const f = _files[fi];
      const content = f.content || '';
      const lines = content.split('\n');
      let matchCount = 0;
      const matchLines = [];

      for (let li = 0; li < lines.length && matchLines.length < 5; li++) {
        const lineLC = lines[li].toLowerCase();
        const queryLC = query.toLowerCase();
        if (_findUseRegex ? new RegExp(query, 'i').test(lines[li]) : lineLC.includes(queryLC)) {
          matchCount++;
          matchLines.push({ line: li + 1, text: lines[li].trim().slice(0, 80) });
        }
      }
      if (matchCount > 0) {
        results.push({ fileIdx: fi, path: f.path, matchCount, matchLines });
      }
    }

    if (results.length === 0) { resultsEl.style.display = 'none'; return; }

    resultsEl.style.display = '';
    resultsEl.innerHTML = results.map(r =>
      `<div class="find-all-file">
        <div class="find-all-file-header" data-file-idx="${r.fileIdx}">
          <strong>${escapeHtml(r.path)}</strong> <span class="find-all-count">${r.matchCount} match${r.matchCount > 1 ? 'es' : ''}</span>
        </div>
        ${r.matchLines.map(ml =>
          `<div class="find-all-line" data-file-idx="${r.fileIdx}" data-line="${ml.line}">
            <span class="find-all-line-num">${ml.line}</span> ${escapeHtml(ml.text)}
          </div>`
        ).join('')}
      </div>`
    ).join('');

    // Click to jump to file + line
    resultsEl.addEventListener('click', (e) => {
      const lineEl = e.target.closest('.find-all-line[data-file-idx]');
      const fileEl = e.target.closest('.find-all-file-header[data-file-idx]');
      const targetEl = lineEl || fileEl;
      if (!targetEl) return;

      const fileIdx = parseInt(targetEl.dataset.fileIdx, 10);
      if (isNaN(fileIdx) || !_files[fileIdx]) return;

      _activeFile = fileIdx;
      _loadFileIntoEditor(fileIdx);
      _renderFileTabs();

      if (lineEl) {
        const line = parseInt(lineEl.dataset.line, 10);
        if (!isNaN(line) && _cmEditorId) {
          CMEditor.setCursor(_cmEditorId, line, 1);
        
        }
      }

      // Re-run search in the new file
      _runSearch();
    });
  }

  input.addEventListener('input', _runSearch);

  function _navNext() {
    if (_findMatches.length === 0) return;
    _findIdx = (_findIdx + 1) % _findMatches.length;
    countEl.textContent = `${_findIdx + 1}/${_findMatches.length}`;
    _jumpToMatch(input.value.length);
  }
  function _navPrev() {
    if (_findMatches.length === 0) return;
    _findIdx = (_findIdx - 1 + _findMatches.length) % _findMatches.length;
    countEl.textContent = `${_findIdx + 1}/${_findMatches.length}`;
    _jumpToMatch(input.value.length);
  }

  // Handle keyboard in both inputs
  function _onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); _navNext(); }
    if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); _navPrev(); }
    if (e.key === 'Escape') { _closeFindBar(); }
    // Ctrl+H toggles replace row
    if ((e.ctrlKey || e.metaKey) && e.key === 'h') {
      e.preventDefault();
      replaceRow.style.display = replaceRow.style.display === 'none' ? 'flex' : 'none';
      if (replaceRow.style.display === 'flex') replaceInput.focus();
    }
  }
  input.addEventListener('keydown', _onKeyDown);
  replaceInput.addEventListener('keydown', _onKeyDown);

  // Replace current match
  function _replaceCurrent() {
    if (_findIdx < 0) return;
    const val = _getEditorContent();
    const pos = _findMatches[_findIdx];
    const query = input.value;
    const replacement = replaceInput.value;
    _setEditorContent(val.slice(0, pos) + replacement + val.slice(pos + query.length));
    // Re-run search to update matches
    _runSearch();
  }

  // Replace all
  function _replaceAll() {
    if (!input.value) return;
    const query = input.value;
    const replacement = replaceInput.value;
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const val = _getEditorContent();
    _setEditorContent(_findUseRegex
      ? val.replace(new RegExp(query, 'gi'), replacement)
      : val.replace(new RegExp(escaped, 'gi'), replacement));
    _runSearch();
  }

  _findBar.querySelector('[data-action="next"]').addEventListener('click', _navNext);
  _findBar.querySelector('[data-action="prev"]').addEventListener('click', _navPrev);
  _findBar.querySelector('[data-action="close"]').addEventListener('click', _closeFindBar);
  _findBar.querySelector('[data-action="toggle-replace"]').addEventListener('click', () => {
    replaceRow.style.display = replaceRow.style.display === 'none' ? 'flex' : 'none';
    if (replaceRow.style.display === 'flex') replaceInput.focus();
  });
  _findBar.querySelector('[data-action="replace"]').addEventListener('click', _replaceCurrent);
  _findBar.querySelector('[data-action="replace-all"]').addEventListener('click', _replaceAll);

  // Regex toggle
  const regexBtn = _findBar.querySelector('[data-action="toggle-regex"]');
  regexBtn.addEventListener('click', () => {
    _findUseRegex = !_findUseRegex;
    regexBtn.classList.toggle('active', _findUseRegex);
    _runSearch();
  });

  // All-files toggle
  const allFilesBtn = _findBar.querySelector('[data-action="toggle-all-files"]');
  allFilesBtn.addEventListener('click', () => {
    _findAllFiles = !_findAllFiles;
    allFilesBtn.classList.toggle('active', _findAllFiles);
    _runSearch();
  });
}

function _jumpToMatch(queryLen) {
  if (_findIdx < 0) return;
  const pos = _findMatches[_findIdx];
  // Select the match in CM6
  if (_cmEditorId) {
    const view = CMEditor.getView(_cmEditorId);
    if (view) {
      view.dispatch({ selection: { anchor: pos, head: pos + queryLen } });
      view.focus();
      // Scroll into view
      const lineNum = view.state.doc.lineAt(pos).number;
      CMEditor.setCursor(_cmEditorId, lineNum, 1);
    }
  }
}

function _closeFindBar() {
  if (_findBar) { _findBar.remove(); _findBar = null; }
  _findMatches = [];
  _findIdx = -1;
  if (_cmEditorId) CMEditor.focus(_cmEditorId);
}

// --- New File ---
function _addNewFile() {
  const name = prompt('File name (e.g. utils.js):');
  if (!name || !name.trim()) return;
  const path = name.trim();
  // Determine role from extension
  let role = 'script';
  if (path.endsWith('.html') || path.endsWith('.htm')) role = 'entry';
  else if (path.endsWith('.css')) role = 'style';
  else if (path.endsWith('.json')) role = 'data';

  _files.push({ path, role, content: '' });
  _activeFile = _files.length - 1;
  _modified.add(path);
  _loadEditor();
  if (_cmEditorId) CMEditor.focus(_cmEditorId);
}

// --- Title ---
function _updateTitle() {
  if (_el.title) _el.title.textContent = _artifact?.display_name || _artifact?.filename || 'Untitled';
}

// --- Events ---
let _eventsWired = false;
function _wireEvents() {
  if (_eventsWired) return;
  _eventsWired = true;

  // Back → return to library (keep library overlay open)
  _el.back?.addEventListener('click', () => {
    _teardown();
  });
  // Close → close workspace AND library, return to chat
  _el.close?.addEventListener('click', () => {
    _teardown();
    import('./library.js').then(m => m.closeLibrary()).catch(() => {});
  });

  // Mode toggle
  _el.modeToggle?.querySelectorAll('.workspace-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => _setMode(btn.dataset.mode));
  });

  // File tree toggle
  document.getElementById('workspace-file-tree-toggle')?.addEventListener('click', _toggleFileTree);
  document.getElementById('file-tree-close')?.addEventListener('click', _toggleFileTree);

  // Header auto-hide (play mode) — show on mouse near top
  _el.workspace?.addEventListener('mousemove', (e) => {
    if (_mode !== 'play') return;
    if (e.clientY < 60) {
      _el.header?.classList.remove('auto-hidden');
      _clearHeaderTimer();
    } else {
      _startHeaderAutoHide();
    }
  });

  // Console messages from iframe
  window.addEventListener('message', (e) => {
    if (!_open) return;
    // Validate source is our iframe
    if (_el.iframe && e.source !== _el.iframe.contentWindow) return;
    if (e.data?.type === 'code-console' || e.data?.type === 'code-error') {
      _addConsoleEntry(_normalizeConsoleMessage(e.data));
    }
  });

  // Console clear
  _el.consoleClear?.addEventListener('click', () => _clearConsoleEntries());

  // Console toggle
  _el.consoleToggle?.addEventListener('click', () => {
    _el.console?.classList.toggle('collapsed');
    _syncConsoleState();
  });
  _el.console?.querySelector('.workspace-console-header')?.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    _el.console?.classList.toggle('collapsed');
    _syncConsoleState();
  });

  // AI prompt send
  _el.promptSend?.addEventListener('click', _onPromptSend);
  _el.promptInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); _onPromptSend(); }
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (!_open) return;
    // Ctrl+F — in-editor find (prevent browser find)
    if ((e.ctrlKey || e.metaKey) && e.key === 'f' && _mode === 'work') {
      e.preventDefault();
      _showFindBar();
      return;
    }
    if (e.key === 'Escape') {
      if (_mode === 'play') {
        _el.header?.classList.remove('auto-hidden');
        _clearHeaderTimer();
      }
    }
    if (e.key === '`' && !e.ctrlKey && !e.metaKey && !_isEditorFocused()) {
      e.preventDefault();
      _el.promptInput?.focus();
    }
    // Hunk navigation (when diff overlay is visible and not editing code)
    if (_pendingHunks.length > 0 && !_isEditorFocused()) {
      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        _focusedHunkIdx = Math.min(_focusedHunkIdx + 1, _pendingHunks.length - 1);
        _updateHunkFocus();
        return;
      }
      if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        _focusedHunkIdx = Math.max(_focusedHunkIdx - 1, 0);
        _updateHunkFocus();
        return;
      }
      if (e.key === 'a' && _focusedHunkIdx >= 0) {
        _setHunkStatus(_pendingHunks[_focusedHunkIdx].id, 'accepted');
        return;
      }
      if (e.key === 'r' && _focusedHunkIdx >= 0) {
        _setHunkStatus(_pendingHunks[_focusedHunkIdx].id, 'rejected');
        return;
      }
      if (e.key === 'Enter' && e.shiftKey) {
        e.preventDefault();
        _onAcceptAll();
        return;
      }
    }
  });

  // Accept/Reject/Confirm buttons
  _el.acceptBar?.querySelector('.accept-all')?.addEventListener('click', _onAcceptAll);
  _el.acceptBar?.querySelector('.reject-all')?.addEventListener('click', _onRejectAll);
  _el.learnDiff?.addEventListener('click', _showDiffCoach);

  // Resize handle
  _wireResize();
}

// --- Hunk Parser ---
function _normalizePatchFilename(filename) {
  return String(filename || '')
    .trim()
    .replace(/^file:\s*/i, '')
    .replace(/^[`'"]|[`'"]$/g, '')
    .replace(/\s*\((?:FULL|full|signature|REFERENCE ONLY:[^)]+)\)\s*$/i, '')
    .replace(/^\.\//, '')
    .trim();
}

function _parseRawResponseIntoHunks(rawResponse, snapshotFiles) {
  const hunks = [];
  let hunkId = 0;

  // Try parsing === FILE: <name> === sections first
  const fileSections = rawResponse.split(/\n?===\s*FILE:\s*/i).slice(1);

  if (fileSections.length > 0) {
    for (const section of fileSections) {
      const nameEnd = section.indexOf('===');
      if (nameEnd < 0) continue;
      const fileName = _normalizePatchFilename(section.slice(0, nameEnd));
      const body = section.slice(nameEnd + 3);
      _extractHunksFromBody(body, fileName, snapshotFiles, hunks, hunkId);
      hunkId = hunks.length;
    }
  } else {
    // No FILE sections — extract bare SEARCH/REPLACE blocks
    _extractHunksFromBody(rawResponse, null, snapshotFiles, hunks, hunkId);
  }

  return hunks;
}

function _extractHunksFromBody(body, fileName, snapshotFiles, hunks, startId) {
  const regex = /<<<<<<<?\.?\s*SEARCH\n([\s\S]*?)\n?={3,}\n([\s\S]*?)\n?>>>>>>>?\.?\s*REPLACE/gi;
  let match;

  while ((match = regex.exec(body)) !== null) {
    const search = match[1].replace(/\r\n/g, '\n').replace(/\s+$/, '');
    const replace = match[2].replace(/\r\n/g, '\n').replace(/\s+$/, '');

    // Determine which file this hunk belongs to
    let file = _normalizePatchFilename(fileName);
    if (!file) {
      // Try to find the file by searching content
      for (const f of snapshotFiles) {
        if (f.content && f.content.includes(search.trim().split('\n')[0])) {
          file = f.path;
          break;
        }
      }
      if (!file && snapshotFiles.length > 0) file = snapshotFiles[0].path;
    }

    // Find match location in the snapshot file
    const sourceFile = snapshotFiles.find(f => f.path === file);
    let matchStart = -1;
    if (sourceFile) {
      const idx = sourceFile.content.indexOf(search);
      if (idx >= 0) {
        matchStart = sourceFile.content.slice(0, idx).split('\n').length;
      } else {
        // Try trimmed matching
        const searchLines = search.split('\n').map(l => l.trim());
        const contentLines = sourceFile.content.split('\n');
        for (let i = 0; i <= contentLines.length - searchLines.length; i++) {
          let found = true;
          for (let j = 0; j < searchLines.length; j++) {
            if (contentLines[i + j].trim() !== searchLines[j]) { found = false; break; }
          }
          if (found) { matchStart = i + 1; break; }
        }
      }
    }
    if (matchStart <= 0) {
      console.warn('[workspace] Skipping unmatched SEARCH/REPLACE hunk', {
        file,
        firstLine: search.trim().split('\n')[0] || '',
      });
      continue;
    }

    // Compute diff lines (search vs replace)
    const searchLines = search.split('\n');
    const replaceLines = replace.split('\n');
    const diffLines = _computeLineDiff(searchLines, replaceLines);

    // Add context lines from source file (3 before, 3 after)
    if (sourceFile && matchStart > 0) {
      const allLines = sourceFile.content.split('\n');
      const contextBefore = [];
      const contextAfter = [];
      for (let i = Math.max(0, matchStart - 4); i < matchStart - 1; i++) {
        contextBefore.push({ type: 'context', line: allLines[i] });
      }
      const endLine = matchStart - 1 + searchLines.length;
      for (let i = endLine; i < Math.min(allLines.length, endLine + 3); i++) {
        contextAfter.push({ type: 'context', line: allLines[i] });
      }
      diffLines.unshift(...contextBefore);
      diffLines.push(...contextAfter);
    }

    hunks.push({
      id: `hunk-${startId + hunks.length}`,
      file,
      search,
      replace,
      status: 'pending',
      diffLines,
      matchStart,
      matchEnd: matchStart > 0 ? matchStart + searchLines.length - 1 : -1,
    });
  }
}

// --- Hunk Diff Rendering ---
function _renderHunkDiff(filePath) {
  const fileHunks = _pendingHunks.filter(h => h.file === filePath);

  let diffContainer = _el.workspace?.querySelector('.workspace-diff-overlay');
  if (!diffContainer) {
    diffContainer = document.createElement('div');
    diffContainer.className = 'workspace-diff-overlay';
    const editorArea = _getEditorArea();
    if (editorArea) editorArea.appendChild(diffContainer);
  }
  const codeContainer = _el.cmContainer;
  if (codeContainer) codeContainer.style.display = 'none';
  diffContainer.style.display = '';

  if (fileHunks.length === 0) {
    diffContainer.innerHTML = `<div class="diff-empty"><span class="diff-empty-icon">&#128270;</span><span>No changes in ${escapeHtml(filePath)}</span></div>`;
    return;
  }

  const scrollDiv = document.createElement('div');
  scrollDiv.className = 'diff-scroll';

  for (let i = 0; i < fileHunks.length; i++) {
    const hunk = fileHunks[i];
    const hunkEl = _createHunkElement(hunk);
    scrollDiv.appendChild(hunkEl);
    if (i < fileHunks.length - 1) {
      const sep = document.createElement('div');
      sep.className = 'diff-hunk-separator';
      sep.textContent = '\u22EF';
      scrollDiv.appendChild(sep);
    }
  }

  diffContainer.innerHTML = '';
  diffContainer.appendChild(scrollDiv);
}

async function _animateHunks(filePath) {
  const fileHunks = _pendingHunks.filter(h => h.file === filePath);
  if (!fileHunks.length) { _renderHunkDiff(filePath); return; }

  _animating = true;

  // Set up the diff overlay container
  let diffContainer = _el.workspace?.querySelector('.workspace-diff-overlay');
  if (!diffContainer) {
    diffContainer = document.createElement('div');
    diffContainer.className = 'workspace-diff-overlay';
    const editorArea = _getEditorArea();
    if (editorArea) editorArea.appendChild(diffContainer);
  }
  const codeContainer = _el.cmContainer;
  if (codeContainer) codeContainer.style.display = 'none';
  diffContainer.style.display = '';

  // Click anywhere in diff overlay to skip animation — instantly reveal all hidden lines
  diffContainer.addEventListener('click', () => {
    if (_animating) {
      _animating = false;
      diffContainer.querySelectorAll('.diff-row-hidden').forEach(r => r.classList.remove('diff-row-hidden'));
    }
  });

  const scrollDiv = document.createElement('div');
  scrollDiv.className = 'diff-scroll';
  diffContainer.innerHTML = '';
  diffContainer.appendChild(scrollDiv);

  // Add a blinking cursor element
  const cursor = document.createElement('div');
  cursor.className = 'diff-cursor';

  for (let i = 0; i < fileHunks.length; i++) {
    if (!_animating) break; // cancelled

    const hunk = fileHunks[i];
    const hunkEl = _createHunkElement(hunk);

    // Hide all diff rows initially
    const rows = hunkEl.querySelectorAll('.diff-row');
    rows.forEach(row => row.classList.add('diff-row-hidden'));

    // Add hunk with slide-in animation
    hunkEl.classList.add('diff-hunk-entering');
    scrollDiv.appendChild(hunkEl);

    // Move cursor into hunk body
    const body = hunkEl.querySelector('.diff-hunk-body');
    if (body) body.appendChild(cursor);

    // Reveal rows one by one
    for (const row of rows) {
      if (!_animating) break;
      await _delay(30);
      row.classList.remove('diff-row-hidden');
      hunkEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Pause between hunks
    if (i < fileHunks.length - 1) {
      const sep = document.createElement('div');
      sep.className = 'diff-hunk-separator';
      sep.textContent = '\u22EF';
      scrollDiv.appendChild(sep);
      await _delay(150);
    }

    // Debounce preview reload after each hunk
    _debouncePreviewDuringAnimation();
  }

  // Remove cursor, animation done
  cursor.remove();
  _animating = false;

  // Final preview reload
  const previewFiles = _applyAcceptedHunks();
  _files = previewFiles;
  _loadPreview();

  // Show accept bar
  _el.acceptBar?.classList.remove('hidden');
  _updateAcceptBarCounter();
}

function _delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function _debouncePreviewDuringAnimation() {
  clearTimeout(_animPreviewTimer);
  _animPreviewTimer = setTimeout(() => {
    const previewFiles = _applyAcceptedHunks();
    _files = previewFiles;
    _loadPreview();
  }, 500);
}

function _createHunkElement(hunk) {
  const div = document.createElement('div');
  div.className = 'diff-hunk';
  div.dataset.hunkId = hunk.id;
  div.dataset.status = hunk.status;

  const rangeText = hunk.matchStart > 0 ? `@@ line ${hunk.matchStart} @@` : '@@ @@';

  div.innerHTML = `
    <div class="diff-hunk-header">
      <span class="diff-hunk-file">${escapeHtml(hunk.file)}</span>
      <span class="diff-hunk-range">${rangeText}</span>
      <span class="diff-hunk-spacer"></span>
      <button class="diff-hunk-btn diff-hunk-learn" title="Learn this change">?</button>
      <button class="diff-hunk-btn diff-hunk-accept" title="Accept (a)">&#10003;</button>
      <button class="diff-hunk-btn diff-hunk-reject" title="Reject (r)">&#10005;</button>
    </div>
    <div class="diff-hunk-body">${_renderHunkLines(hunk)}</div>
  `;

  // Wire accept/reject
  div.querySelector('.diff-hunk-learn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    _showHunkCoach(hunk);
  });
  div.querySelector('.diff-hunk-accept')?.addEventListener('click', (e) => {
    e.stopPropagation();
    _setHunkStatus(hunk.id, 'accepted');
  });
  div.querySelector('.diff-hunk-reject')?.addEventListener('click', (e) => {
    e.stopPropagation();
    _setHunkStatus(hunk.id, 'rejected');
  });

  // Click hunk to focus it
  div.addEventListener('click', () => {
    _focusedHunkIdx = _pendingHunks.findIndex(h => h.id === hunk.id);
    _updateHunkFocus();
  });

  return div;
}

function _renderHunkLines(hunk) {
  let oldNum = hunk.matchStart > 0 ? hunk.matchStart - 3 : 1;
  let newNum = oldNum;
  if (oldNum < 1) oldNum = 1;
  if (newNum < 1) newNum = 1;

  const entries = hunk.diffLines;

  // Pre-compute intra-line diffs for paired remove→add lines
  for (let i = 0; i < entries.length - 1; i++) {
    if (entries[i].type === 'remove' && entries[i + 1].type === 'add') {
      const result = _intraLineDiffWs(entries[i].line, entries[i + 1].line);
      if (result.oldHtml) {
        entries[i]._intraHtml = result.oldHtml;
        entries[i + 1]._intraHtml = result.newHtml;
      }
    }
  }

  return entries.map(entry => {
    const content = entry._intraHtml || escapeHtml(entry.line);
    if (entry.type === 'context') {
      const ln = oldNum++;
      newNum++;
      return `<div class="diff-row"><span class="diff-ln diff-ln-old">${ln}</span><span class="diff-ln diff-ln-new">${ln}</span><span class="diff-sigil"> </span><span class="diff-content">${content}</span></div>`;
    } else if (entry.type === 'remove') {
      const ln = oldNum++;
      return `<div class="diff-row diff-row-remove"><span class="diff-ln diff-ln-old">${ln}</span><span class="diff-ln diff-ln-new"></span><span class="diff-sigil">-</span><span class="diff-content">${content}</span></div>`;
    } else if (entry.type === 'add') {
      const ln = newNum++;
      return `<div class="diff-row diff-row-add"><span class="diff-ln diff-ln-old"></span><span class="diff-ln diff-ln-new">${ln}</span><span class="diff-sigil">+</span><span class="diff-content">${content}</span></div>`;
    } else {
      const ln = oldNum++;
      newNum++;
      return `<div class="diff-row"><span class="diff-ln diff-ln-old">${ln}</span><span class="diff-ln diff-ln-new">${ln}</span><span class="diff-sigil"> </span><span class="diff-content">${content}</span></div>`;
    }
  }).join('');
}

/** Intra-line diff for workspace (same algorithm as chat, uses escapeHtml) */
function _intraLineDiffWs(oldLine, newLine) {
  let prefixLen = 0;
  const minLen = Math.min(oldLine.length, newLine.length);
  while (prefixLen < minLen && oldLine[prefixLen] === newLine[prefixLen]) prefixLen++;
  let suffixLen = 0;
  while (suffixLen < (minLen - prefixLen) &&
    oldLine[oldLine.length - 1 - suffixLen] === newLine[newLine.length - 1 - suffixLen]) suffixLen++;
  const oldChanged = oldLine.slice(prefixLen, oldLine.length - suffixLen);
  const newChanged = newLine.slice(prefixLen, newLine.length - suffixLen);
  if (prefixLen + suffixLen < 3 && oldChanged.length > oldLine.length * 0.7) {
    return { oldHtml: null, newHtml: null };
  }
  return {
    oldHtml: oldChanged.length > 0
      ? `${escapeHtml(oldLine.slice(0, prefixLen))}<span class="diff-highlight">${escapeHtml(oldChanged)}</span>${escapeHtml(oldLine.slice(oldLine.length - suffixLen))}`
      : `${escapeHtml(oldLine)}`,
    newHtml: newChanged.length > 0
      ? `${escapeHtml(newLine.slice(0, prefixLen))}<span class="diff-highlight">${escapeHtml(newChanged)}</span>${escapeHtml(newLine.slice(newLine.length - suffixLen))}`
      : `${escapeHtml(newLine)}`,
  };
}

// --- File-level Status ---

function _getFileStatus(filePath) {
  const fileHunks = _pendingHunks.filter(h => h.file === filePath);
  if (!fileHunks.length) return 'unchanged';
  if (fileHunks.every(h => h.status === 'accepted')) return 'accepted';
  if (fileHunks.every(h => h.status === 'rejected')) return 'rejected';
  if (fileHunks.every(h => h.status !== 'pending')) return 'mixed';
  return 'pending';
}

// --- Hunk Status Management ---
let _previewReloadTimer = null;

function _setHunkStatus(hunkId, status) {
  if (_animating) {
    _animating = false;
    // Render all remaining hunks instantly
    const filePath = _files[_activeFile]?.path || _pendingHunks[0]?.file;
    setTimeout(() => _renderHunkDiff(filePath), 50);
  }
  const hunk = _pendingHunks.find(h => h.id === hunkId);
  if (!hunk) return;
  hunk.status = status;

  // Update DOM
  const hunkEl = _el.workspace?.querySelector(`[data-hunk-id="${hunkId}"]`);
  if (hunkEl) hunkEl.dataset.status = status;

  // Update accept bar counter
  _updateAcceptBarCounter();

  // Update file tab indicators
  _renderFileTabs();

  // Check if all hunks are resolved — update preview, then auto-save after a beat
  const allResolved = _pendingHunks.every(h => h.status !== 'pending');
  if (allResolved) {
    _files = _applyAcceptedHunks();
    _loadPreview();
    // Let the user see the final preview for 1.5s before saving
    clearTimeout(_previewReloadTimer);
    _previewReloadTimer = setTimeout(() => _finalizeDiff(), 1500);
    return;
  }

  // Debounce preview reload showing current accepted state
  clearTimeout(_previewReloadTimer);
  _previewReloadTimer = setTimeout(() => {
    const previewFiles = _applyAcceptedHunks();
    _files = previewFiles;
    _loadPreview();
  }, 300);
}

function _updateAcceptBarCounter() {
  const total = _pendingHunks.length;
  const countEl = document.getElementById('workspace-hunk-count');
  if (total === 0) {
    if (countEl) countEl.textContent = 'Whole edit preview';
    return;
  }
  const accepted = _pendingHunks.filter(h => h.status === 'accepted').length;
  const rejected = _pendingHunks.filter(h => h.status === 'rejected').length;
  const pending = total - accepted - rejected;

  if (countEl) countEl.textContent = `${accepted}/${total} accepted` + (pending > 0 ? ` \u00B7 ${pending} pending` : '');
}

function _updateHunkFocus() {
  _el.workspace?.querySelectorAll('.diff-hunk').forEach(el => el.classList.remove('diff-hunk-focused'));
  if (_focusedHunkIdx >= 0 && _focusedHunkIdx < _pendingHunks.length) {
    const id = _pendingHunks[_focusedHunkIdx].id;
    const el = _el.workspace?.querySelector(`[data-hunk-id="${id}"]`);
    if (el) {
      el.classList.add('diff-hunk-focused');
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }
}

// --- Selective Hunk Application ---
function _rebaseReplacementLines(replace, baseIndent) {
  return replace.split('\n').map(line => line.trim() ? baseIndent + line.trimStart() : line);
}

function _applyHunkPatch(content, search, replace) {
  if (content.includes(search)) {
    return { content: content.replace(search, replace), applied: true };
  }

  const searchLines = search.split('\n');
  const contentLines = content.split('\n');
  const trimmed = searchLines.map(line => line.trim());

  if (searchLines.length <= contentLines.length) {
    for (let i = 0; i <= contentLines.length - searchLines.length; i++) {
      let found = true;
      for (let j = 0; j < searchLines.length; j++) {
        if (contentLines[i + j].trim() !== trimmed[j]) {
          found = false;
          break;
        }
      }
      if (found) {
        const baseIndent = contentLines[i].match(/^\s*/)?.[0] || '';
        const rebased = _rebaseReplacementLines(replace, baseIndent);
        contentLines.splice(i, searchLines.length, ...rebased);
        return { content: contentLines.join('\n'), applied: true };
      }
    }
  }

  const searchNorm = search.trim().replace(/\s+/g, ' ');
  for (let i = 0; i < contentLines.length; i++) {
    const maxSpan = Math.min(searchLines.length + 2, contentLines.length - i);
    for (let span = 1; span <= maxSpan; span++) {
      const chunk = contentLines.slice(i, i + span).join('\n');
      if (chunk.trim().replace(/\s+/g, ' ') === searchNorm) {
        const baseIndent = contentLines[i].match(/^\s*/)?.[0] || '';
        const rebased = _rebaseReplacementLines(replace, baseIndent);
        contentLines.splice(i, span, ...rebased);
        return { content: contentLines.join('\n'), applied: true };
      }
    }
  }

  return { content, applied: false };
}

function _applyAcceptedHunks() {
  if (!_patchedFiles.length || !_snapshot?.length) return _files;
  const hasReviewHunks = _pendingHunks.length > 0;

  // Start from snapshot (original state)
  const result = _snapshot.map(f => ({ ...f, content: f.content }));

  // For each file, determine which hunks are accepted
  for (const file of result) {
    const fileHunks = _pendingHunks.filter(h => h.file === file.path);
    const acceptedHunks = fileHunks.filter(h => h.status === 'accepted');

    if (hasReviewHunks && fileHunks.length === 0) {
      continue;
    }

    if (acceptedHunks.length === fileHunks.length) {
      // All accepted — use the backend's fully-patched version
      const patched = _patchedFiles.find(f => f.path === file.path);
      if (patched) file.content = patched.content;
    } else if (acceptedHunks.length === 0) {
      // All rejected — keep snapshot (already the case)
    } else {
      // Mixed — apply only accepted hunks from snapshot
      for (const hunk of acceptedHunks) {
        const result = _applyHunkPatch(file.content, hunk.search, hunk.replace);
        file.content = result.content;
        if (!result.applied) console.warn('[workspace] Accepted hunk no longer matched during selective apply', hunk.file);
      }
    }
  }

  return result;
}

function _hideDiffView() {
  const diffContainer = _el.workspace?.querySelector('.workspace-diff-overlay');
  if (diffContainer) diffContainer.style.display = 'none';
  const codeContainer = _el.cmContainer;
  if (codeContainer) codeContainer.style.display = '';
  // CM6 handles line numbers natively — no separate gutter to restore
}

function _computeLineDiff(oldLines, newLines) {
  // Simple LCS-based diff
  const m = oldLines.length, n = newLines.length;

  // For large files, use a faster greedy approach
  if (m + n > 2000) return _greedyDiff(oldLines, newLines);

  // Build LCS table
  const dp = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = oldLines[i - 1] === newLines[j - 1]
        ? dp[i - 1][j - 1] + 1
        : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  // Backtrack
  const result = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldLines[i - 1] === newLines[j - 1]) {
      result.unshift({ type: 'equal', line: newLines[j - 1] });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      result.unshift({ type: 'add', line: newLines[j - 1] });
      j--;
    } else {
      result.unshift({ type: 'remove', line: oldLines[i - 1] });
      i--;
    }
  }
  return result;
}

function _greedyDiff(oldLines, newLines) {
  // Fast line-by-line comparison for large files
  const result = [];
  const oldSet = new Set(oldLines);
  const newSet = new Set(newLines);
  let oi = 0, ni = 0;
  while (oi < oldLines.length || ni < newLines.length) {
    if (oi < oldLines.length && ni < newLines.length && oldLines[oi] === newLines[ni]) {
      result.push({ type: 'equal', line: newLines[ni] });
      oi++; ni++;
    } else if (oi < oldLines.length && !newSet.has(oldLines[oi])) {
      result.push({ type: 'remove', line: oldLines[oi] });
      oi++;
    } else if (ni < newLines.length && !oldSet.has(newLines[ni])) {
      result.push({ type: 'add', line: newLines[ni] });
      ni++;
    } else if (ni < newLines.length) {
      result.push({ type: 'add', line: newLines[ni] });
      ni++;
    } else {
      result.push({ type: 'remove', line: oldLines[oi] });
      oi++;
    }
  }
  return result;
}

// --- Resize ---
function _wireResize() {
  const handle = _el.resize;
  if (!handle) return;
  let startX, startW;
  handle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startX = e.clientX;
    startW = _el.editor?.offsetWidth || 400;
    handle.classList.add('dragging');
    const onMove = (e2) => {
      const delta = e2.clientX - startX;
      if (_el.editor) _el.editor.style.width = Math.max(200, startW + delta) + 'px';
    };
    const onUp = () => {
      handle.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

// --- Console ---
function _syncConsoleState() {
  const visible = !!_el.console
    && !_el.console.classList.contains('hidden')
    && !_el.console.classList.contains('collapsed');
  _el.workspace?.classList.toggle('console-open', visible);
}

function _clearConsoleEntries() {
  _consoleEntries = [];
  if (_el.consoleLog) _el.consoleLog.innerHTML = '';
  if (_el.consoleCount) _el.consoleCount.textContent = '';
  if (_mode === 'play') _el.console?.classList.add('hidden');
  _syncConsoleState();
}

function _consoleArgToString(arg) {
  if (arg === undefined) return 'undefined';
  if (arg === null) return 'null';
  if (typeof arg === 'string') return arg;
  try {
    return typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg);
  } catch {
    return String(arg);
  }
}

function _normalizeConsoleMessage(data = {}) {
  const isErrorEvent = data.type === 'code-error';
  const detail = data.detail || data.message || '';
  const lineMatch = String(detail).match(/\(line\s+(\d+)\)|line\s+(\d+)/i);
  const line = data.line || (lineMatch ? Number(lineMatch[1] || lineMatch[2]) : 0);
  const message = data.message
    || data.detail
    || (Array.isArray(data.args) ? data.args.map(_consoleArgToString).join(' ') : '')
    || 'Unknown preview message';
  return {
    ...data,
    level: isErrorEvent ? 'error' : (data.level || 'log'),
    message,
    line,
  };
}

function _addConsoleEntry(data) {
  if (_consoleEntries.length >= 100) _consoleEntries.shift();
  // Source map translation
  let fileLine = '';
  if (data.line && _sourceMap.length > 0) {
    for (let i = _sourceMap.length - 1; i >= 0; i--) {
      const sm = _sourceMap[i];
      if (data.line >= sm.assembledLineStart && data.line < sm.assembledLineStart + sm.lineCount) {
        const origLine = data.line - sm.assembledLineStart + sm.fileLineStart;
        fileLine = `${sm.file}:${origLine}`;
        break;
      }
    }
  }
  const entry = { level: data.level || 'log', message: data.message || '', fileLine };
  _consoleEntries.push(entry);
  _renderConsoleEntry(entry);
  // Update count
  const errorCount = _consoleEntries.filter(e => e.level === 'error').length;
  if (_el.consoleCount) _el.consoleCount.textContent = errorCount > 0 ? `${errorCount}` : '';
  // Auto-open on warnings/errors, including Play mode where the console is hidden by default.
  if (entry.level === 'error' || entry.level === 'warn') {
    _el.console?.classList.remove('hidden', 'collapsed');
  }
  _syncConsoleState();
}

function _renderConsoleEntry(entry) {
  if (!_el.consoleLog) return;
  const div = document.createElement('div');
  div.className = `workspace-console-entry ${entry.level}`;
  const fileSpan = entry.fileLine ? `<span class="console-file" data-file="${escapeHtml(entry.fileLine)}">${escapeHtml(entry.fileLine)}</span> — ` : '';
  const actions = entry.level === 'error'
    ? `<span class="workspace-console-actions">
        <button type="button" class="workspace-console-action" data-console-action="explain">Explain</button>
        <button type="button" class="workspace-console-action" data-console-action="fix">Fix with me</button>
      </span>`
    : '';
  div.innerHTML = `${fileSpan}${escapeHtml(entry.message)}${actions}`;
  div.querySelector('.console-file')?.addEventListener('click', (e) => {
    const [path, line] = e.target.dataset.file.split(':');
    const idx = _files.findIndex(f => f.path === path);
    if (idx >= 0) {
      _setMode('work');
      _activeFile = idx;
      _loadEditor();
      // Scroll to line (approximate)
      if (_cmEditorId && line) {
        CMEditor.setCursor(_cmEditorId, parseInt(line, 10), 1);
      }
    }
  });
  div.querySelector('[data-console-action="explain"]')?.addEventListener('click', (e) => {
    e.stopPropagation();
    _showErrorCoach(entry);
  });
  div.querySelector('[data-console-action="fix"]')?.addEventListener('click', (e) => {
    e.stopPropagation();
    _startGuidedErrorFix(entry);
  });
  _el.consoleLog.appendChild(div);
  _el.consoleLog.scrollTop = _el.consoleLog.scrollHeight;
}

// --- AI Prompt ---
let _buildPoll = null;
// Conversation history for iterative editing
let _promptHistory = [];

function _addPromptHistoryEntry(description, status = 'pending') {
  _promptHistory.push({ text: description, status, timestamp: Date.now() });
  _renderPromptHistory();
}

function _updateLastHistoryStatus(status) {
  if (_promptHistory.length > 0) {
    _promptHistory[_promptHistory.length - 1].status = status;
    _renderPromptHistory();
  }
}

function _renderPromptHistory() {
  const historyEl = document.getElementById('workspace-prompt-history');
  if (!historyEl) return;

  if (_promptHistory.length === 0) {
    historyEl.innerHTML = '';
    historyEl.style.display = 'none';
    return;
  }

  // Show last 5 entries
  const entries = _promptHistory.slice(-5);
  historyEl.style.display = '';
  historyEl.innerHTML = entries.map(e => {
    const icon = e.status === 'done' ? '\u2705' : e.status === 'error' ? '\u274C' : '\u23F3';
    return `<div class="prompt-history-item ${e.status}">${icon} ${escapeHtml(e.text.slice(0, 80))}${e.text.length > 80 ? '\u2026' : ''}</div>`;
  }).join('');

  // Auto-scroll to bottom
  historyEl.scrollTop = historyEl.scrollHeight;
}

const _ASSIST_LABELS = {
  do: 'Do it',
  coach: 'Coach me',
  challenge: 'Challenge me',
};

function _getAssistMode() {
  return _el.learningStyle?.value || 'do';
}

function _setAssistMode(mode) {
  if (_el.learningStyle && _ASSIST_LABELS[mode]) _el.learningStyle.value = mode;
}

function _assistHistoryText(description, assistMode) {
  if (assistMode === 'do') return description;
  return `${_ASSIST_LABELS[assistMode] || 'Coach me'}: ${description}`;
}

function _setPromptDraft(text) {
  if (!_el.promptInput) return;
  _el.promptInput.disabled = false;
  _el.promptInput.classList.remove('loading');
  _el.promptInput.value = text;
  _el.promptInput.focus();
}

function _detectLearningConcepts(text, file = '') {
  const haystack = `${file}\n${text}`.toLowerCase();
  const concepts = [];
  const add = (label, re) => { if (re.test(haystack) && !concepts.includes(label)) concepts.push(label); };

  add('HTML structure', /\.html|<\s*(main|section|button|input|form|div|span|canvas|img)\b/);
  add('CSS layout', /\.css|display\s*:|grid|flex|align-items|justify-content|position\s*:|gap\s*:|margin|padding/);
  add('Styling', /color\s*:|background|border|box-shadow|font-|transition|animation|transform/);
  add('DOM events', /addeventlistener|onclick|oninput|onsubmit|event\.|preventdefault|keydown|keyup|pointer|click/);
  add('State', /\bstate\b|setstate|usestate|let\s+\w+\s*=|const\s+\w+\s*=|localstorage|sessionstorage|dataset/);
  add('Functions', /function\s+\w+|=>|return\s+|class\s+\w+|method/);
  add('Arrays and lists', /\.map\s*\(|\.filter\s*\(|\.reduce\s*\(|foreach\s*\(|push\s*\(|array|list|items/);
  add('Async and APIs', /async\s+|await\s+|fetch\s*\(|promise|then\s*\(|catch\s*\(|api|json\s*\(/);
  add('Errors', /typeerror|referenceerror|syntaxerror|undefined|null|not a function|cannot read|unexpected token/);

  return concepts.slice(0, 4);
}

function _primaryConcept(concepts) {
  return concepts[0] || 'project structure';
}

function _coachTryText(concepts, assistMode, source = 'diff') {
  const concept = _primaryConcept(concepts);
  if (assistMode === 'challenge') {
    return source === 'error'
      ? `Challenge: before asking for a fix, point to the word in the error that tells you what kind of problem it is.`
      : `Challenge: before accepting, predict what user-visible behavior this ${concept} change should create.`;
  }
  if (source === 'error') {
    return 'Try it: read the first line of the error first, then jump to the file link. Most browser errors name the broken assumption right there.';
  }
  return `Try it: run the preview and test the part touched by this ${concept} change before accepting it.`;
}

function _dismissCoachTray() {
  _el.workspace?.querySelector('.workspace-coach-tray')?.remove();
}

function _showCoachTray({ title, copy, concepts = [], tryIt = '', actionText = 'Ask follow-up', actionPrompt = '' }) {
  if (!_el.body) return;
  _dismissCoachTray();
  const conceptHtml = concepts.length
    ? `<div class="workspace-coach-concepts">${concepts.map(c => `<span class="workspace-coach-chip">${escapeHtml(c)}</span>`).join('')}</div>`
    : '';
  const tryHtml = tryIt ? `<p class="workspace-coach-try">${escapeHtml(tryIt)}</p>` : '';
  const actionHtml = actionPrompt
    ? `<button type="button" class="workspace-coach-action primary" data-coach-action="prompt">${escapeHtml(actionText)}</button>`
    : '';
  const tray = document.createElement('div');
  tray.className = 'workspace-coach-tray';
  tray.innerHTML = `
    <div class="workspace-coach-head">
      <span class="workspace-coach-kicker">Project learning</span>
      <button type="button" class="workspace-coach-close" aria-label="Close">&times;</button>
    </div>
    <h3 class="workspace-coach-title">${escapeHtml(title)}</h3>
    <p class="workspace-coach-copy">${escapeHtml(copy)}</p>
    ${conceptHtml}
    ${tryHtml}
    <div class="workspace-coach-actions">${actionHtml}</div>
  `;
  tray.querySelector('.workspace-coach-close')?.addEventListener('click', _dismissCoachTray);
  tray.querySelector('[data-coach-action="prompt"]')?.addEventListener('click', () => {
    _setAssistMode('coach');
    _setPromptDraft(actionPrompt);
  });
  _el.body.appendChild(tray);
}

function _hunkLearningText(hunk) {
  const changedText = `${hunk.search || ''}\n${hunk.replace || ''}`;
  const concepts = _detectLearningConcepts(changedText, hunk.file);
  const added = hunk.diffLines.filter(l => l.type === 'add').length;
  const removed = hunk.diffLines.filter(l => l.type === 'remove').length;
  const lineText = added || removed
    ? `${added} added line${added === 1 ? '' : 's'} and ${removed} removed line${removed === 1 ? '' : 's'}`
    : 'a small local change';
  return {
    concepts,
    copy: `This hunk changes ${hunk.file} around line ${hunk.matchStart > 0 ? hunk.matchStart : 'the matched block'} with ${lineText}. The main idea looks like ${_primaryConcept(concepts)}.`,
  };
}

function _showHunkCoach(hunk) {
  if (!hunk) return;
  const { concepts, copy } = _hunkLearningText(hunk);
  _showCoachTray({
    title: `What changed in ${hunk.file}`,
    copy,
    concepts,
    tryIt: _coachTryText(concepts, _getAssistMode()),
    actionPrompt: `Explain this ${_primaryConcept(concepts)} change in ${hunk.file} and give me one tiny exercise.`,
  });
}

function _showDiffCoach() {
  const hunks = _pendingHunks.length ? _pendingHunks : [];
  if (hunks.length > 0) {
    const text = hunks.map(h => `${h.file}\n${h.search || ''}\n${h.replace || ''}`).join('\n');
    const concepts = _detectLearningConcepts(text, hunks[0]?.file || '');
    const files = Array.from(new Set(hunks.map(h => h.file).filter(Boolean)));
    _showCoachTray({
      title: hunks.length === 1 ? `Learn this change` : `Learn these ${hunks.length} changes`,
      copy: `These edits touch ${files.length} file${files.length === 1 ? '' : 's'} and mainly look like ${_primaryConcept(concepts)}. Use the blue added lines to see what new behavior Augmentum is introducing.`,
      concepts,
      tryIt: _coachTryText(concepts, _getAssistMode()),
      actionPrompt: `Walk me through these project changes and ask me one quick check question.`,
    });
    return;
  }

  const changedFiles = _patchedFiles.length ? _patchedFiles : _files;
  const text = changedFiles.map(f => `${f.path}\n${f.content || ''}`).join('\n');
  const concepts = _detectLearningConcepts(text, changedFiles[0]?.path || '');
  _showCoachTray({
    title: 'Learn this edit',
    copy: `This edit updated ${changedFiles.length} file${changedFiles.length === 1 ? '' : 's'} and looks mainly like ${_primaryConcept(concepts)}. Check the preview first, then ask for a deeper walkthrough if the result surprises you.`,
    concepts,
    tryIt: _coachTryText(concepts, _getAssistMode()),
    actionPrompt: `Explain the recent project edit and turn it into one 30-second practice step.`,
  });
}

function _showErrorCoach(entry) {
  const text = `${entry.message || ''}\n${entry.fileLine || ''}`;
  const concepts = _detectLearningConcepts(text, entry.fileLine || '');
  const where = entry.fileLine ? ` at ${entry.fileLine}` : '';
  _showCoachTray({
    title: 'Console error',
    copy: `The preview reported "${entry.message || 'an error'}"${where}. This looks like ${_primaryConcept(concepts)}. Start with the error name, then inspect the file link if one is available.`,
    concepts,
    tryIt: _coachTryText(concepts, _getAssistMode(), 'error'),
    actionText: 'Ask why',
    actionPrompt: `Explain this console error in my project: ${entry.message || 'unknown error'}${where}.`,
  });
}

function _startGuidedErrorFix(entry) {
  const where = entry.fileLine ? ` at ${entry.fileLine}` : '';
  _setAssistMode('coach');
  _setMode('work');
  _setPromptDraft(`Fix this error with me: ${entry.message || 'unknown error'}${where}`);
}

function _setPromptBusy(label) {
  if (_el.promptSend) {
    _el.promptSend.textContent = 'Cancel';
    _el.promptSend.classList.add('loading');
  }
  if (_el.promptInput) {
    _el.promptInput.placeholder = label;
    _el.promptInput.disabled = true;
    _el.promptInput.classList.add('loading');
  }
}

function _stopBuildStatusFeed() {
  if (!_buildPoll) return;
  if (typeof _buildPoll.close === 'function') _buildPoll.close();
  else clearInterval(_buildPoll);
  _buildPoll = null;
}

function _generationPanel() {
  return _el.workspace?.querySelector('.workspace-generation-panel') || null;
}

function _dismissGenerationPanel() {
  _stopGenerationTimer();
  _generationLastStage = '';
  _generationPanel()?.remove();
}

function _formatGenerationElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = String(totalSeconds % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function _tickGenerationTimer() {
  const panel = _generationPanel();
  const elapsedEl = panel?.querySelector('.workspace-gen-elapsed');
  if (!elapsedEl || !_generationStartedAt) return;
  elapsedEl.textContent = _formatGenerationElapsed(Date.now() - _generationStartedAt);
}

function _startGenerationTimer() {
  _stopGenerationTimer();
  _generationStartedAt = Date.now();
  _tickGenerationTimer();
  _generationTimer = setInterval(_tickGenerationTimer, 1000);
}

function _stopGenerationTimer() {
  if (_generationTimer) clearInterval(_generationTimer);
  _generationTimer = null;
  _tickGenerationTimer();
}

function _setGenerationMinimized(minimized) {
  const panel = _generationPanel();
  if (!panel) return;
  panel.classList.toggle('minimized', !!minimized);
  panel.dataset.minimized = minimized ? 'true' : 'false';
  const btn = panel.querySelector('.workspace-gen-minimize');
  if (btn) {
    btn.innerHTML = minimized ? '+' : '&minus;';
    btn.setAttribute('aria-label', minimized ? 'Expand generation panel' : 'Minimize generation panel');
    btn.setAttribute('aria-expanded', minimized ? 'false' : 'true');
  }
}

function _closeGenerationPanel() {
  const panel = _generationPanel();
  if (!panel) return;
  if (_activeOperation || panel.dataset.status === 'running') {
    _setGenerationMinimized(true);
    return;
  }
  _dismissGenerationPanel();
}

function _setGenerationProgress(progress) {
  const panel = _generationPanel();
  const track = panel?.querySelector('.workspace-gen-progress');
  const bar = panel?.querySelector('.workspace-gen-progress-bar');
  const text = panel?.querySelector('.workspace-gen-progress-text');
  if (!track || !bar || !text) return;

  if (progress === null || progress === undefined || Number.isNaN(Number(progress))) {
    track.dataset.mode = 'indeterminate';
    bar.style.width = '42%';
    text.textContent = 'Working';
    return;
  }

  const pct = Math.max(0, Math.min(100, Math.round(Number(progress))));
  track.dataset.mode = 'determinate';
  bar.style.width = `${pct}%`;
  text.textContent = `${pct}%`;
}

function _safeGenClass(value) {
  return String(value || 'pending').toLowerCase().replace(/[^a-z0-9_-]/g, '-');
}

function _buildQualityStatus(data = {}) {
  const project = data.project || {};
  const value = data.qualityStatus || data.quality_status || project.qualityStatus || project.quality_status || 'clean';
  return value && value !== 'clean' ? value : '';
}

function _buildQualityMessages(data = {}) {
  const project = data.project || {};
  const warnings = data.warnings || data.qualityWarnings || project.warnings || project.qualityWarnings || [];
  const errors = data.blockingErrors || data.blocking_errors || project.blockingErrors || project.blocking_errors || [];
  return [...(Array.isArray(warnings) ? warnings : []), ...(Array.isArray(errors) ? errors : [])];
}

function _buildNeedsReview(data = {}) {
  return Boolean(_buildQualityStatus(data) || _buildQualityMessages(data).length);
}

function _computeGenerationSummary(update = {}) {
  if (update.summary) return update.summary;
  const done = Array.isArray(update.filesComplete) ? update.filesComplete.length : 0;
  const remaining = Array.isArray(update.filesRemaining) ? update.filesRemaining.length : 0;
  if (done || remaining) return `Live preview updates as files finish (${done}/${done + remaining}).`;
  if (Array.isArray(update.contextFiles)) return `Sending ${update.contextFiles.length} file${update.contextFiles.length === 1 ? '' : 's'} as edit context.`;
  if (Array.isArray(update.patches)) return `Showing ${update.patches.length} patch candidate${update.patches.length === 1 ? '' : 's'} with match status.`;
  return '';
}

function _showGenerationPanel({ kind, title }) {
  if (!_el.body) return null;
  let panel = _generationPanel();
  if (!panel) {
    panel = document.createElement('div');
    panel.className = 'workspace-generation-panel';
    panel.innerHTML = `
      <div class="workspace-gen-head">
        <div class="workspace-gen-heading">
          <div class="workspace-gen-kicker">Generation</div>
          <div class="workspace-gen-title"></div>
        </div>
        <span class="workspace-gen-status">Starting</span>
        <button type="button" class="workspace-gen-minimize" aria-label="Minimize generation panel" aria-expanded="true">&minus;</button>
        <button type="button" class="workspace-gen-close" aria-label="Dismiss">&times;</button>
      </div>
      <div class="workspace-gen-progress" data-mode="indeterminate">
        <span class="workspace-gen-progress-bar"></span>
      </div>
      <div class="workspace-gen-meta">
        <span class="workspace-gen-elapsed">0s</span>
        <span class="workspace-gen-progress-text">Working</span>
      </div>
      <div class="workspace-gen-body">
        <div class="workspace-gen-stage"></div>
        <div class="workspace-gen-summary"></div>
        <div class="workspace-gen-metrics"></div>
        <div class="workspace-gen-context"></div>
        <div class="workspace-gen-passes"></div>
        <div class="workspace-gen-files"></div>
        <div class="workspace-gen-patches"></div>
        <div class="workspace-gen-log"></div>
      </div>
    `;
    panel.querySelector('.workspace-gen-minimize')?.addEventListener('click', (e) => {
      e.stopPropagation();
      _setGenerationMinimized(!panel.classList.contains('minimized'));
    });
    panel.querySelector('.workspace-gen-close')?.addEventListener('click', (e) => {
      e.stopPropagation();
      _closeGenerationPanel();
    });
    panel.addEventListener('click', (e) => {
      if (panel.classList.contains('minimized') && !e.target.closest('button')) {
        _setGenerationMinimized(false);
      }
    });
    _el.body.appendChild(panel);
  }
  _startGenerationTimer();
  _generationLastStage = '';
  _setGenerationMinimized(false);
  panel.dataset.kind = kind || '';
  panel.dataset.status = 'running';
  panel.querySelector('.workspace-gen-title').textContent = title || 'Project update';
  panel.querySelector('.workspace-gen-status').textContent = kind === 'quick' ? 'Quick edit' : 'Rebuild';
  panel.querySelector('.workspace-gen-stage').textContent = '';
  panel.querySelector('.workspace-gen-summary').textContent = kind === 'quick'
    ? 'Targeting the smallest patch that can satisfy your request.'
    : 'Streaming build checkpoints into the preview as files finish.';
  panel.querySelector('.workspace-gen-metrics').textContent = '';
  panel.querySelector('.workspace-gen-context').innerHTML = '';
  panel.querySelector('.workspace-gen-passes').innerHTML = '';
  panel.querySelector('.workspace-gen-files').innerHTML = '';
  panel.querySelector('.workspace-gen-patches').innerHTML = '';
  panel.querySelector('.workspace-gen-log').innerHTML = '';
  _setGenerationProgress(kind === 'quick' ? 12 : null);
  _appendGenerationLog(kind === 'quick' ? 'Preparing a targeted edit.' : 'Starting the full project pipeline.');
  return panel;
}

function _appendGenerationLog(text) {
  const panel = _generationPanel();
  const logEl = panel?.querySelector('.workspace-gen-log');
  if (!logEl || !text) return;
  const row = document.createElement('div');
  row.className = 'workspace-gen-log-row';
  row.textContent = text;
  logEl.appendChild(row);
  logEl.scrollTop = logEl.scrollHeight;
}

function _updateGenerationPanelLegacy(update = {}) {
  const panel = _generationPanel();
  if (!panel) return;
  if (update.status) {
    panel.dataset.status = update.status;
    if (['complete', 'error', 'cancelled', 'canceled'].includes(update.status)) {
      _stopGenerationTimer();
    }
  }
  if (update.label) {
    const statusEl = panel.querySelector('.workspace-gen-status');
    if (statusEl) statusEl.textContent = update.label;
  }
  if (update.stage) {
    const stageEl = panel.querySelector('.workspace-gen-stage');
    if (stageEl) stageEl.textContent = update.stage;
    if (update.stage !== _generationLastStage) {
      _appendGenerationLog(update.stage);
      _generationLastStage = update.stage;
    }
  }
  const summary = _computeGenerationSummary(update);
  if (summary) {
    const summaryEl = panel.querySelector('.workspace-gen-summary');
    if (summaryEl) summaryEl.textContent = summary;
  }
  if (update.metrics) {
    const metricsEl = panel.querySelector('.workspace-gen-metrics');
    const items = [];
    if (update.metrics.buildId) items.push(`build ${update.metrics.buildId}`);
    if (update.metrics.tokens) items.push(`${Number(update.metrics.tokens).toLocaleString()} tokens`);
    if (update.metrics.calls) items.push(`${update.metrics.calls} LLM call${update.metrics.calls === 1 ? '' : 's'}`);
    if (update.metrics.patches !== undefined) items.push(`${update.metrics.patches} patch${update.metrics.patches === 1 ? '' : 'es'}`);
    if (metricsEl) metricsEl.textContent = items.join(' · ');
  }
  if (update.contextFiles) {
    const contextEl = panel.querySelector('.workspace-gen-context');
    if (contextEl) {
      contextEl.innerHTML = `<div class="workspace-gen-section-label">Context sent</div>` +
        update.contextFiles.map(f => `<div class="workspace-gen-file-row ${f.mode}">
          <span>${escapeHtml(f.path)}</span><span>${escapeHtml(f.mode)} · ${f.lines}L</span>
        </div>`).join('');
    }
  }
  if (update.passes) _renderGenerationPasses(update.passes);
  if (update.filesComplete || update.filesRemaining) {
    _renderGenerationFiles(update.filesComplete || [], update.filesRemaining || []);
  }
  if (update.patches) {
    const patchesEl = panel.querySelector('.workspace-gen-patches');
    if (patchesEl) {
      patchesEl.innerHTML = `<div class="workspace-gen-section-label">Patches</div>` +
        update.patches.slice(0, 8).map(p => `<div class="workspace-gen-patch-row ${p.applied ? 'applied' : 'missed'}">
          <span>${escapeHtml(p.file || 'unknown')}</span>
          <span>${escapeHtml(p.match || 'matched')}${p.line ? ` · line ${p.line}` : ''}</span>
        </div>`).join('');
    }
  }
}

function _renderGenerationPatches(patches) {
  const panel = _generationPanel();
  const patchesEl = panel?.querySelector('.workspace-gen-patches');
  if (!patchesEl) return;
  const shown = patches.slice(0, 8);
  const rows = shown.map(p => {
    const state = p.applied ? 'applied' : 'missed';
    const match = p.match || (p.applied ? 'matched' : 'not matched');
    const line = p.line ? ` | line ${p.line}` : '';
    return `<div class="workspace-gen-patch-row ${state}">
      <span>${escapeHtml(p.file || 'unknown')}</span>
      <span>${escapeHtml(match)}${line}</span>
    </div>`;
  }).join('');
  const more = patches.length > shown.length
    ? `<div class="workspace-gen-patch-row pending"><span>${patches.length - shown.length} more patches</span><span>hidden</span></div>`
    : '';
  patchesEl.innerHTML = `<div class="workspace-gen-section-label">Patches (${patches.length})</div>${rows}${more}`;
}

function _updateGenerationPanel(update = {}) {
  const panel = _generationPanel();
  if (!panel) return;

  if (update.status) {
    panel.dataset.status = update.status;
    if (['complete', 'error', 'cancelled', 'canceled'].includes(update.status)) {
      _stopGenerationTimer();
    }
  }
  if (update.label) {
    const statusEl = panel.querySelector('.workspace-gen-status');
    if (statusEl) statusEl.textContent = update.label;
  }
  if (update.stage) {
    const stageEl = panel.querySelector('.workspace-gen-stage');
    if (stageEl) stageEl.textContent = update.stage;
    if (update.stage !== _generationLastStage) {
      _appendGenerationLog(update.stage);
      _generationLastStage = update.stage;
    }
  }

  const summary = _computeGenerationSummary(update);
  if (summary) {
    const summaryEl = panel.querySelector('.workspace-gen-summary');
    if (summaryEl) summaryEl.textContent = summary;
  }

  if (update.metrics) {
    const metricsEl = panel.querySelector('.workspace-gen-metrics');
    const items = [];
    if (update.metrics.buildId) items.push(`build ${update.metrics.buildId}`);
    if (update.metrics.tokens) items.push(`${Number(update.metrics.tokens).toLocaleString()} tokens`);
    if (update.metrics.calls) items.push(`${update.metrics.calls} LLM call${update.metrics.calls === 1 ? '' : 's'}`);
    if (update.metrics.patches !== undefined) items.push(`${update.metrics.patches} patch${update.metrics.patches === 1 ? '' : 'es'}`);
    if (metricsEl) metricsEl.textContent = items.join(' | ');
  }

  if (update.contextFiles) {
    const contextEl = panel.querySelector('.workspace-gen-context');
    if (contextEl) {
      const rows = update.contextFiles.slice(0, 6).map(f => `<div class="workspace-gen-file-row ${_safeGenClass(f.mode)}">
        <span>${escapeHtml(f.path)}</span><span>${escapeHtml(f.mode)} | ${f.lines}L</span>
      </div>`).join('');
      const more = update.contextFiles.length > 6
        ? `<div class="workspace-gen-file-row pending"><span>${update.contextFiles.length - 6} more files</span><span>included</span></div>`
        : '';
      contextEl.innerHTML = `<div class="workspace-gen-section-label">Context sent (${update.contextFiles.length})</div>${rows}${more}`;
    }
  }

  if (update.passes) _renderGenerationPasses(update.passes);
  if (update.filesComplete || update.filesRemaining) {
    _renderGenerationFiles(update.filesComplete || [], update.filesRemaining || []);
  }
  if (update.patches) _renderGenerationPatches(update.patches);
  if ('progress' in update) _setGenerationProgress(update.progress);
  if (update.log) _appendGenerationLog(update.log);
}

function _renderGenerationPasses(passes) {
  const panel = _generationPanel();
  const passesEl = panel?.querySelector('.workspace-gen-passes');
  if (!passesEl || !passes) return;
  passesEl.innerHTML = `<div class="workspace-gen-section-label">Pipeline</div>` +
    passes.map(p => {
      const detail = [p.detail || '', p.iterations ? `try ${p.iterations}` : ''].filter(Boolean).join(' | ');
      return `<div class="workspace-gen-pass ${_safeGenClass(p.status || 'pending')}">
      <span class="workspace-gen-pass-dot"></span>
      <span>${escapeHtml(p.name || 'step')}</span>
      <span>${escapeHtml(detail)}</span>
    </div>`;
    }).join('');
}

function _renderGenerationFiles(done, remaining) {
  const panel = _generationPanel();
  const filesEl = panel?.querySelector('.workspace-gen-files');
  if (!filesEl) return;
  const rows = [];
  for (const f of done) rows.push(`<div class="workspace-gen-file-row done"><span>${escapeHtml(f)}</span><span>ready</span></div>`);
  remaining.forEach((f, i) => {
    rows.push(`<div class="workspace-gen-file-row ${i === 0 ? 'active' : 'pending'}"><span>${escapeHtml(f)}</span><span>${i === 0 ? 'generating' : 'queued'}</span></div>`);
  });
  if (!rows.length) {
    filesEl.innerHTML = '';
    return;
  }
  filesEl.innerHTML = `<div class="workspace-gen-section-label">Files (${done.length}/${done.length + remaining.length})</div>${rows.join('')}`;
}

function _workspaceBuildProgress(data) {
  if (!data) return null;
  if (data.status === 'complete') return 100;
  if (data.status === 'error' || data.status === 'cancelled' || data.status === 'canceled') return 100;

  const done = Array.isArray(data.filesComplete) ? data.filesComplete.length : 0;
  const remaining = Array.isArray(data.filesRemaining) ? data.filesRemaining.length : 0;
  const totalFiles = done + remaining;
  const running = (data.passes || []).find(p => p.status === 'running');
  if (totalFiles > 0 && (!running || running.name === 'generate')) {
    return 18 + Math.round((done / Math.max(totalFiles, 1)) * 44);
  }

  const passFloor = {
    plan: 12,
    generate: 30,
    validate: 68,
    improve: 78,
    polish: 86,
    verify: 94,
    deliver: 98,
  };
  const completePasses = (data.passes || []).filter(p => p.status === 'complete');
  const lastComplete = completePasses[completePasses.length - 1]?.name;
  if (running?.name && passFloor[running.name] !== undefined) return passFloor[running.name];
  if (lastComplete && passFloor[lastComplete] !== undefined) return Math.min(96, passFloor[lastComplete] + 5);
  return null;
}

function _checkpointSignature(files) {
  return (files || []).map(f => `${f.path}:${String(f.content || '').length}`).join('|');
}

function _applyCheckpointPreview(files) {
  if (!Array.isArray(files) || !files.length) return;
  const sig = _checkpointSignature(files);
  if (!sig || sig === _lastCheckpointSignature) return;
  _lastCheckpointSignature = sig;
  _files = files.map(f => ({ ...f, content: f.content || '' }));
  _loadPreview();
  if (_mode === 'work') {
    if (_activeFile >= _files.length) _activeFile = 0;
    _renderFileTabs();
    if (!_isEditorFocused()) _loadEditor();
  }
}

function _startWorkspaceBuildFeed(buildId, assistMode) {
  _stopBuildStatusFeed();
  if (!buildId) return;
  let polling = false;

  const apply = (data) => _applyWorkspaceBuildStatus(data, assistMode);
  const startPolling = () => {
    if (polling) return;
    polling = true;
    _buildPoll = setInterval(async () => {
      try {
        const sr = await fetch(`/api/builds/${encodeURIComponent(buildId)}`);
        if (!sr.ok) {
          _appendGenerationLog(`Build status request failed: HTTP ${sr.status}`);
          return;
        }
        const body = await sr.json();
        apply(body.run || body);
      } catch (err) {
        _appendGenerationLog(`Waiting for build status: ${err.message}`);
      }
    }, 1500);
  };

  if (typeof EventSource === 'function') {
    try {
      const es = new EventSource(`/api/builds/${encodeURIComponent(buildId)}/stream`);
      _buildPoll = es;
      es.onmessage = (ev) => {
        try { apply(JSON.parse(ev.data)); } catch { /* ignore malformed progress */ }
      };
      es.addEventListener('end', () => _stopBuildStatusFeed());
      es.onerror = () => {
        if (_buildPoll === es) {
          es.close();
          _buildPoll = null;
          startPolling();
        }
      };
      return;
    } catch {
      // Fall through to polling.
    }
  }
  startPolling();
}

// Consumes the coder-builder snapshot (build_status_snapshot): status, active,
// currentFile, steps[], behaviors[], artifact_id, workspace_id, totalTokens,
// llmCalls, qualityStatus, error. On completion the rebuilt files come from the
// newly published artifact (not inline), so we fetch + reload them.
function _applyWorkspaceBuildStatus(data, assistMode) {
  if (!data || !data.status) return;
  if (data.id || data.build_id) _activeBuildId = data.id || data.build_id || _activeBuildId;
  const status = (data.status || '').toLowerCase();
  const active = data.active || status === 'running';
  const quality = (data.qualityStatus || data.quality_status || '').toLowerCase();
  const needsReview = status === 'complete' && quality && quality !== 'clean';
  const tokens = data.totalTokens ? ` · ${Number(data.totalTokens).toLocaleString()} tok` : '';
  const behaviors = Array.isArray(data.behaviors) ? data.behaviors : [];
  const passed = behaviors.filter(b => (b.status || '').toLowerCase() === 'pass').length;
  const behaviorNote = behaviors.length ? ` · ${passed}/${behaviors.length} checks` : '';
  const statusLabel = active ? ('Building' + tokens) : (needsReview ? 'Needs review' : status);
  const stageLabel = data.currentFile || (active ? 'Building in the workspace' : status);

  _updateGenerationPanel({
    status: active ? 'running' : status,
    label: statusLabel,
    stage: stageLabel + (active ? behaviorNote : ''),
    summary: active
      ? 'The build runs in a real coder workspace and browser-tests as it goes.'
      : needsReview
        ? 'Build finished, but the behavior gate flagged something to review.'
        : 'Build finished. Review the rebuilt project before accepting.',
    metrics: {
      buildId: data.id || data.build_id || _activeBuildId,
      tokens: data.totalTokens || 0,
      calls: data.llmCalls || 0,
    },
    progress: active ? null : 100,
  });
  if (_el.promptInput && active) {
    _el.promptInput.placeholder = statusLabel;
  }

  if (status === 'paused') {
    // Checkpoint, not a failure — load what it built and offer "keep going".
    _stopBuildStatusFeed();
    _showRebuildCheckpointGate(data, assistMode);
    return;
  }

  if (status === 'complete') {
    _stopBuildStatusFeed();
    _loadRebuiltArtifact(data, assistMode, needsReview);
    return;
  }

  if (status === 'error' || status === 'cancelled' || status === 'canceled') {
    _stopBuildStatusFeed();
    // Even a stopped/failed build usually produced real changes in the
    // workspace (it published an artifact). NEVER silently discard them — load
    // what it built (flagged for review) instead of restoring the old snapshot.
    // Only restore when nothing at all was produced.
    const artifactId = data.artifact_id || data.artifactId || '';
    if (artifactId) {
      _loadRebuiltArtifact(data, assistMode, true, true);
      return;
    }
    if (_snapshot) {
      _files = _snapshot.map(f => ({ ...f, content: f.content }));
      _loadPreview();
      if (_mode === 'work') _loadEditor();
    }
    _updateGenerationPanel({
      status,
      label: status === 'error' ? 'Failed' : 'Cancelled',
      stage: data.error || (status === 'error' ? 'Build failed.' : 'Build cancelled.'),
      summary: 'Your previous project has been restored.',
      progress: 100,
    });
    _updateLastHistoryStatus('error');
    _resetPromptBar();
    if (typeof window.showToast === 'function') {
      window.showToast(data.error ? `Rebuild failed: ${data.error}` : 'Rebuild cancelled', data.error ? 'error' : 'info', 5000);
    }
  }
}

// A budget/stuck checkpoint: never a forced failure. Load what the build
// produced (so the change is kept + reviewable), then offer "Keep going"
// (resume on the same workspace) right in the generation panel.
async function _showRebuildCheckpointGate(data, assistMode) {
  await _loadRebuiltArtifact(data, assistMode, true, true);
  const panel = _generationPanel();
  const body = panel?.querySelector('.workspace-gen-body');
  if (!body) return;
  let gate = body.querySelector('.workspace-gen-gate');
  if (!gate) {
    gate = document.createElement('div');
    gate.className = 'workspace-gen-gate';
    gate.style.cssText = 'display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap;';
    body.appendChild(gate);
  }
  const reason = data.stop_reason ? ` (${data.stop_reason})` : '';
  gate.innerHTML = `
    <span style="flex:1;min-width:180px;font-size:12px;opacity:.85;">Reached a checkpoint${escapeHtml(reason)} — the change is loaded. Keep the build going for more, or accept what's here.</span>
    <button type="button" class="btn btn-sm btn-primary" data-gen-continue>Keep going</button>`;
  gate.querySelector('[data-gen-continue]')?.addEventListener('click', async (e) => {
    e.stopPropagation();
    gate.remove();
    await _continueWorkspaceBuild(data.id || data.build_id || _activeBuildId, assistMode);
  });
}

async function _continueWorkspaceBuild(buildId, assistMode) {
  if (!buildId) return;
  try {
    const r = await fetch(`/api/builds/${encodeURIComponent(buildId)}/resume`, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    _activeBuildId = buildId;
    _updateGenerationPanel({
      status: 'running', label: 'Continuing', stage: 'Resuming the build…',
      summary: 'Picking up where it left off in the same workspace.', progress: null,
    });
    _startWorkspaceBuildFeed(buildId, assistMode);
    if (typeof window.showToast === 'function') window.showToast('Continuing the build…', 'info');
  } catch (err) {
    if (typeof window.showToast === 'function') {
      window.showToast(`Couldn't continue: ${err.message || err}`, 'error');
    }
  }
}

// Coder rebuilds publish a NEW artifact instead of returning files inline —
// load its files into the play/work surface so the diff/accept flow still
// works, and re-point the workspace at the latest artifact for future edits.
async function _loadRebuiltArtifact(data, assistMode, needsReview, partial = false) {
  const artifactId = data.artifact_id || data.artifactId || '';
  let files = [];
  if (artifactId) {
    try {
      const r = await fetch(`/api/artifacts/${encodeURIComponent(artifactId)}`);
      if (r.ok) {
        const full = await r.json();
        const source = typeof full.source_json === 'string'
          ? JSON.parse(full.source_json) : (full.source_json || {});
        files = source?.files || [];
        if (_artifact) { _artifact.id = artifactId; _artifact.source_json = full.source_json; }
      }
    } catch { /* keep current files if the fetch fails */ }
  }

  if (!files.length) {
    // Couldn't load the rebuilt files — point the user at the live workspace.
    _updateGenerationPanel({
      status: 'complete',
      label: needsReview ? 'Needs review' : 'Complete',
      stage: 'Rebuilt in the workspace.',
      summary: 'Open it in Code to see the running preview and continue.',
      metrics: { buildId: _activeBuildId, workspaceId: data.workspace_id || '' },
      progress: 100,
    });
    _resetPromptBar();
    if (typeof window.showToast === 'function') {
      window.showToast('Rebuild complete — open it in Code to continue', 'success');
    }
    return;
  }

  _files = files;
  _patchedFiles = _files.map(f => ({ ...f, content: f.content || '' }));
  _pendingHunks = [];
  _rawResponse = '';
  _focusedHunkIdx = -1;
  _lastCheckpointSignature = _checkpointSignature(_files);
  _loadPreview();
  if (_mode === 'work') _loadEditor();
  _el.acceptBar?.classList.remove('hidden');
  const acceptText = _el.acceptBar?.querySelector('.workspace-accept-text');
  if (acceptText) acceptText.textContent = `Rebuild complete · ${_files.length} file${_files.length === 1 ? '' : 's'}`;
  _updateAcceptBarCounter();
  _updateLastHistoryStatus('done');
  _updateGenerationPanel({
    status: partial ? 'error' : 'complete',
    label: partial ? 'Stopped early' : (needsReview ? 'Needs review' : 'Complete'),
    stage: partial
      ? `The build stopped before finishing, but its changes are loaded (${_files.length} file${_files.length === 1 ? '' : 's'}).`
      : needsReview
        ? `Verification flagged review for ${_files.length} file${_files.length === 1 ? '' : 's'}.`
        : `Ready to review ${_files.length} file${_files.length === 1 ? '' : 's'}.`,
    summary: partial
      ? 'It didn’t reach a clean finish — review the change, then accept it or continue in Code.'
      : needsReview
        ? 'The rebuilt files loaded, but the behavior gate found something to review before accepting.'
        : 'The rebuilt files are loaded in the preview. Accept to keep them or reject to return to the previous project.',
    metrics: { buildId: _activeBuildId, tokens: data.totalTokens || 0, calls: data.llmCalls || 0 },
    progress: 100,
  });
  if (assistMode !== 'do') setTimeout(() => _showDiffCoach(), 0);
  _resetPromptBar();
  if (typeof window.showToast === 'function') {
    window.showToast(
      partial ? 'Build stopped early — its changes are loaded for review'
        : needsReview ? 'Rebuild complete — review recommended'
          : `Rebuild complete — ${_files.length} file${_files.length === 1 ? '' : 's'}`,
      partial || needsReview ? 'warning' : 'success',
    );
  }
}

async function _cancelActiveOperation() {
  if (!_activeOperation) return;
  const op = _activeOperation;
  if (_activePromptAbort) {
    try { _activePromptAbort.abort(); } catch {}
  }
  if (op === 'rebuild' && _activeBuildId) {
    try {
      await fetch(`/api/builds/${encodeURIComponent(_activeBuildId)}/cancel`, { method: 'POST' });
    } catch { /* best effort */ }
  }
  _stopBuildStatusFeed();
  if (_snapshot) {
    _files = _snapshot.map(f => ({ ...f, content: f.content }));
    _loadPreview();
    if (_mode === 'work') _loadEditor();
  }
  _updateGenerationPanel({
    status: 'cancelled',
    label: 'Cancelled',
    stage: 'Stopped at your request.',
    summary: 'No generated changes were kept.',
    progress: 100,
  });
  _updateLastHistoryStatus('error');
  _resetPromptBar();
  if (typeof window.showToast === 'function') window.showToast(`${op === 'quick' ? 'Edit' : 'Rebuild'} cancelled`, 'info');
}

async function _onPromptSend() {
  if (_activeOperation) {
    await _cancelActiveOperation();
    return;
  }
  const input = _el.promptInput;
  if (!input?.value.trim() || !_artifact) return;
  const description = input.value.trim();
  input.value = '';
  const assistMode = _getAssistMode();

  // Add to conversation history
  _addPromptHistoryEntry(_assistHistoryText(description, assistMode), 'pending');

  const modeSelect = document.getElementById('workspace-prompt-mode');
  const editMode = modeSelect?.value || 'quick';
  _activeOperation = editMode === 'quick' ? 'quick' : 'rebuild';
  _activeBuildId = '';
  _lastCheckpointSignature = '';
  _activePromptAbort = new AbortController();

  // Snapshot for revert
  _snapshot = _files.map(f => ({ ...f, content: f.content }));

  const loadingText = assistMode === 'coach'
    ? 'Coaching...'
    : assistMode === 'challenge'
      ? 'Preparing challenge...'
      : (editMode === 'quick' ? 'Editing...' : 'Building...');
  _setPromptBusy(loadingText);
  _showGenerationPanel({
    kind: _activeOperation,
    title: editMode === 'quick' ? description : `Rebuild: ${description}`,
  });

  if (editMode === 'quick') {
    // Quick mode — single LLM call via /fix endpoint (fast, no pipeline).
    // Send canonical source; the backend may compress prompt context, but the
    // patch applier must always operate against real user files.
    const filesToSend = _files.map(f => ({ ...f, content: f.content || '' }));
    _updateGenerationPanel({
      label: 'Asking model',
      stage: 'Sending targeted context for a SEARCH/REPLACE edit.',
      summary: 'Quick edit keeps the current project and asks for focused patch candidates.',
      progress: null,
      contextFiles: filesToSend.map(f => {
        const content = f.content || '';
        return {
          path: f.path,
          mode: 'full',
          lines: content ? content.split('\n').length : 0,
        };
      }),
    });
    try {
      const resp = await fetch('/api/artifacts/fix', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: _activePromptAbort?.signal,
        body: JSON.stringify({
          description,
          files: filesToSend,
          model: _getCurrentModel(),
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const result = await resp.json();
      _updateGenerationPanel({
        label: 'Applying patches',
        stage: `Model returned ${result.patch_details?.length ?? result.patches_applied ?? 0} patch candidate${(result.patch_details?.length ?? result.patches_applied ?? 0) === 1 ? '' : 's'}.`,
        summary: 'Checking each candidate against the exact source before showing review.',
        patches: result.patch_details || [],
        metrics: { patches: result.patches_applied || 0 },
        progress: 72,
      });
      if (result.success && result.files?.length > 0) {
        _patchedFiles = result.files;
        _rawResponse = result.raw_response || '';

        // Validate patched files with CodeMind AST (Repair Cascade gate)
        if (_codeMindReady) {
          for (const pf of _patchedFiles) {
            const lang = _getLangClass(pf.path);
            const cmLang = CodeMind.resolveLanguage(lang);
            if (cmLang) {
              const validation = await CodeMind.validate(pf.content || '', lang);
              if (!validation.valid && validation.errors.length > 0) {
                console.warn(`[CodeMind] LLM output has ${validation.errors.length} syntax error(s) in ${pf.path}:`,
                  validation.errors.slice(0, 3).map(e => `Ln ${e.startRow + 1}: ${e.message}`).join('; '));
              }
            }
          }
        }

        _updateLastHistoryStatus('done');

        // Parse hunks from raw response
        _pendingHunks = _parseRawResponseIntoHunks(_rawResponse, _snapshot);
        const appliedPatchCount = Number(result.patches_applied || 0);
        if (_pendingHunks.length > 0 && _pendingHunks.length !== appliedPatchCount) {
          console.warn('[workspace] Falling back to whole-preview review because parsed hunks do not match applied patches', {
            parsed: _pendingHunks.length,
            applied: appliedPatchCount,
          });
          _appendGenerationLog('Showing a whole-file preview because not every applied patch could be displayed as a hunk.');
          _pendingHunks = [];
        }
        console.debug(`[workspace] Parsed ${_pendingHunks.length} hunks from ${_rawResponse.length} chars. patches_applied=${result.patches_applied}`);
        if (_pendingHunks.length === 0 && _rawResponse) {
          console.warn('[workspace] Raw LLM response (no hunks parsed):', _rawResponse.slice(0, 1500));
        }
        if (_pendingHunks.length > 0) {
          // Keep _files at snapshot state — user decides what to accept
          _setMode('work');
          _renderFileTabs();
          _animateHunks(_files[_activeFile]?.path || _pendingHunks[0]?.file);
          _focusedHunkIdx = 0;
          _updateHunkFocus();
          if (typeof window.showToast === 'function') {
            window.showToast(`${_pendingHunks.length} change${_pendingHunks.length !== 1 ? 's' : ''} — review in work mode`, 'success');
          }
        } else {
          // No parseable hunks — apply all (backend already patched)
          _files = result.files;
          if (_mode === 'work') _loadEditor();
          if (typeof window.showToast === 'function') {
            window.showToast(`${result.patches_applied} change${result.patches_applied !== 1 ? 's' : ''} applied`, 'success');
          }
        }
        _loadPreview();
        if (_pendingHunks.length === 0) {
          // When hunks exist, _animateHunks shows the bar after animation
          _el.acceptBar?.classList.remove('hidden');
        }
        const acceptText = _el.acceptBar?.querySelector('.workspace-accept-text');
        if (acceptText) acceptText.textContent = _pendingHunks.length > 0
          ? `\u2728 ${_pendingHunks.length} change${_pendingHunks.length !== 1 ? 's' : ''}`
          : `\u2728 ${result.patches_applied} change${result.patches_applied !== 1 ? 's' : ''} applied`;
        _updateAcceptBarCounter();
        _updateGenerationPanel({
          status: 'complete',
          label: 'Ready to review',
          stage: _pendingHunks.length > 0
            ? `Review ${_pendingHunks.length} parsed change${_pendingHunks.length === 1 ? '' : 's'} in Work mode.`
            : `${result.patches_applied} change${result.patches_applied === 1 ? '' : 's'} applied and ready to accept.`,
          summary: _pendingHunks.length > 0
            ? 'Review each hunk before accepting the generated edit.'
            : 'The backend applied the edit as a whole-file preview. Accept to keep it.',
          patches: result.patch_details || [],
          metrics: { patches: result.patches_applied || 0 },
          progress: 100,
        });
        if (assistMode !== 'do') {
          setTimeout(() => _showDiffCoach(), _pendingHunks.length > 0 ? 350 : 0);
        }
      } else {
        _updateLastHistoryStatus('error');
        const reason = result.error || (result.patches_applied === 0
          ? 'LLM produced changes but none matched the source code'
          : 'No changes applied');
        if (typeof window.showToast === 'function') {
          window.showToast(reason, 'warning', 5000);
        }
        _updateGenerationPanel({
          status: 'error',
          label: 'No patches applied',
          stage: reason,
          summary: 'The model response did not safely match the current files, so nothing was kept.',
          patches: result.patch_details || [],
          metrics: { patches: result.patches_applied || 0 },
          progress: 100,
        });
        // Log raw response so the user can debug what the LLM actually returned
        if (result.raw_response) {
          console.warn('[workspace] LLM response (no patches matched):', result.raw_response.slice(0, 2000));
        }
        _resetPromptBar();
        _files = _snapshot;
        _snapshot = null;
      }
      _resetPromptBar();
    } catch (err) {
      if (err.name === 'AbortError') return;
      _updateLastHistoryStatus('error');
      if (typeof window.showToast === 'function') {
        window.showToast(`Edit failed: ${err.message}`, 'error', 5000);
      }
      _updateGenerationPanel({
        status: 'error',
        label: 'Failed',
        stage: err.message,
        summary: 'Your previous project has been restored.',
        progress: 100,
      });
      _resetPromptBar();
      _files = _snapshot;
      _snapshot = null;
    }
    return;
  }

  // Rebuild mode — the coder-workspace builder (run_build), seeded from this
  // artifact and browser-tested as it goes. Multi-step changes get proper
  // verification between steps, and the workspace persists for continuation —
  // unlike the retired quickjs pipeline. The surgical Edit path above still
  // handles quick diffs.
  try {
    const resp = await fetch('/api/builds/from-artifact', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: _activePromptAbort?.signal,
      body: JSON.stringify({
        artifact_id: _artifact.id,
        instructions: description,
        model: _getCurrentModel(),
        session_id: _artifact.session_id || '',
        name: _artifact.title || '',
      }),
    });
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try { detail = (await resp.json()).detail || detail; } catch { /* keep status */ }
      throw new Error(detail);
    }
    const started = await resp.json();
    if (!started.build_id) throw new Error('Build did not start');
    _activeBuildId = started.build_id;
    _updateGenerationPanel({
      label: 'Workspace build started',
      stage: 'Setting up the coder workspace…',
      summary: 'The rebuild runs in a real workspace and browser-tests as it goes.',
      metrics: { buildId: _activeBuildId },
      progress: null,
    });
    _appendGenerationLog(`Build id: ${_activeBuildId}`);
    _startWorkspaceBuildFeed(_activeBuildId, assistMode);
    return;
  } catch (err) {
    if (err.name === 'AbortError') return;
    _el.promptInput.placeholder = `Failed: ${err.message}`;
    _updateGenerationPanel({
      status: 'error',
      label: 'Failed',
      stage: err.message,
      summary: 'The rebuild did not start, so your current project is unchanged.',
      progress: 100,
    });
    _updateLastHistoryStatus('error');
    setTimeout(_resetPromptBar, 3000);
  }
}

function _resetPromptBar() {
  _activeOperation = null;
  _activePromptAbort = null;
  _activeBuildId = '';
  if (_el.promptInput) {
    _el.promptInput.disabled = false;
    _el.promptInput.placeholder = 'Ask AI to modify this project...';
    _el.promptInput.classList.remove('loading');
  }
  if (_el.promptSend) {
    _el.promptSend.textContent = 'Send';
    _el.promptSend.classList.remove('loading');
  }
  if (_el.promptSend) _el.promptSend.textContent = 'Send';
}

// --- Accept All / Reject All / Confirm ---
async function _onAcceptAll() {
  _pendingHunks.forEach(h => h.status = 'accepted');
  _el.workspace?.querySelectorAll('.diff-hunk').forEach(el => el.dataset.status = 'accepted');
  _files = _applyAcceptedHunks();
  await _finalizeDiff();
}

function _onRejectAll() {
  _pendingHunks.forEach(h => h.status = 'rejected');
  _el.workspace?.querySelectorAll('.diff-hunk').forEach(el => el.dataset.status = 'rejected');
  // Restore snapshot (discard all changes)
  if (_snapshot) _files = _snapshot.map(f => ({ ...f, content: f.content }));
  _clearDiffState();
  _loadPreview();
  if (typeof window.showToast === 'function') window.showToast('Changes dismissed', 'info');
}

/** Save accepted hunks, clear diff state, return to editor. */
async function _finalizeDiff() {
  if (!_artifact) return;
  try {
    await fetch(`/api/artifacts/${_artifact.id}/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_json: JSON.stringify({ files: _files }) }),
    });
    document.dispatchEvent(new CustomEvent('artifact:saved', { detail: { id: _artifact.id } }));
    const accepted = _pendingHunks.filter(h => h.status === 'accepted').length;
    const wholePreview = _pendingHunks.length === 0;
    _clearDiffState();
    _loadPreview();
    if (_mode === 'work') _loadFileIntoEditor(_activeFile);
    if (typeof window.showToast === 'function') {
      const msg = wholePreview
        ? `${_files.length} file${_files.length !== 1 ? 's' : ''} saved`
        : `${accepted} change${accepted !== 1 ? 's' : ''} applied`;
      window.showToast(msg, 'success');
    }
  } catch (err) {
    if (typeof window.showToast === 'function') window.showToast('Save failed: ' + err.message, 'error');
  }
}

/** Reset all diff/hunk state and hide UI. */
function _clearDiffState() {
  _snapshot = null;
  _patchedFiles = [];
  _pendingHunks = [];
  _rawResponse = '';
  _focusedHunkIdx = -1;
  _modified.clear();
  _el.acceptBar?.classList.add('hidden');
  _dismissCoachTray();
  _hideDiffView();
  _renderFileTabs();
}

// --- Save ---
async function _saveToArtifact() {
  if (!_artifact) return;
  try {
    await fetch(`/api/artifacts/${_artifact.id}/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_json: JSON.stringify({ files: _files }) }),
    });
    document.dispatchEvent(new CustomEvent('artifact:saved', { detail: { id: _artifact.id } }));
    _modified.clear();
    _renderFileTabs();
    if (typeof window.showToast === 'function') window.showToast('Saved', 'success');
  } catch (err) {
    if (typeof window.showToast === 'function') window.showToast('Save failed: ' + err.message, 'error');
  }
}
