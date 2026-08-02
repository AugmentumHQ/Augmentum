/**
 * command-palette.js — Universal launcher (Ctrl/Cmd+K).
 *
 * One overlay, three result kinds: workspace files, workspaces, and
 * actions registered by other modules. Designed so any module can
 * `registerCommand({...})` to surface its surface in the palette
 * without the palette needing to know about that module.
 *
 * Sources (resolved on every input change):
 *   - Files     /api/coder/workspaces/{ws}/files-flat (cached per ws)
 *   - Workspaces /api/coder/workspaces (cached short)
 *   - Actions   in-memory registry (synchronous)
 *
 * Filtering:
 *   - Empty query → recently used items first, then a sample of each kind
 *   - "> foo"     → actions matching "foo" only
 *   - "/" or path → files only
 *   - anything else → fuzzy across all three
 *
 * The palette is mode-agnostic — works outside coder mode too, just
 * the file source is empty when there's no active workspace. That's
 * the right behavior: actions like "open settings" or "switch
 * workspace" stay reachable.
 */

import { escapeHtml, showToast } from './app.js';

const RECENT_KEY = 'augmentum.cp.recent';
const RECENT_MAX = 20;
const FILES_CACHE_TTL_MS = 60_000;   // 1 min — agent can edit out from under us
const WORKSPACES_CACHE_TTL_MS = 30_000;

let _overlay = null;
let _input = null;
let _resultsEl = null;
let _commands = [];           // [{id, label, hint, group, run, when}]
let _filesCache = null;       // { workspaceId, ts, files }
let _workspacesCache = null;  // { ts, workspaces }
let _activeIndex = 0;
let _currentResults = [];
let _open = false;
let _onOpenHooks = [];
let _resolveActiveWorkspaceId = () => '';
let _resolveOpenFile = (_id, _path, _name) => {};
let _resolveSwitchWorkspace = (_id) => {};

/**
 * Public API
 */

export function initCommandPalette({
  getActiveWorkspaceId,
  openFile,
  switchWorkspace,
} = {}) {
  _overlay = document.getElementById('command-palette-overlay');
  _input = document.getElementById('command-palette-input');
  _resultsEl = document.getElementById('command-palette-results');
  if (!_overlay || !_input || !_resultsEl) return false;

  if (getActiveWorkspaceId) _resolveActiveWorkspaceId = getActiveWorkspaceId;
  if (openFile) _resolveOpenFile = openFile;
  if (switchWorkspace) _resolveSwitchWorkspace = switchWorkspace;

  _input.addEventListener('input', _onInput);
  _input.addEventListener('keydown', _onKeyDown);
  _overlay.addEventListener('click', (ev) => {
    // Click the backdrop (anything outside .cp-card) → close.
    if (ev.target === _overlay) close();
  });
  _resultsEl.addEventListener('click', (ev) => {
    const row = ev.target.closest('.cp-result');
    if (!row) return;
    const idx = Number(row.dataset.idx);
    if (Number.isFinite(idx)) _executeResultAt(idx);
  });
  _resultsEl.addEventListener('mousemove', (ev) => {
    const row = ev.target.closest('.cp-result');
    if (!row) return;
    const idx = Number(row.dataset.idx);
    if (Number.isFinite(idx) && idx !== _activeIndex) {
      _setActiveIndex(idx);
    }
  });

  // Global open shortcut: Ctrl/Cmd+K. Captured early so chat / editor
  // bindings don't intercept it. Ignored when the user is in a text
  // field that explicitly requests Ctrl+K for its own purpose — none
  // of ours do today, so the broad listener is safe.
  document.addEventListener('keydown', (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && (ev.key === 'k' || ev.key === 'K')) {
      ev.preventDefault();
      open();
    }
  });

  return true;
}

/**
 * Register a command. Idempotent on ``id`` — re-registering replaces.
 * Modules call this at init time; the palette picks up changes on the
 * next open. Returns an unregister function for hot-reload friendliness.
 */
export function registerCommand({
  id, label, hint = '', group = 'Action', run, when = null, keywords = '',
  agent = null,
}) {
  if (!id || typeof run !== 'function') return () => {};
  // `agent` opts the command into the companion's app menu (app.act).
  // Shape: { description, speak?, stakes? } — description is what the
  // matcher reads, speak is her authored ack, stakes defaults to
  // trivial_reversible (the only matchable class in v1). Registering
  // is the curation step: outcomes only, never UI plumbing.
  const cmd = { id, label, hint, group, run, when, keywords, agent };
  const existing = _commands.findIndex((c) => c.id === id);
  if (existing >= 0) _commands[existing] = cmd;
  else _commands.push(cmd);
  if (agent) _scheduleAgentSync();
  return () => {
    _commands = _commands.filter((c) => c.id !== id);
    if (agent) _scheduleAgentSync();
  };
}

/**
 * Run a registered command by id — the receiving end of the
 * companion's `palette.run` surface channel. Honors the `when` guard:
 * by the time her dispatch lands the context may have moved on, in
 * which case we decline quietly rather than firing a stale action.
 */
export function runCommandById(id) {
  const cmd = _commands.find((c) => c.id === id);
  if (!cmd) {
    console.info('[palette] agent action unknown', id);
    return false;
  }
  try {
    if (cmd.when && !cmd.when()) {
      showToast(`"${cmd.label}" isn't available right now`, 'info');
      return false;
    }
    cmd.run();
    _pushRecent({ kind: 'action', key: id });
    return true;
  } catch (err) {
    console.warn('[palette] agent action failed', id, err);
    showToast(`Couldn't run "${cmd.label}"`, 'error');
    return false;
  }
}

// ── Agent catalog sync (app menu) ────────────────────────────────────
// The companion's app.act verb matches intent against this catalog
// server-side. We sync the agent-enabled commands with their CURRENT
// `when` liveness; surfaces whose context flips (music starts playing)
// call refreshAgentCatalog() so liveness stays honest.

let _agentSyncTimer = null;

export function agentCatalog() {
  return _commands
    .filter((c) => c.agent && c.agent.description)
    .map((c) => ({
      id: c.id,
      description: String(c.agent.description),
      keywords: String(c.keywords || ''),
      stakes: String(c.agent.stakes || 'trivial_reversible'),
      speak: String(c.agent.speak || ''),
      live: (() => {
        try { return !c.when || !!c.when(); } catch (_) { return false; }
      })(),
    }));
}

export function refreshAgentCatalog() {
  _scheduleAgentSync();
}

function _scheduleAgentSync() {
  clearTimeout(_agentSyncTimer);
  _agentSyncTimer = setTimeout(() => {
    const entries = agentCatalog();
    if (!entries.length) return;
    import('./architect-observer.js')
      .then((m) => m.reportAttention('surface.commands.catalog', { entries }))
      .catch(() => {});
  }, 800);
}

export function open() {
  if (!_overlay) return;
  if (_open) {
    _input.focus();
    _input.select();
    return;
  }
  _open = true;
  _overlay.classList.remove('hidden');
  // Animate in next frame for the opacity transition.
  requestAnimationFrame(() => _overlay.classList.add('visible'));
  _input.value = '';
  _input.focus();
  _runHooks();
  _refreshResults();
}

export function close() {
  if (!_overlay || !_open) return;
  _open = false;
  _overlay.classList.remove('visible');
  // Allow the transition to finish before hiding so the next open
  // animates again from opacity 0.
  setTimeout(() => {
    if (!_open) _overlay.classList.add('hidden');
  }, 140);
}

export function onOpen(fn) {
  if (typeof fn === 'function') _onOpenHooks.push(fn);
}

function _runHooks() {
  for (const fn of _onOpenHooks) {
    try { fn(); } catch (err) { console.warn('command-palette hook failed', err); }
  }
}

/**
 * Internals
 */

function _onInput() {
  _activeIndex = 0;
  _refreshResults();
}

function _onKeyDown(ev) {
  if (ev.key === 'Escape') {
    ev.preventDefault();
    close();
    return;
  }
  if (ev.key === 'ArrowDown') {
    ev.preventDefault();
    if (!_currentResults.length) return;
    _setActiveIndex(Math.min(_activeIndex + 1, _currentResults.length - 1));
    return;
  }
  if (ev.key === 'ArrowUp') {
    ev.preventDefault();
    if (!_currentResults.length) return;
    _setActiveIndex(Math.max(_activeIndex - 1, 0));
    return;
  }
  if (ev.key === 'Enter') {
    ev.preventDefault();
    if (_currentResults.length) _executeResultAt(_activeIndex);
    return;
  }
  // Home / End for snappy nav in long lists.
  if (ev.key === 'Home') {
    ev.preventDefault();
    _setActiveIndex(0);
    return;
  }
  if (ev.key === 'End') {
    ev.preventDefault();
    _setActiveIndex(_currentResults.length - 1);
  }
}

function _setActiveIndex(idx) {
  _activeIndex = idx;
  const rows = _resultsEl.querySelectorAll('.cp-result');
  rows.forEach((r, i) => r.classList.toggle('active', i === idx));
  const target = rows[idx];
  if (target) target.scrollIntoView({ block: 'nearest' });
}

async function _refreshResults() {
  const query = (_input.value || '').trim();
  const lower = query.toLowerCase();
  const isActionFilter = lower.startsWith('>');
  const isFileFilter = lower.startsWith('/');
  const term = isActionFilter ? lower.slice(1).trim()
             : isFileFilter ? lower.slice(1).trim()
             : lower;

  const wantFiles = !isActionFilter;
  const wantWorkspaces = !isActionFilter && !isFileFilter;
  const wantActions = !isFileFilter;

  // Empty query → recent items first.
  if (!term && !isActionFilter && !isFileFilter) {
    const recent = _loadRecent();
    if (recent.length) {
      const seeded = recent
        .map((entry) => _materializeRecent(entry))
        .filter(Boolean);
      _renderResults(seeded.length ? seeded : _defaultPicks());
    } else {
      _renderResults(_defaultPicks());
    }
    return;
  }

  // Load sources in parallel. Each returns [] on miss / error so the
  // results array is concatenable without further guards.
  const [filesPool, workspacesPool] = await Promise.all([
    wantFiles ? _getFiles() : Promise.resolve([]),
    wantWorkspaces ? _getWorkspaces() : Promise.resolve([]),
  ]);

  const actionsPool = wantActions ? _filterableActions() : [];

  const results = [
    ...(wantActions ? _scoreItems(actionsPool, term, _actionToResult, 'action') : []),
    ...(wantFiles ? _scoreItems(filesPool, term, _fileToResult, 'file') : []),
    ...(wantWorkspaces ? _scoreItems(workspacesPool, term, _workspaceToResult, 'workspace') : []),
  ];

  // Sort by score desc, then natural label asc, cap at 60 to keep
  // the DOM cheap on long workspaces.
  results.sort((a, b) => (b.score - a.score) || a.label.localeCompare(b.label));
  _renderResults(results.slice(0, 60));
}

/**
 * Default picks shown on an empty query when there are no recent
 * items yet — surfaces a handful of high-leverage actions so the
 * palette is discoverable from cold open.
 */
function _defaultPicks() {
  const all = _filterableActions();
  return all.slice(0, 8).map((cmd) => _actionToResult(cmd, 100));
}

function _materializeRecent(entry) {
  // Recent entries store {kind, key, label, hint, group}. We rebuild
  // a runnable result by resolving the key against current state —
  // this way a deleted workspace / removed command quietly drops out
  // of recents instead of crashing on execute.
  if (entry.kind === 'action') {
    const cmd = _commands.find((c) => c.id === entry.key);
    if (!cmd || (cmd.when && !cmd.when())) return null;
    return _actionToResult(cmd, 100);
  }
  if (entry.kind === 'file') {
    // No way to verify file existence without a fetch — surface as-is
    // and let the open path show an error if it's gone.
    return {
      kind: 'file',
      score: 100,
      label: entry.label,
      hint: entry.hint || '',
      group: 'Recent file',
      icon: 'F',
      run: () => _resolveOpenFile(_resolveActiveWorkspaceId(), entry.key, entry.label),
      recentKey: entry.key,
      recentKind: 'file',
    };
  }
  if (entry.kind === 'workspace') {
    return {
      kind: 'workspace',
      score: 100,
      label: entry.label,
      hint: entry.hint || '',
      group: 'Recent workspace',
      icon: 'W',
      run: () => _resolveSwitchWorkspace(entry.key),
      recentKey: entry.key,
      recentKind: 'workspace',
    };
  }
  return null;
}

function _filterableActions() {
  return _commands.filter((c) => !c.when || c.when());
}

/**
 * Light fuzzy score — substring match wins, character-order match
 * counts, exact prefix gets a bonus. Not a real fuzzy library; this
 * is good enough for ~5000-row datasets and ships with zero deps.
 *
 * Returns 0 for no-match so the caller can filter, positive otherwise.
 */
function _fuzzyScore(text, query) {
  if (!query) return 1;
  if (!text) return 0;
  const t = text.toLowerCase();
  const q = query.toLowerCase();
  if (t === q) return 1000;
  if (t.startsWith(q)) return 600 - (t.length - q.length);
  const idx = t.indexOf(q);
  if (idx >= 0) return 400 - idx - (t.length - q.length) * 0.1;

  // Subsequence: every char in q appears in t in order.
  let ti = 0;
  let qi = 0;
  let runs = 0;
  let inRun = false;
  while (ti < t.length && qi < q.length) {
    if (t[ti] === q[qi]) {
      if (!inRun) { runs += 1; inRun = true; }
      qi += 1;
    } else {
      inRun = false;
    }
    ti += 1;
  }
  if (qi < q.length) return 0;
  // Fewer runs (more contiguous) = higher score.
  return Math.max(50, 200 - runs * 10 - (t.length - q.length) * 0.05);
}

function _scoreItems(items, query, builder, kind) {
  if (!items.length) return [];
  const scored = [];
  for (const item of items) {
    const text = kind === 'action'
      ? `${item.label} ${item.keywords || ''} ${item.group || ''}`
      : kind === 'file'
      ? `${item.name} ${item.path}`
      : kind === 'workspace'
      ? `${item.name || ''} ${item.id || ''}`
      : '';
    const score = _fuzzyScore(text, query);
    if (score > 0) scored.push(builder(item, score));
  }
  return scored;
}

function _actionToResult(cmd, score) {
  return {
    kind: 'action',
    score,
    label: cmd.label,
    hint: cmd.hint || '',
    group: cmd.group || 'Action',
    icon: '⌘',
    run: () => cmd.run(),
    recentKey: cmd.id,
    recentKind: 'action',
  };
}

function _fileToResult(f, score) {
  // Compress the path by dropping the /workspace prefix so the row
  // doesn't waste horizontal space on it.
  const display = (f.path || '').replace(/^\/workspace\/?/, '') || f.name;
  const parent = display.includes('/')
    ? display.slice(0, display.lastIndexOf('/'))
    : '';
  return {
    kind: 'file',
    score,
    label: f.name,
    hint: parent,
    group: 'File',
    icon: 'F',
    run: () => _resolveOpenFile(_resolveActiveWorkspaceId(), f.path, f.name),
    recentKey: f.path,
    recentKind: 'file',
  };
}

function _workspaceToResult(ws, score) {
  return {
    kind: 'workspace',
    score,
    label: ws.name || ws.id || 'workspace',
    hint: ws.id ? ws.id.slice(0, 8) : '',
    group: 'Workspace',
    icon: 'W',
    run: () => _resolveSwitchWorkspace(ws.id),
    recentKey: ws.id,
    recentKind: 'workspace',
  };
}

function _renderResults(results) {
  _currentResults = results;
  if (!results.length) {
    _resultsEl.innerHTML = `<div class="cp-empty">No matches. Try a different query, or type <code>&gt;</code> to filter to actions.</div>`;
    return;
  }
  _resultsEl.innerHTML = results.map((r, i) => `
    <div class="cp-result ${i === _activeIndex ? 'active' : ''}" data-idx="${i}" role="option">
      <span class="cp-result-icon" data-kind="${escapeHtml(r.kind)}">${escapeHtml(r.icon)}</span>
      <span class="cp-result-body">
        <span class="cp-result-label">${escapeHtml(r.label)}</span>
        ${r.hint ? `<span class="cp-result-hint">${escapeHtml(r.hint)}</span>` : ''}
      </span>
      <span class="cp-result-group">${escapeHtml(r.group)}</span>
    </div>
  `).join('');
}

function _executeResultAt(idx) {
  const result = _currentResults[idx];
  if (!result) return;
  if (result.recentKind && result.recentKey) {
    _pushRecent({
      kind: result.recentKind,
      key: result.recentKey,
      label: result.label,
      hint: result.hint,
      group: result.group,
    });
  }
  close();
  // Defer to next tick so the close animation isn't competing with
  // the action's own DOM work (especially file-open which mounts an
  // editor and steals focus).
  setTimeout(() => {
    try {
      result.run();
    } catch (err) {
      console.warn('command-palette run failed', err);
      showToast('Action failed', 'error');
    }
  }, 0);
}

/**
 * Source fetchers — keep both cached so consecutive keystrokes don't
 * thrash the backend. Files cache is per-workspace; switching
 * workspaces invalidates it.
 */

async function _getFiles() {
  const workspaceId = _resolveActiveWorkspaceId();
  if (!workspaceId) return [];
  const now = Date.now();
  if (
    _filesCache
    && _filesCache.workspaceId === workspaceId
    && (now - _filesCache.ts) < FILES_CACHE_TTL_MS
  ) {
    return _filesCache.files;
  }
  try {
    const resp = await fetch(
      `/api/coder/workspaces/${encodeURIComponent(workspaceId)}/files-flat`,
      { credentials: 'include' },
    );
    if (!resp.ok) return [];
    const data = await resp.json();
    const files = Array.isArray(data.files) ? data.files : [];
    _filesCache = { workspaceId, ts: now, files };
    return files;
  } catch {
    return [];
  }
}

async function _getWorkspaces() {
  const now = Date.now();
  if (_workspacesCache && (now - _workspacesCache.ts) < WORKSPACES_CACHE_TTL_MS) {
    return _workspacesCache.workspaces;
  }
  try {
    const resp = await fetch('/api/coder/workspaces', { credentials: 'include' });
    if (!resp.ok) return [];
    const data = await resp.json();
    const workspaces = Array.isArray(data) ? data
      : Array.isArray(data.workspaces) ? data.workspaces
      : [];
    _workspacesCache = { ts: now, workspaces };
    return workspaces;
  } catch {
    return [];
  }
}

/**
 * Recent items — small persisted MRU. Keys are typed so a workspace
 * id can't collide with a file path or action id.
 */

function _loadRecent() {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function _pushRecent(entry) {
  if (!entry || !entry.kind || !entry.key) return;
  try {
    const list = _loadRecent().filter(
      (e) => !(e.kind === entry.kind && e.key === entry.key),
    );
    list.unshift(entry);
    if (list.length > RECENT_MAX) list.length = RECENT_MAX;
    localStorage.setItem(RECENT_KEY, JSON.stringify(list));
  } catch { /* quota / private mode — graceful no-op */ }
}

/**
 * Invalidate the file cache. Called by coder.js when the active
 * workspace changes or a turn ends with file mutations.
 */
export function invalidateFilesCache() {
  _filesCache = null;
}
