/**
 * library.js — three-pane Library orchestrator.
 *
 * Public surface (unchanged from the legacy entry):
 *   openLibrary({ initialSelection? })  open the overlay (idempotent)
 *   closeLibrary()                      close it (idempotent)
 *   openCommandPalette()                ⌘K launcher
 *   initLibrary()                       no-op; lazy init on first openLibrary
 *   isOpen()                            state probe
 *
 * Implementation lives in ./library/ (sidebar, main pane, detail pane,
 * three-pane layout, command palette, games-browse, api, types). This file
 * is the orchestrator that:
 *   • builds the overlay DOM once and reuses it
 *   • wires the panes together (selection / search / item-select / bulk)
 *   • registers with ViewStack so close restores focus correctly
 *   • persists open-state in localStorage for refresh recovery
 *   • subscribes to `library:games-source-refresh` so ROM/BIOS uploads
 *     repopulate the active Game Portal source automatically
 *   • subscribes to `artifact:saved` so the detail pane reflects an edit
 *     made elsewhere (workspace / studio)
 *
 * Item classification (`_type`, `_isPublication`) is stamped in
 * ./library/api.js so every fetch path delivers fully-typed items —
 * detail-pane.js's open dispatcher then routes by `_type` alone.
 */

import { fetchHome, importArtifact, listCollections } from './library/api.js';
import { BuildController } from './library/build.js';
import { CommandPalette } from './library/command-palette.js';
import { DetailPane } from './library/detail-pane.js';
import { MainPane } from './library/main-pane.js';
import { Sidebar } from './library/sidebar.js';
import { createThreePane } from './library/three-pane.js';
import { ViewStack } from './view-stack.js';

let _overlay = null;
let _three = null;
let _sidebar = null;
let _mainPane = null;
let _detailPane = null;
let _palette = null;
let _buildController = null;
let _opened = false;
let _eventsWired = false;

const _PALETTE_ACTIONS = [
  { id: 'open-pinned',    label: 'Open Pinned',                       hint: 'View your pinned items' },
  { id: 'open-recent',    label: 'Open Recent',                       hint: 'Recently touched items' },
  { id: 'open-continue',  label: 'Continue where you left off',       hint: 'Resume in-progress items' },
  { id: 'open-all',       label: 'Browse all items',                  hint: 'Everything in your Library' },
  { id: 'open-games',     label: 'Browse Game Portal',                hint: 'js13k / emulator / marketplace' },
  { id: 'forge-game',     label: 'Forge a Game…',                     hint: 'Generate + playtest a new game live' },
  { id: 'new-collection', label: 'New collection…',                   hint: 'Create a custom grouping' },
];

const _state = {
  home: {
    pinned: [], recent: [], continue: [], collections_summary: [],
    type_counts: {}, total_count: 0,
  },
  collections: [],
  // Default selection lands on the dashboard ("nothing yet"); empty kind
  // signals the main pane to render the dashboard state.
  selection: { kind: '', id: '' },
  query: '',
  activeItem: null,
};

// ── DOM construction ───────────────────────────────────────────────

function _ensureDom() {
  if (_overlay) return;

  _overlay = document.createElement('div');
  _overlay.id = 'library-shell-overlay';
  _overlay.className = 'lib-overlay hidden';
  _overlay.setAttribute('role', 'dialog');
  _overlay.setAttribute('aria-modal', 'true');
  _overlay.setAttribute('aria-label', 'Library');
  // tabindex=-1 makes the div programmatically focusable so the
  // openLibrary() ``_overlay.focus()`` call actually moves focus —
  // without this, the keydown listener below never fires on Escape
  // until the user interacts with something inside the overlay first.
  _overlay.setAttribute('tabindex', '-1');

  const header = document.createElement('header');
  header.className = 'lib-header';
  header.innerHTML = `
    <h1 class="lib-title">Library</h1>
    <div class="lib-header-actions">
      <button class="lib-close" id="lib-close-btn" type="button"
              aria-label="Close library (Esc)" title="Close (Esc)">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
             stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
        <span>Close</span>
      </button>
    </div>
  `;
  _overlay.appendChild(header);

  const body = document.createElement('div');
  body.className = 'lib-body';
  _overlay.appendChild(body);
  _three = createThreePane(body);

  _sidebar = new Sidebar(_three.sidebar, {
    onSelect: _handleSelect,
    onSource: _handleSource,
    onSearch: _handleSearch,
  });

  _mainPane = new MainPane(_three.main, {
    onItemSelect: _handleItemSelect,
    onBack: () => _three.showSidebar(),
    onBulkChange: _handleBulkChange,
    getHome: () => _state.home,
    // BuildController owns the live-builds list; the closure defers the
    // lookup so the (later-constructed) controller is in scope by call time.
    getActiveBuilds: () => (_buildController ? _buildController.list() : []),
  });

  // Autonomous app builder: kicks off POST /api/builds, shows live building
  // cards in the main pane, and reloads the library to surface the finished
  // artifact on completion.
  _buildController = new BuildController({
    mainPane: _mainPane,
    onComplete: () => _reload(),
  });

  _detailPane = new DetailPane(_three.detail, {
    onChange: _handleDetailChange,
    onBack: () => _three.showMain(),
  });

  _palette = new CommandPalette({
    onItemActivate: _handleItemSelect,
    onAction: _handlePaletteAction,
  });

  document.body.appendChild(_overlay);

  _overlay.querySelector('#lib-close-btn').addEventListener('click', closeLibrary);
  // Two Escape listeners: the overlay-scoped one handles inputs and
  // sub-widgets that stopPropagation, the document-scoped one is the
  // fallback for when focus is still outside the overlay (the initial
  // moment after openLibrary() before the user has clicked anything).
  // The document listener checks _opened so it no-ops when the overlay
  // isn't visible.
  _overlay.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') {
      ev.preventDefault();
      closeLibrary();
    }
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && _opened) {
      ev.preventDefault();
      closeLibrary();
    }
  });

  _wireSubstrateEvents();
}

// ── Substrate events (window + document) ───────────────────────────

function _wireSubstrateEvents() {
  if (_eventsWired) return;
  _eventsWired = true;

  // Game sources signal a refresh when their inventory changes (ROM
  // upload finishes, BIOS file installed, etc.). Re-render the games
  // browse if we're sitting on it — show() drops into _buildShell +
  // re-fetch which picks up the new title.
  window.addEventListener('library:games-source-refresh', () => {
    if (_state.selection.kind === 'browse-games' && _mainPane && _opened) {
      _mainPane.show(_state.selection, { query: _state.query });
    }
  });

  // When an artifact is saved elsewhere (workspace or studio), the
  // detail pane may be showing it — refresh so display_name / tags /
  // hero preview stay current.
  document.addEventListener('artifact:saved', (ev) => {
    const id = ev?.detail?.id;
    if (!id || !_opened) return;
    if (_state.activeItem?.id === id) {
      // Cheapest correctness: kick a full reload so counts + recents
      // pick up the change too.
      _reload();
    }
  });
}

// ── Handlers ───────────────────────────────────────────────────────

function _handleSelect(selection) {
  _state.selection = selection;
  _sidebar.render(_state.home, _state.collections, _state.selection);
  _three.showMain();
  _mainPane?.show(selection, { query: _state.query });
}

function _handleSource(action) {
  switch (action) {
    case 'build-app':      _buildController?.openDialog(); break;
    case 'import':         _openImportPicker();      break;
    case 'browse-games':   _handleSelect({ kind: 'browse-games', id: '' }); break;
    case 'create':         _openCreateFlow();        break;
    case 'new-collection': _openNewCollectionFlow(); break;
    default: console.log('[library] unknown source action', action);
  }
}

function _ensureImportInput() {
  if (_state._importInput) return _state._importInput;
  const input = document.createElement('input');
  input.type = 'file';
  input.style.display = 'none';
  input.addEventListener('change', async (ev) => {
    const file = ev.target.files?.[0];
    ev.target.value = '';
    if (!file) return;
    await _doImport(file);
  });
  document.body.appendChild(input);
  _state._importInput = input;
  return input;
}

async function _doImport(file) {
  const { showToast } = await import('./app.js').catch(() => ({}));
  try {
    const data = await importArtifact(file);
    showToast?.(`Imported: ${file.name}`, 'success');
    await _reload();
    // If the import returned a usable id, jump the user straight into
    // the new item so they see the import landed (Steam's "your new
    // game is here" moment).
    if (data?.id) {
      _state.activeItem = { id: data.id, ...data };
      _detailPane?.show(_state.activeItem);
      _three.showDetail();
    }
  } catch (err) {
    showToast?.(err.message || 'Import failed', 'error');
  }
}

function _openImportPicker() {
  _ensureImportInput().click();
}

function _openCreateFlow() {
  // Empty-state CTA from the legacy library: close, open agentic mode,
  // focus the chat input. Future: an in-Library "What do you want to
  // build?" sheet that scaffolds in place.
  closeLibrary();
  document.querySelector('[data-mode="agentic"]')?.click();
  const input = document.querySelector('#chat-input, .chat-input, textarea');
  input?.focus();
}

function _openNewCollectionFlow() {
  const name = prompt('Name your collection');
  if (!name || !name.trim()) return;
  import('./library/api.js').then(async (m) => {
    try {
      await m.createCollection({ name: name.trim() });
      await _reload();
    } catch (err) {
      console.warn('[library] create collection failed', err);
    }
  });
}

function _handleSearch(query) {
  _state.query = query;
  _mainPane?.setQuery(query);
}

function _handleItemSelect(item) {
  _state.activeItem = item;
  if (typeof window !== 'undefined') {
    window.__libraryActiveItem = item;
  }
  _detailPane?.show(item);
  _three.showDetail();
}

function _handlePaletteAction(action) {
  if (!action) return;
  switch (action.id) {
    case 'open-pinned':    _handleSelect({ kind: 'pinned',   id: '' });        break;
    case 'open-recent':    _handleSelect({ kind: 'recent',   id: '' });        break;
    case 'open-continue':  _handleSelect({ kind: 'continue', id: '' });        break;
    case 'open-all':       _handleSelect({ kind: 'all',      id: '' });        break;
    case 'open-games':     _handleSelect({ kind: 'browse-games', id: '' });    break;
    case 'forge-game':     import('./foundry/theater.js').then((m) => m.openFoundryTheater()); break;
    case 'new-collection': _handleSource('new-collection');                    break;
    default: console.log('[library] palette action', action);
  }
}

async function _handleBulkChange(summary) {
  // Counts + activity may have shifted; cheapest correctness is to
  // refresh the home payload (drives the sidebar counts) and reload
  // any in-flight collection view.
  await _reload();
  const sel = _state.selection;
  if (sel?.kind === 'collection' && sel.id) {
    _mainPane?.show(sel, { query: _state.query });
  }
  if (summary?.deleted) {
    const { showToast } = await import('./app.js').catch(() => ({}));
    showToast?.(
      `Deleted ${summary.deleted} item${summary.deleted === 1 ? '' : 's'}`,
      'success',
    );
  }
}

function _handleDetailChange(change) {
  // Pin / tag mutations may shift the visible item in the main list
  // (a newly-pinned item could change Pinned-sort order, a new tag could
  // make it pop into a Tag-filtered collection). Cheapest correctness
  // move: refresh the home payload so counts + recents update; defer
  // re-sorting the main list to the next selection change.
  if (change?.kind === 'pin' || change?.kind === 'delete') {
    _reload();
  }
}

// ── Load + reload ──────────────────────────────────────────────────

async function _reload() {
  try {
    const [home, listing] = await Promise.all([
      fetchHome(),
      listCollections(),
    ]);
    _state.home = home;
    _state.collections = listing.collections || [];
    if (typeof window !== 'undefined') {
      window.__libraryState = _state;  // dev hook for inspection
    }
  } catch (err) {
    console.warn('[library] reload failed', err);
  }
  _sidebar.render(_state.home, _state.collections, _state.selection);
  _mainPane?.show(_state.selection, { query: _state.query });
}

// ── Public API ─────────────────────────────────────────────────────

/**
 * Open the Library overlay.
 *
 * @param {object} [opts]
 * @param {{kind:string,id?:string}} [opts.initialSelection] — land directly
 *   on a sidebar entry (e.g. `{ kind: 'browse-games' }` for the Game Portal,
 *   `{ kind: 'pinned' }` for the pinned set). Default is the dashboard.
 */
export async function openLibrary(opts = {}) {
  _ensureDom();
  if (opts?.initialSelection?.kind) {
    _state.selection = {
      kind: opts.initialSelection.kind,
      id: opts.initialSelection.id || '',
    };
  }
  _overlay.classList.remove('hidden');
  _overlay.focus();
  _opened = true;
  // Persist for refresh-recovery (app.js init reads this and re-opens).
  try { localStorage.setItem('augmentum_library_open', '1'); } catch {}
  // Register with ViewStack so close restores focus to whatever was
  // underneath and a backdrop-Esc anywhere in the stack pops us.
  ViewStack.pushOverlay('library', { onClose: _doCloseLibrary });
  await _reload();
}

export function openCommandPalette() {
  _ensureDom();
  _palette?.open(_PALETTE_ACTIONS);
}

export function closeLibrary() {
  // Routing through ViewStack guarantees onClose fires exactly once,
  // even if a child surface (workspace / studio / game) was pushed on
  // top of us and we get closed indirectly when they pop.
  if (ViewStack.hasOverlay('library')) {
    ViewStack.popOverlay('library');  // → _doCloseLibrary
    return;
  }
  _doCloseLibrary();
}

function _doCloseLibrary() {
  if (!_overlay) return;
  _overlay.classList.add('hidden');
  _opened = false;
  try { localStorage.removeItem('augmentum_library_open'); } catch {}
  // The command palette is overlay-scoped; close it if it was open.
  _palette?.close?.();
}

export function isOpen() {
  return _opened;
}

export function initLibrary() {
  // No-op. Library DOM is built lazily on first openLibrary() call;
  // exported so legacy callers that did initLibrary() at boot continue
  // to compile without changes.
}

// Dev hooks for the console / debugging.
if (typeof window !== 'undefined') {
  window.openLibrary = openLibrary;
  window.closeLibrary = closeLibrary;
  window.openLibraryPalette = openCommandPalette;
}
