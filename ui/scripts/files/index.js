/**
 * Files panel — entry point. Owns DOM caching, event wiring, keyboard
 * shortcuts, and the public API (initFiles / openFiles / closeFiles /
 * toggleFiles). Re-exported from ../files.js so existing imports keep
 * working.
 */

import { showToast } from '../app.js';
import {
  state, SORT_LABELS, COMICS_SORTS, GENERIC_SORTS_FOR_COMICS_PROMOTION,
  STATUS_LABELS, MEDIA_CHIP, PODCASTS_CHIP, AUDIO_LIBRARY_CHIPS,
  COMICS_CHIP, VIDEO_CLOUD_CHIPS,
  SHOWS_CHIP, MOVIES_CHIP, MUSIC_VIDEOS_CHIP, savePref,
} from './state.js';
import { isTextInput } from './helpers.js';
import { fetchListPage, uploadFiles, fetchFileEntry } from './api.js';
import {
  renderGrid, appendCards, updateSentinel, observeSentinel, setupScrollObserver,
  updateViewToggle, selectOnly, selectAll, deselectAll, toggleSelect, selectRange,
  updateSelectionUI, updateDetail, startRename,
  setSelectMode, exitSelectMode,
} from './render.js';
import { observeVisiblePeeks, setupPeekObserver } from './helpers.js';
import {
  showContextMenu, hideContextMenu, activateFile,
  toggleFavoriteAction, deleteFileAction, bulkDownload, bulkDelete,
  restoreAll, emptyTrash, referenceInChat, summarizeFile, loadStats,
  downloadFile, readAloudFile, unpinLibrivoxAction,
  narrationTap, narrationHoldStart, narrationHoldCancel,
} from './actions.js';
import { openGallery, openMediaPreview, openProject } from './preview.js';
import { openReadAlong } from './read-along.js';
import {
  isComicsChipActive, renderComicsGrid, resetComicsView, initComicsListeners,
} from './comics.js';
import {
  isLiveTvChipActive, renderLiveTvRails, resetLiveTvView, initLiveTvListeners,
} from './live-tv-rails.js';
import { renderContinueRail, patchRailProgress } from './continue-rail.js';

// --- DOM cache -------------------------------------------------------

function _cacheDom() {
  const $ = (id) => document.getElementById(id);
  state.el = {
    panel:        $('files-panel'),
    closeBtn:     $('files-close-btn'),
    backBtn:      $('files-back-btn'),
    backBtnLabel: document.querySelector('#files-back-btn .files-topbar-back-label'),
    viewToggle:   $('files-view-toggle'),
    search:       $('files-search-input'),
    scopeChooser: $('files-scope-chooser'),
    tabs:         $('files-source-tabs'),
    grid:         $('files-grid'),
    detail:       $('files-detail'),
    continueRail: $('files-continue-rail'),
    stats:        $('files-stats'),
    sortBtn:      $('files-sort-btn'),
    sortDropdown: $('files-sort-dropdown'),
    filterBtn:      $('files-filter-btn'),
    filterDropdown: $('files-filter-dropdown'),
    bulkBar:      $('files-bulk-bar'),
    bulkCount:    $('files-bulk-count'),
    uploadBtn:    $('files-upload-btn'),
    uploadInput:  $('files-upload-input'),
    dropOverlay:  $('files-drop-overlay'),
    mediaBtn:     $('files-media-btn'),
    mediaViewBar: $('files-media-view-bar'),
    selectBtn:    $('files-select-btn'),
    ttsStudioBtn: $('files-tts-studio-btn'),
  };
}

// --- Upload handler --------------------------------------------------

// Local byte-formatter — keeps the toast self-contained without
// cross-module coupling. Matches the convention used elsewhere
// (1024-based, two decimals at MB+). The backend reports raw byte
// counts in error strings; we humanize them on display so the user
// can actually tell whether their file is "a bit over" or "way over"
// the limit.
function _humanBytes(n) {
  if (typeof n !== 'number' || !isFinite(n) || n < 0) return String(n);
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// Rewrite backend error strings that contain "(N bytes)" → "(N MB)"
// so the toast is readable. Other error shapes pass through unchanged.
function _humanizeUploadError(msg) {
  if (typeof msg !== 'string') return String(msg || 'failed');
  return msg.replace(/\((\d{4,})\s*bytes\)/g, (_, n) => `(${_humanBytes(parseInt(n, 10))})`);
}

async function _handleUpload(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  showToast(`Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`, 'info', 2000);
  try {
    const result = await uploadFiles(files);
    const ok = (result.uploaded || []).length;
    const errs = result.errors || [];
    const dedup = (result.uploaded || []).filter(u => u.deduped).length;

    if (errs.length === 0) {
      // Pure success path — single concise toast, dedup detail if any.
      let msg = `Uploaded ${ok}`;
      if (dedup) msg += ` (${dedup} already stored)`;
      showToast(msg, 'success', 3000);
    } else {
      // Surface the actual reason(s) so the user can act on it instead
      // of guessing whether it was network, cap, quota, MIME, etc. The
      // backend already returns {filename, error} per failure — we were
      // just throwing it away. Show the first specific reason, plus a
      // count if there are more, plus the success count if non-zero.
      const first = errs[0];
      const firstName = first?.filename ? `${first.filename}: ` : '';
      const firstMsg = _humanizeUploadError(first?.error || 'unknown error');
      const moreSuffix = errs.length > 1 ? ` (+${errs.length - 1} more)` : '';
      const okPrefix = ok > 0 ? `${ok} uploaded · ` : '';
      const msg = `${okPrefix}Failed — ${firstName}${firstMsg}${moreSuffix}`;
      showToast(msg, ok > 0 ? 'warning' : 'error', 6000);
      // Long-form detail to console for users who want to see all of
      // them — keeps the toast readable without burying information.
      console.warn('[files] upload errors:', errs);
    }

    // Refresh list + stats to pick up new rows and chip counts
    loadFiles({ reset: true });
    const { loadStats } = await import('./actions.js');
    loadStats();
  } catch (err) {
    console.warn('[files] upload error:', err);
    showToast(`Upload failed: ${err.message || 'network error'}`, 'error');
  }
}

// --- Data loading ----------------------------------------------------

export async function loadFiles({ reset = true } = {}) {
  if (state.loading) return;
  if (!reset && !state.hasMore) return;
  state.loading = true;
  if (reset) {
    state.offset = 0;
    state.files = [];
    state.hasMore = false;
    state.focusedIndex = -1;
    state.lastClickedIndex = -1;
    state.lastRenderedBucket = '';
    // Update the continue rail BEFORE the comics early-return below,
    // so switching from a media chip to comics (or any non-rail chip)
    // immediately hides the rail rather than leaving the previous
    // chip's items visible. The function synchronously hides for
    // chips outside RAIL_CHIPS — no network round-trip in that path.
    renderContinueRail(state.currentSource);
  }
  // Comics chip runs its own series/chapters renderer — the generic
  // list pipeline would return flat chapter rows (20k+) which is the
  // exact UX we're avoiding. Delegate + return; comics.js manages
  // its own breadcrumb, fetch, and empty-state.
  if (isComicsChipActive()) {
    try {
      await renderComicsGrid();
    } finally {
      state.loading = false;
    }
    return;
  }
  // Live TV chip is rail-based, not grid-based. Same delegation
  // contract as comics — file_index doesn't carry live channels
  // (no static rows, EPG-driven 'now playing'), so the generic
  // list pipeline would return zero rows here anyway.
  if (isLiveTvChipActive()) {
    try {
      await renderLiveTvRails();
    } finally {
      state.loading = false;
    }
    return;
  }
  try {
    const data = await fetchListPage(state.offset);
    if (!data) return;
    const batch = data.files || [];
    const startIndex = state.files.length;
    state.files = reset ? batch : state.files.concat(batch);
    state.hasMore = !!data.has_more;
    state.offset += batch.length;
    if (reset) {
      const alive = new Set(state.files.map(f => f.id));
      for (const id of [...state.selection]) {
        if (!alive.has(id)) state.selection.delete(id);
      }
      renderGrid();
      // (Continue rail already kicked off at the top of loadFiles
      // before the comics early-return — don't duplicate the fetch.)
    } else {
      appendCards(batch, startIndex);
      updateSentinel();
      observeSentinel();
      observeVisiblePeeks();
    }
  } catch (err) {
    console.warn('[files] load error:', err);
  } finally {
    state.loading = false;
  }
}

// --- Chip / search / view / sort handlers ----------------------------

function _setSource(source) {
  // Leaving the Comics chip: discard any drill-down state so returning
  // to Comics later lands on the top-level series grid, not a stale
  // chapter list for a series the user may not remember drilling into.
  if (state.currentSource === 'comics' && source !== 'comics') {
    resetComicsView();
  }
  if (state.currentSource === 'live_tv' && source !== 'live_tv') {
    resetLiveTvView();
  }
  state.currentSource = source;
  savePref('source', source);
  state.el.tabs?.querySelectorAll('.files-chip').forEach(chip => {
    chip.classList.toggle('active', chip.dataset.source === source);
  });
  deselectAll();
  if (source === 'images' && state.currentView === 'grid') {
    state.currentView = 'gallery';
  }
  // Chip-aware chrome reveals/hides with the chip. Filter dropdown
  // contents adapt to the current chip's schema; Filter button hides
  // entirely on chips with no filter dimensions. Entering Comics
  // triggers a genre-list refresh so the genre section is populated
  // by the time the user opens the dropdown.
  _renderMediaViewBar();
  _renderFilterButton();
  _closeFilterDropdown();   // close stale popover from previous chip
  _renderSortOptions();
  // View-toggle (grid/list/gallery) only applies to chips that flow
  // through the generic renderGrid() pipeline. Comics owns its own
  // layout (series cards → series detail), so the toggle is hidden
  // there to avoid a misleading no-op control. Mirrors the Filter
  // button's chip-aware visibility.
  // View-toggle hides for chips that render their own layout (comics
  // = series cards, live_tv = horizontal rails). The grid/list/gallery
  // pills would be misleading no-ops on those chips.
  state.el.viewToggle?.classList.toggle(
    'hidden', source === COMICS_CHIP || source === 'live_tv',
  );
  // The "Record" (TTS studio) button belongs to the local Audio section.
  state.el.ttsStudioBtn?.classList.toggle('hidden', source !== 'audio');
  // Reset the per-section expanded state on chip switch — the schema
  // changes and a stale "expanded" entry from a different chip's
  // section would either miss (different id) or, worse, collide on a
  // shared id like 'genre'. Cleaner to start fresh.
  _filterExpanded.clear();
  if (source === COMICS_CHIP) _refreshComicGenres();
  if (VIDEO_CLOUD_CHIPS.has(source)) _refreshVideoGenres();
  loadFiles({ reset: true });
}

// --- Comic genres ----------------------------------------------------
// Cached for the session — a user's library genre-set doesn't change
// mid-session. Refreshes when the Comics chip is entered or when the
// media-servers:changed event fires (new series synced).

let _comicGenresCache = null;
let _videoGenresCache = null;

async function _refreshComicGenres() {
  // Cache populates after the chip enters Comics; the unified Filter
  // dropdown reads from _comicGenresCache when it builds the genre
  // section. If the dropdown is open when fresh data lands, re-render
  // it in place so the new options appear without needing a re-open.
  try {
    const resp = await fetch('/api/files/comics/genres');
    if (!resp.ok) return;
    const data = await resp.json();
    _comicGenresCache = Array.isArray(data?.genres) ? data.genres : [];
    if (state.el.filterDropdown?.classList.contains('open')) {
      _renderFilterDropdown();
    }
  } catch (err) {
    console.warn('[files] genres fetch failed:', err);
  }
}

async function _refreshVideoGenres() {
  // Mirror of _refreshComicGenres for the video chip family. Backend
  // /video/genres aggregates distinct genres across user's Emby /
  // Jellyfin rows (entity_kind in series/movie/music_video) so the
  // count reflects rollup-level entities rather than per-episode
  // inflation.
  try {
    const resp = await fetch('/api/files/video/genres');
    if (!resp.ok) return;
    const data = await resp.json();
    _videoGenresCache = Array.isArray(data?.genres) ? data.genres : [];
    if (state.el.filterDropdown?.classList.contains('open')) {
      _renderFilterDropdown();
    }
  } catch (err) {
    console.warn('[files] video genres fetch failed:', err);
  }
}

function escapeAttr(s) {
  return String(s || '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// --- Scope (Local / Cloud) --------------------------------------------
// Scope is the top-level split between user-authored/uploaded content
// ("Local") and content from connected remote servers ("Cloud"). Chips
// below the scope toggle declare `data-scope="local|cloud|any"` so this
// function can hide/show them in one pass.

function _setScope(scope) {
  if (scope !== 'local' && scope !== 'cloud') return;
  if (scope === state.currentScope) return;

  // Crossfade the chip cloud through the swap so the chip set reads
  // as "different set, same surface" rather than a hard jump-cut.
  // .scope-changing → opacity: 0 (CSS rule); we apply the membership
  // change in the middle of the fade and remove the class on the
  // following frame so the new chips fade back in. Reduced-motion
  // users skip the fade via the matching CSS guard.
  const tabs = state.el.tabs;
  const animate = tabs && !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

  const applyChange = () => {
    state.currentScope = scope;
    savePref('scope', scope);

    _renderScopeToggle();
    _renderChipsForScope();

    // If the previously-selected chip doesn't exist in the new scope,
    // fall back to 'all' — same-scope. Favorites, Trash, All survive
    // the switch because they're data-scope="any". Comics/Audiobooks
    // only exist on Cloud; Images/Documents/Audio/Code/Archives only
    // exist on Local.
    const currentChip = state.el.tabs?.querySelector(
      `.files-chip[data-source="${state.currentSource}"]`,
    );
    const chipOrphaned = currentChip && _chipHiddenForScope(currentChip);
    _setSource(chipOrphaned ? 'all' : state.currentSource);

    // Chip count badges are scope-partitioned — refresh so numbers match
    // the grid the user is about to see under the new scope.
    loadStats();
  };

  if (!animate) {
    applyChange();
    return;
  }

  // Two-phase: fade out, swap chips, fade back in. The 180ms here
  // matches the CSS transition duration on .files-chips.
  tabs.classList.add('scope-changing');
  setTimeout(() => {
    applyChange();
    // requestAnimationFrame ensures the DOM mutations from applyChange
    // have committed before we drop the fade class — otherwise
    // browsers may collapse the two opacity changes into a single
    // frame and skip the fade-in entirely.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        tabs.classList.remove('scope-changing');
      });
    });
  }, 180);
}

function _renderScopeToggle() {
  // Title-line chooser button — single label tracking active scope.
  // The popover renders both options with counts; this is just the
  // collapsed state. Capitalize for display: state stores 'local' /
  // 'cloud' but the button reads "Local" / "Cloud".
  const chooser = state.el.scopeChooser;
  if (!chooser) return;
  const label = chooser.querySelector('[data-scope-current]');
  if (label) {
    label.textContent = state.currentScope === 'cloud' ? 'Cloud' : 'Local';
  }
  // aria-pressed isn't right here (chooser opens a menu, not toggle a
  // boolean). aria-expanded tracks the popover state, set by open/close.
}

// --- Scope chooser popover -----------------------------------------------
// Tap the title-line scope button → small popover offering Local / Cloud
// with current item counts. Outside-click and Escape dismiss. Mirrors
// the comics chapter "more" popover pattern so the menu chrome reads
// as one consistent element across the app.

let _scopeMenu = null;
let _scopeMenuDocClick = null;

function _closeScopeMenu() {
  if (_scopeMenu) {
    _scopeMenu.remove();
    _scopeMenu = null;
  }
  if (_scopeMenuDocClick) {
    document.removeEventListener('click', _scopeMenuDocClick, true);
    document.removeEventListener('keydown', _scopeMenuDocClick, true);
    _scopeMenuDocClick = null;
  }
  state.el.scopeChooser?.setAttribute('aria-expanded', 'false');
}

function _openScopeMenu() {
  _closeScopeMenu();
  const anchor = state.el.scopeChooser;
  if (!anchor) return;
  const localCount = document.querySelector('[data-scope-count="local"]')?.textContent || '';
  const cloudCount = document.querySelector('[data-scope-count="cloud"]')?.textContent || '';
  const menu = document.createElement('div');
  menu.className = 'files-scope-menu';
  menu.setAttribute('role', 'menu');
  menu.innerHTML = `
    <button type="button" role="menuitem" data-scope="local"
            class="files-scope-menu-item${state.currentScope === 'local' ? ' is-active' : ''}">
      <span class="files-scope-menu-label">Local</span>
      <span class="files-scope-menu-meta">${localCount || ''}</span>
    </button>
    <button type="button" role="menuitem" data-scope="cloud"
            class="files-scope-menu-item${state.currentScope === 'cloud' ? ' is-active' : ''}">
      <span class="files-scope-menu-label">Cloud</span>
      <span class="files-scope-menu-meta">${cloudCount || ''}</span>
    </button>
  `;
  document.body.appendChild(menu);
  _scopeMenu = menu;

  // Position under the chooser, left-aligned with it. Clamp to viewport.
  const r = anchor.getBoundingClientRect();
  const mw = menu.offsetWidth;
  const left = Math.max(8, Math.min(window.innerWidth - mw - 8, r.left));
  const top = Math.min(window.innerHeight - menu.offsetHeight - 8, r.bottom + 6);
  menu.style.left = `${left}px`;
  menu.style.top  = `${top}px`;

  anchor.setAttribute('aria-expanded', 'true');

  _scopeMenuDocClick = (e) => {
    if (e.type === 'keydown' && e.key !== 'Escape') return;
    if (e.type === 'click' && (menu.contains(e.target) || anchor.contains(e.target))) return;
    _closeScopeMenu();
  };
  setTimeout(() => {
    document.addEventListener('click',   _scopeMenuDocClick, true);
    document.addEventListener('keydown', _scopeMenuDocClick, true);
  }, 0);

  menu.addEventListener('click', (e) => {
    const item = e.target.closest('[data-scope]');
    if (!item) return;
    e.preventDefault();
    e.stopPropagation();
    const scope = item.dataset.scope;
    _closeScopeMenu();
    if (scope) _setScope(scope);
  });
}

function _chipHiddenForScope(chip) {
  const chipScope = chip.dataset.scope || 'any';
  return chipScope !== 'any' && chipScope !== state.currentScope;
}

function _renderChipsForScope() {
  const tabs = state.el.tabs;
  if (!tabs) return;
  // Hide chips/dividers that don't belong to the current scope. Using
  // a class rather than the hidden attribute so CSS transitions work.
  tabs.querySelectorAll('[data-scope]').forEach(el => {
    el.classList.toggle('files-chip-hidden', _chipHiddenForScope(el));
  });
}

function _isMediaSource(source) {
  return AUDIO_LIBRARY_CHIPS.has(source);
}

function _supportsCatalogSource(source) {
  return source === MEDIA_CHIP;
}

function _isVideoCloudSource(source) {
  return VIDEO_CLOUD_CHIPS.has(source);
}

function _supportsProgressSort() {
  return _isMediaSource(state.currentSource)
    || _isVideoCloudSource(state.currentSource)
    || (state.currentScope === 'cloud' && state.currentSource === 'all');
}

function _videoChipForFile(file) {
  const kind = String(file?.source_metadata?.entity_kind || '').toLowerCase();
  if (['series', 'season', 'episode'].includes(kind)) return SHOWS_CHIP;
  if (kind === 'movie') return MOVIES_CHIP;
  if (kind === 'music_video') return MUSIC_VIDEOS_CHIP;
  return 'all';
}

// --- Unified Filter dropdown ---------------------------------------------
// Single button + popover whose contents adapt to the active chip:
//   - audiobooks / podcasts → playback status
//   - comics                → read state + series status + genre
//   - everything else       → no dimensions, button hides
//
// Sections are rendered from per-chip schemas defined inline below. Each
// schema entry binds a label set to the appropriate state setter
// (_setMediaStatus / _setComicCompletion / etc.), so the existing
// per-filter click → state → reload pipeline stays intact — the
// dropdown is just a different surface for the same writes.

// Per-section expanded state. Persists across re-renders so picking an
// option doesn't collapse the section the user just opened. Cleared on
// chip switch (via _setSource) since the schema membership changes.
const _filterExpanded = new Set();

// Year-range bounds for the slider — start in 1900 since film history
// reliably begins around then; upper bound is "next year" so a movie
// dated 2026 in early-2026 still falls inside the slider's max.
const _YEAR_MIN_BOUND = 1900;
const _YEAR_MAX_BOUND = new Date().getFullYear() + 1;

function _filterSchemaForChip() {
  // Returns an array of section descriptors. Empty array = no filters
  // for the current chip; the button hides itself in that case.
  // Section kinds: 'radio' (radios), 'list' (scrollable, data-driven),
  // 'range' (two-thumb slider for numeric ranges).
  const onAudiobooks = _supportsCatalogSource(state.currentSource)
    && state.currentMediaView === 'library';
  const onPodcasts = state.currentSource === PODCASTS_CHIP;
  const onComics   = state.currentSource === COMICS_CHIP;
  const onVideo    = VIDEO_CLOUD_CHIPS.has(state.currentSource);

  if (onAudiobooks || onPodcasts) {
    return [{
      id: 'playback',
      label: 'Playback',
      kind: 'radio',
      currentValue: state.currentMediaStatus,
      defaultValue: 'all',
      options: [
        { value: 'all',          label: 'All' },
        { value: 'in_progress',  label: 'In progress' },
        { value: 'finished',     label: 'Finished' },
        { value: 'not_started',  label: 'Not started' },
      ],
      setter: _setMediaStatus,
    }];
  }

  if (onComics) {
    const sections = [
      {
        id: 'read-state',
        label: 'Reading',
        kind: 'radio',
        currentValue: state.currentComicStatus,
        defaultValue: 'all',
        options: [
          { value: 'all',        label: 'All' },
          { value: 'reading',    label: 'In progress' },
          { value: 'caught-up',  label: 'Caught up' },
          { value: 'unread',     label: 'Unread' },
        ],
        setter: _setMediaStatus, // routes to comicStatus internally when on comics
      },
      {
        id: 'series-status',
        label: 'Series',
        kind: 'radio',
        currentValue: state.currentComicCompletion || '',
        defaultValue: '',
        options: [
          { value: '',           label: 'All series' },
          { value: 'ongoing',    label: 'Ongoing' },
          { value: 'completed',  label: 'Completed' },
          { value: 'hiatus',     label: 'Hiatus' },
        ],
        setter: _setComicCompletion,
      },
      {
        id: 'genre',
        label: 'Genre',
        kind: 'list',
        currentValue: state.currentComicGenre || '',
        defaultValue: '',
        options: _genreOptions(_comicGenresCache),
        setter: _setComicGenre,
        emptyMessage: 'No genre data yet',
      },
    ];
    return sections;
  }

  if (onVideo) {
    // Shows / Movies / Music Videos. Watch state + Genre + Year span
    // every video chip; Status (ongoing/ended) and Rating are deferred
    // until the row metadata carries those fields.
    return [
      {
        id: 'video-watch-state',
        label: 'Watch state',
        kind: 'radio',
        currentValue: state.currentVideoStatus,
        defaultValue: 'all',
        options: [
          { value: 'all',          label: 'All' },
          { value: 'in_progress',  label: 'Watching' },
          { value: 'finished',     label: 'Watched' },
          { value: 'not_started',  label: 'Unwatched' },
        ],
        setter: _setVideoStatus,
      },
      {
        id: 'video-genre',
        label: 'Genre',
        kind: 'list',
        currentValue: state.currentVideoGenre || '',
        defaultValue: '',
        options: _genreOptions(_videoGenresCache),
        setter: _setVideoGenre,
        emptyMessage: 'No genre data yet',
      },
      {
        id: 'video-year',
        label: 'Year',
        kind: 'range',
        currentFrom: state.currentVideoYearFrom || 0,
        currentTo:   state.currentVideoYearTo   || 0,
        defaultFrom: 0,
        defaultTo:   0,
        minBound: _YEAR_MIN_BOUND,
        maxBound: _YEAR_MAX_BOUND,
        setter: _setVideoYearRange,
      },
    ];
  }

  return [];
}

function _genreOptions(cache) {
  // Shared shape used by both the comics and video genre sections.
  // First option is the reset ("Any genre"); each subsequent option
  // carries its row count for ranking visibility.
  const opts = [{ value: '', label: 'Any genre' }];
  for (const g of (cache || [])) {
    opts.push({ value: g.key, label: g.label, count: g.count });
  }
  return opts;
}

function _sectionIsActive(section) {
  // Section is "active" when its current value differs from default.
  // Range sections check both bounds; radio/list check the single value.
  if (section.kind === 'range') {
    return section.currentFrom !== section.defaultFrom
        || section.currentTo   !== section.defaultTo;
  }
  return section.currentValue !== section.defaultValue;
}

function _activeFilterCount() {
  const schema = _filterSchemaForChip();
  let n = 0;
  for (const section of schema) {
    if (_sectionIsActive(section)) n++;
  }
  return n;
}

function _sectionSummary(section) {
  // The collapsed-row summary that lives next to the section label —
  // shows the user what's currently selected without having to expand
  // the section. Default state reads "All" / "Any" / "Any year" so
  // the row never appears blank.
  if (section.kind === 'range') {
    const { currentFrom, currentTo, minBound, maxBound } = section;
    if (!currentFrom && !currentTo)        return 'Any year';
    if (currentFrom && currentTo)          return `${currentFrom} — ${currentTo}`;
    if (currentFrom)                        return `${currentFrom} — ${maxBound}`;
    return `${minBound} — ${currentTo}`;
  }
  // radio + list: find the matching option's label, fall back to
  // the raw value or "Any" for empty defaults.
  const match = (section.options || []).find(o => o.value === section.currentValue);
  if (match) return match.label;
  if (!section.currentValue) {
    return section.kind === 'list' ? 'Any' : 'All';
  }
  return String(section.currentValue);
}

function _renderFilterButton() {
  const btn = state.el.filterBtn;
  if (!btn) return;
  const schema = _filterSchemaForChip();
  const hasFilters = schema.length > 0;
  btn.classList.toggle('hidden', !hasFilters);
  if (!hasFilters) {
    _closeFilterDropdown();
    return;
  }
  const count = _activeFilterCount();
  btn.classList.toggle('has-filters', count > 0);
  const countEl = btn.querySelector('[data-filter-count]');
  if (countEl) countEl.textContent = count > 0 ? String(count) : '';
}

function _renderFilterDropdown() {
  const dd = state.el.filterDropdown;
  if (!dd) return;
  const schema = _filterSchemaForChip();
  if (!schema.length) { dd.innerHTML = ''; return; }

  // Auto-expand any section that has an active filter on first render
  // after a chip switch, so users can see the active state directly
  // without having to remember to expand. They can still collapse
  // manually; the explicit collapse is preserved (we only ADD to
  // _filterExpanded here, never remove).
  for (const section of schema) {
    if (_sectionIsActive(section)) _filterExpanded.add(section.id);
  }

  const parts = [];
  for (const section of schema) {
    const expanded = _filterExpanded.has(section.id);
    const summary = escapeAttr(_sectionSummary(section));
    const sectionActive = _sectionIsActive(section);
    parts.push(`
      <div class="files-filter-section${expanded ? ' is-expanded' : ''}${sectionActive ? ' is-active' : ''}"
           data-section="${escapeAttr(section.id)}">
        <button type="button" class="files-filter-section-header"
                data-action="toggle-section"
                data-section="${escapeAttr(section.id)}"
                aria-expanded="${expanded ? 'true' : 'false'}">
          <span class="files-filter-section-label">${escapeAttr(section.label)}</span>
          <span class="files-filter-section-summary">${summary}</span>
          <svg class="files-filter-section-chev" width="11" height="11"
               viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"
               aria-hidden="true">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </button>
        <div class="files-filter-section-body" role="region">
    `);
    parts.push(_renderSectionBody(section));
    parts.push(`</div></div>`);
  }

  // Footer reset, only shown if any filter is non-default.
  if (_activeFilterCount() > 0) {
    parts.push(
      `<div class="files-filter-footer">
        <button type="button" class="files-filter-reset" data-action="reset-filters">Reset filters</button>
      </div>`,
    );
  }

  dd.innerHTML = parts.join('');

  // Range sections have post-render hookup since their values come from
  // input ranges (not data-attributes). Wire each one once after the
  // body is in the DOM.
  for (const section of schema) {
    if (section.kind === 'range') {
      _wireRangeSection(dd, section);
    }
  }
}

function _renderSectionBody(section) {
  if (section.kind === 'range') {
    const { currentFrom, currentTo, minBound, maxBound } = section;
    // Slider thumbs default to bounds when value is 0 (unset). Display
    // labels track live during drag for tactile feedback.
    const fromVal = currentFrom || minBound;
    const toVal   = currentTo   || maxBound;
    return `
      <div class="files-filter-range" data-section="${escapeAttr(section.id)}">
        <div class="files-filter-range-display" aria-hidden="true">
          <span class="files-filter-range-from">${fromVal}</span>
          <span class="files-filter-range-sep">—</span>
          <span class="files-filter-range-to">${toVal}</span>
        </div>
        <div class="files-filter-range-track">
          <div class="files-filter-range-fill"></div>
          <input type="range" class="files-filter-range-input files-filter-range-input-from"
                 min="${minBound}" max="${maxBound}" step="1"
                 value="${fromVal}"
                 aria-label="Year from">
          <input type="range" class="files-filter-range-input files-filter-range-input-to"
                 min="${minBound}" max="${maxBound}" step="1"
                 value="${toVal}"
                 aria-label="Year to">
        </div>
        <div class="files-filter-range-bounds" aria-hidden="true">
          <span>${minBound}</span><span>${maxBound}</span>
        </div>
        <button type="button" class="files-filter-range-reset"
                data-action="reset-range"
                data-section="${escapeAttr(section.id)}">
          Any year
        </button>
      </div>
    `;
  }
  if (section.kind === 'list') {
    const { options, currentValue, emptyMessage } = section;
    if (!options.length || (options.length === 1 && !options[0].value)) {
      return `<div class="files-filter-empty">${escapeAttr(emptyMessage || 'No options yet')}</div>`;
    }
    const items = options.map(opt => {
      const isActive = opt.value === currentValue;
      const countHtml = opt.count != null
        ? `<span class="files-filter-item-count">${opt.count}</span>` : '';
      return `<button type="button" role="menuitemradio"
                       class="files-filter-item${isActive ? ' is-active' : ''}"
                       aria-checked="${isActive ? 'true' : 'false'}"
                       data-section="${escapeAttr(section.id)}"
                       data-value="${escapeAttr(opt.value)}">
                <span class="files-filter-item-label">${escapeAttr(opt.label)}</span>
                ${countHtml}
              </button>`;
    });
    return `<div class="files-filter-list">${items.join('')}</div>`;
  }
  // Default: radio.
  return section.options.map(opt => {
    const isActive = opt.value === section.currentValue;
    return `<button type="button" role="menuitemradio"
                     class="files-filter-item${isActive ? ' is-active' : ''}"
                     aria-checked="${isActive ? 'true' : 'false'}"
                     data-section="${escapeAttr(section.id)}"
                     data-value="${escapeAttr(opt.value)}">
              <span class="files-filter-item-label">${escapeAttr(opt.label)}</span>
            </button>`;
  }).join('');
}

/**
 * Two-thumb range slider via two overlapping <input type="range">.
 * The native inputs handle keyboard + a11y for free; CSS layers them
 * so visually they share a single track. Live labels follow the
 * thumbs during drag; the actual filter fires on `change` (release)
 * to avoid spamming the backend mid-drag.
 */
function _wireRangeSection(rootEl, section) {
  const wrap = rootEl.querySelector(
    `.files-filter-range[data-section="${section.id}"]`,
  );
  if (!wrap) return;
  const fromInput = wrap.querySelector('.files-filter-range-input-from');
  const toInput   = wrap.querySelector('.files-filter-range-input-to');
  const fromLabel = wrap.querySelector('.files-filter-range-from');
  const toLabel   = wrap.querySelector('.files-filter-range-to');
  const fill      = wrap.querySelector('.files-filter-range-fill');
  if (!fromInput || !toInput) return;

  const min = section.minBound;
  const max = section.maxBound;

  const updateVisuals = () => {
    const f = Math.min(Number(fromInput.value), Number(toInput.value));
    const t = Math.max(Number(fromInput.value), Number(toInput.value));
    fromLabel.textContent = String(f);
    toLabel.textContent = String(t);
    if (fill) {
      const pctFrom = ((f - min) / (max - min)) * 100;
      const pctTo   = ((t - min) / (max - min)) * 100;
      fill.style.left  = `${pctFrom}%`;
      fill.style.right = `${100 - pctTo}%`;
    }
  };

  // Keep thumbs from crossing — the lower input can't exceed the upper,
  // and vice versa. Without this constraint, dragging the "from" past
  // "to" would end up sending year_from > year_to to the backend.
  const clampOnInput = (e) => {
    const f = Number(fromInput.value);
    const t = Number(toInput.value);
    if (f > t) {
      if (e.target === fromInput) fromInput.value = String(t);
      else                         toInput.value   = String(f);
    }
    updateVisuals();
  };

  fromInput.addEventListener('input', clampOnInput);
  toInput.addEventListener('input',   clampOnInput);

  // Fire the actual filter on `change` (release), not `input` (drag).
  // change-only avoids re-querying the backend dozens of times during
  // a single drag gesture.
  const commit = () => {
    const f = Math.min(Number(fromInput.value), Number(toInput.value));
    const t = Math.max(Number(fromInput.value), Number(toInput.value));
    // Map full-range to "no filter" (0,0) so the filter clears when
    // the user drags both thumbs to the bounds.
    const fromOut = (f === min) ? 0 : f;
    const toOut   = (t === max) ? 0 : t;
    section.setter(fromOut, toOut);
  };
  fromInput.addEventListener('change', commit);
  toInput.addEventListener('change',   commit);

  // Initial visual sync so the fill bar matches the thumbs on first
  // render (CSS percentages need JS-derived starting values).
  updateVisuals();
}

function _toggleSectionExpanded(sectionId) {
  if (_filterExpanded.has(sectionId)) _filterExpanded.delete(sectionId);
  else                                _filterExpanded.add(sectionId);
  // Re-render to flip the expanded class and update aria-expanded.
  // No state setter fires here — pure UI, no backend call.
  _renderFilterDropdown();
}

function _openFilterDropdown() {
  const dd = state.el.filterDropdown;
  const btn = state.el.filterBtn;
  if (!dd || !btn) return;
  _renderFilterDropdown();
  dd.classList.add('open');
  btn.setAttribute('aria-expanded', 'true');
}

function _closeFilterDropdown() {
  const dd = state.el.filterDropdown;
  const btn = state.el.filterBtn;
  if (!dd) return;
  dd.classList.remove('open');
  btn?.setAttribute('aria-expanded', 'false');
}

function _toggleFilterDropdown() {
  state.el.filterDropdown?.classList.contains('open')
    ? _closeFilterDropdown()
    : _openFilterDropdown();
}

function _applyFilterSelection(sectionId, value) {
  const schema = _filterSchemaForChip();
  const section = schema.find(s => s.id === sectionId);
  if (!section || !section.setter) return;
  section.setter(value);
  // Re-render the dropdown body so the active-state checkmark updates
  // without closing the popover. The button badge updates via
  // _renderFilterButton in setSource / setX paths.
  _renderFilterDropdown();
  _renderFilterButton();
}

function _resetAllFilters() {
  const schema = _filterSchemaForChip();
  for (const section of schema) {
    if (section.currentValue !== section.defaultValue) {
      section.setter(section.defaultValue);
    }
  }
  _renderFilterDropdown();
  _renderFilterButton();
}


function _renderMediaViewBar() {
  const bar = state.el.mediaViewBar;
  if (!bar) return;
  const show = _supportsCatalogSource(state.currentSource);
  bar.classList.toggle('hidden', !show);
  if (!show) return;
  bar.querySelectorAll('.files-media-view-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mediaView === state.currentMediaView);
  });
}

function _renderSortOptions() {
  const dropdown = state.el.sortDropdown;
  if (!dropdown) return;
  const showAuthor = _isMediaSource(state.currentSource);
  const showProgress = _supportsProgressSort();
  const showComics = state.currentSource === COMICS_CHIP;
  dropdown.querySelectorAll('.files-sort-option').forEach(opt => {
    const slug = opt.dataset.sort;
    const isAuthorOpt = slug === 'author';
    const isProgressOpt = slug === 'progress';
    const isComicsOpt = COMICS_SORTS.has(slug);
    const hidden = (
      (isAuthorOpt && !showAuthor)
      || (isProgressOpt && !showProgress)
      || (isComicsOpt && !showComics)
    );
    opt.classList.toggle('hidden', hidden);
  });
  // If the current sort is a gated option whose chip we just left, reset
  // to 'newest' so the grid doesn't stay sorted by a now-hidden field.
  // We keep the pref in localStorage so returning to the original chip
  // restores the prior sort.
  const stale =
    (state.currentSort === 'author' && !showAuthor) ||
    (state.currentSort === 'progress' && !showProgress) ||
    (COMICS_SORTS.has(state.currentSort) && !showComics);
  if (stale) {
    state.currentSort = 'newest';
    const labelEl = state.el.sortBtn?.querySelector('.files-sort-label');
    if (labelEl) labelEl.textContent = SORT_LABELS.newest;
    state.el.sortDropdown?.querySelectorAll('.files-sort-option').forEach(opt => {
      opt.classList.toggle('active', opt.dataset.sort === 'newest');
    });
    return;
  }
  // Symmetric to the exit path above: when entering Comics with a generic
  // default like 'newest'/'oldest' (which sort by sync activity, not by
  // when the user actually read), promote to 'continue' so the chip lands
  // on the sort that matches the chip's mental model. Explicit picks
  // ('name', 'unread', 'updated') stay untouched. The promoted choice
  // persists, so a user who later picks something else is respected.
  if (showComics && GENERIC_SORTS_FOR_COMICS_PROMOTION.has(state.currentSort)) {
    state.currentSort = 'continue';
    savePref('sort', 'continue');
    const labelEl = state.el.sortBtn?.querySelector('.files-sort-label');
    if (labelEl) labelEl.textContent = SORT_LABELS.continue;
    state.el.sortDropdown?.querySelectorAll('.files-sort-option').forEach(opt => {
      opt.classList.toggle('active', opt.dataset.sort === 'continue');
    });
  }
}

function _setMediaStatus(status) {
  // One click handler, two dialects — route to the comic status path
  // when the Comics chip is active so filter values live in the right
  // slot of state. Otherwise treat as audiobook playback status.
  if (state.currentSource === COMICS_CHIP) {
    state.currentComicStatus = status;
    savePref('comicStatus', status);
  } else {
    state.currentMediaStatus = status;
    savePref('status', status);
  }
  _renderFilterButton();
  deselectAll();
  loadFiles({ reset: true });
}

function _setComicCompletion(completion) {
  state.currentComicCompletion = completion || '';
  savePref('comicCompletion', state.currentComicCompletion);
  _renderFilterButton();
  deselectAll();
  loadFiles({ reset: true });
}

function _setComicGenre(genre) {
  state.currentComicGenre = (genre || '').trim();
  savePref('comicGenre', state.currentComicGenre);
  _renderFilterButton();
  deselectAll();
  loadFiles({ reset: true });
}

// --- Video filter setters ------------------------------------------------
// Watch state, genre, and year-range filters for Shows/Movies/Music
// Videos. Same shape as the comic setters; each writes the new value
// into the shared `state` slot, persists to localStorage, refreshes the
// Filter button badge, deselects, and re-runs the list query.

function _setVideoStatus(status) {
  state.currentVideoStatus = status || 'all';
  savePref('videoStatus', state.currentVideoStatus);
  _renderFilterButton();
  deselectAll();
  loadFiles({ reset: true });
}

function _setVideoGenre(genre) {
  state.currentVideoGenre = (genre || '').trim();
  savePref('videoGenre', state.currentVideoGenre);
  _renderFilterButton();
  deselectAll();
  loadFiles({ reset: true });
}

function _setVideoYearRange(yearFrom, yearTo) {
  state.currentVideoYearFrom = Number(yearFrom) || 0;
  state.currentVideoYearTo   = Number(yearTo)   || 0;
  savePref('videoYearFrom', String(state.currentVideoYearFrom));
  savePref('videoYearTo',   String(state.currentVideoYearTo));
  _renderFilterButton();
  deselectAll();
  loadFiles({ reset: true });
}

async function _setMediaView(view) {
  if (view === state.currentMediaView) {
    // Re-opening Catalog from itself still launches the overlay — lets the
    // user relaunch after closing it without switching chips first.
    if (view === 'catalog') await _openCatalog();
    return;
  }
  state.currentMediaView = view;
  savePref('mediaView', view);
  _renderMediaViewBar();
  _renderFilterButton();
  if (view === 'catalog') {
    await _openCatalog();
  } else {
    // Leaving Catalog: the overlay's own close handler already hides it;
    // refresh Library so freshly-pinned rows appear at the top.
    deselectAll();
    loadFiles({ reset: true });
  }
}

async function _openCatalog() {
  try {
    const mod = await import('../librivox-browse.js');
    mod.openLibrivoxBrowse();
  } catch (err) {
    console.warn('[files] catalog open failed:', err);
  }
}

function _onSearchInput() {
  clearTimeout(state.searchDebounce);
  state.searchDebounce = setTimeout(() => {
    deselectAll();
    loadFiles({ reset: true });
  }, 300);
}

function _toggleView() {
  // The Comics chip runs its own renderer (renderComicsGrid) and
  // doesn't react to grid/list/gallery view modes. Calling renderGrid()
  // here would push it through the generic empty-state path and flash
  // the "no connected services" CTA, since state.files is empty for
  // comics (its content lives in comics.js's _view.chapterCache /
  // seriesGrid). Save the pref so it applies to non-comics chips
  // later, but skip the re-render — the button is also hidden on the
  // comics chip via _setSource so the click usually never lands here.
  state.currentView = state.currentView === 'grid' ? 'list'
    : state.currentView === 'list' ? 'gallery'
    : 'grid';
  savePref('view', state.currentView);
  if (isComicsChipActive()) return;
  renderGrid();
}

function _toggleSortDropdown() { state.el.sortDropdown?.classList.toggle('open'); }
function _closeSortDropdown()  { state.el.sortDropdown?.classList.remove('open'); }

function _setSort(sort) {
  state.currentSort = sort;
  savePref('sort', sort);
  const labelEl = state.el.sortBtn?.querySelector('.files-sort-label');
  if (labelEl) labelEl.textContent = SORT_LABELS[sort] || 'Newest';
  state.el.sortDropdown?.querySelectorAll('.files-sort-option').forEach(opt => {
    opt.classList.toggle('active', opt.dataset.sort === sort);
  });
  _closeSortDropdown();
  loadFiles({ reset: true });
}

function _scrollToFocused() {
  const card = state.el.grid?.querySelector(`.files-card[data-index="${state.focusedIndex}"]`);
  card?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

// --- Keyboard handler ------------------------------------------------

function _handleKeydown(e) {
  if (!state.isOpen) return;
  if (state.galleryOverlay) return;
  if (state.renamingId) return;
  const typing = isTextInput(document.activeElement);
  const len = state.files.length;
  if (!len && e.key !== 'Escape') return;

  switch (e.key) {
    case 'ArrowDown':
    case 'ArrowRight': {
      if (typing) return;
      e.preventDefault();
      state.focusedIndex = Math.min(state.focusedIndex + 1, len - 1);
      selectOnly(state.files[state.focusedIndex]?.id);
      state.lastClickedIndex = state.focusedIndex;
      _scrollToFocused();
      break;
    }
    case 'ArrowUp':
    case 'ArrowLeft': {
      if (typing) return;
      e.preventDefault();
      state.focusedIndex = Math.max(state.focusedIndex - 1, 0);
      selectOnly(state.files[state.focusedIndex]?.id);
      state.lastClickedIndex = state.focusedIndex;
      _scrollToFocused();
      break;
    }
    case 'Enter': {
      if (typing) return;
      if (state.focusedIndex >= 0 && state.files[state.focusedIndex]) {
        e.preventDefault();
        activateFile(state.files[state.focusedIndex]);
      }
      break;
    }
    case 'Delete':
    case 'Backspace': {
      if (typing) return;
      if (state.selection.size > 0) {
        e.preventDefault();
        if (state.selection.size === 1) deleteFileAction([...state.selection][0]);
        else bulkDelete();
      }
      break;
    }
    case 'F2': {
      if (typing) return;
      if (state.selection.size === 1) {
        e.preventDefault();
        startRename([...state.selection][0]);
      }
      break;
    }
    case ' ': {
      if (typing) return;
      e.preventDefault();
      if (state.selection.size === 1 && state.el.detail?.classList.contains('hidden')) {
        updateDetail();
      } else if (!state.el.detail?.classList.contains('hidden')) {
        state.el.detail?.classList.add('hidden');
      }
      break;
    }
    case 'a':
    case 'A': {
      if (typing) return;
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        selectAll();
      }
      break;
    }
    case 'Escape': {
      if (state.selectMode) exitSelectMode();
      else if (state.selection.size > 0) deselectAll();
      else closeFiles();
      break;
    }
  }
}

// --- "Back to where I came from" affordance --------------------------
//
// The back button surfaces only when the panel was opened from another
// overlay (Discovery card click, mini-player author link, etc.). It
// gives the user a one-tap way to return to whatever they were doing
// without hunting for the right toggle button. The button is intentionally
// distinct from the close button: close means "I'm done here," back
// means "drop me where I was." Both close Files; only back also opens
// the prior overlay.

const _BACK_TARGETS = {
  // Map cameFromOverlay key → human-readable label + opener function.
  // Each opener uses the canonical toggle button so the prior overlay's
  // own state-restore logic kicks in (Discovery's view-toggle pref,
  // last-used tab, etc.). No coupling to the prior module's internals.
  browse: {
    label: 'Back to Browse',
    open: () => document.getElementById('toggle-browse-btn')?.click(),
  },
  // Library / Studio could surface here later by adding entries; the
  // listener that sets cameFromOverlay is the only thing that needs to
  // change when adding a new origin overlay.
};

function _updateBackButton() {
  const btn = state.el.backBtn;
  if (!btn) return;
  const target = _BACK_TARGETS[state.cameFromOverlay];
  if (target) {
    btn.classList.remove('hidden');
    btn.setAttribute('title', target.label);
    btn.setAttribute('aria-label', target.label);
    if (state.el.backBtnLabel) state.el.backBtnLabel.textContent = target.label;
  } else {
    btn.classList.add('hidden');
  }
}

function _setCameFrom(originKey) {
  state.cameFromOverlay = originKey || '';
  _updateBackButton();
}

function _navigateBack() {
  const target = _BACK_TARGETS[state.cameFromOverlay];
  state.cameFromOverlay = '';
  _updateBackButton();
  closeFiles();
  // Open the prior overlay after closeFiles' transition kicks off.
  // Same 50ms grace as the rest of the panel switching (close starts
  // a fade; we don't want the prior overlay to fight for the same
  // compositor layer mid-transition).
  if (target) setTimeout(target.open, 60);
}

// --- Event wiring ----------------------------------------------------

function _wireEvents() {
  state.el.closeBtn?.addEventListener('click', closeFiles);
  state.el.backBtn?.addEventListener('click', _navigateBack);
  state.el.viewToggle?.addEventListener('click', _toggleView);
  state.el.search?.addEventListener('input', _onSearchInput);

  // Audiobooks view toggle — Library (local pins) vs Catalog (live browse).
  // Catalog launches the LibriVox overlay on demand; Library is the default
  // listing view.
  state.el.mediaViewBar?.addEventListener('click', (e) => {
    const btn = e.target.closest('.files-media-view-btn');
    if (btn?.dataset.mediaView) _setMediaView(btn.dataset.mediaView);
  });

  // Unified Filter dropdown — click button toggles popover; click an
  // item commits the section's value via the schema's setter. Both
  // playback-status (audiobooks) and read-state/series-status/genre
  // (comics) flow through the same surface here.
  state.el.filterBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggleFilterDropdown();
  });
  state.el.filterDropdown?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (e.target.closest('[data-action="reset-filters"]')) {
      _resetAllFilters();
      return;
    }
    // Section header click → toggle expand. Routed before item-select
    // so the chevron + label area react as expected even though the
    // inner spans bubble up to the button.
    const header = e.target.closest('[data-action="toggle-section"]');
    if (header) {
      _toggleSectionExpanded(header.dataset.section);
      return;
    }
    // Range reset (per-section "Any year" affordance).
    const rangeReset = e.target.closest('[data-action="reset-range"]');
    if (rangeReset) {
      const sectionId = rangeReset.dataset.section;
      const schema = _filterSchemaForChip();
      const section = schema.find(s => s.id === sectionId);
      if (section?.kind === 'range' && section.setter) {
        section.setter(section.defaultFrom, section.defaultTo);
      }
      return;
    }
    // Filter item (radio / list option) — commit and re-render.
    const item = e.target.closest('[data-section][data-value]');
    if (!item) return;
    _applyFilterSelection(item.dataset.section, item.dataset.value);
  });

  // Re-fetch genres when upstream sync adds new series.
  window.addEventListener('media-servers:changed', () => {
    if (state.currentSource === COMICS_CHIP) _refreshComicGenres();
    if (VIDEO_CLOUD_CHIPS.has(state.currentSource)) _refreshVideoGenres();
  });

  // When the catalog overlay closes, revert the toggle + refresh so newly
  // pinned books appear without the user having to manually flip back.
  window.addEventListener('librivox-browse:closed', () => {
    if (state.currentMediaView !== 'catalog') return;
    state.currentMediaView = 'library';
    savePref('mediaView', 'library');
    _renderMediaViewBar();
    _renderFilterButton();
    if (_isMediaSource(state.currentSource)) {
      loadFiles({ reset: true });
    }
  });

  // Media Servers panel — dynamic import keeps startup lean for users
  // who never touch this feature.
  state.el.mediaBtn?.addEventListener('click', async () => {
    const mod = await import('../media-servers.js');
    mod.openMediaServers();
  });
  window.addEventListener('media-servers:changed', () => {
    if (state.isOpen) {
      loadFiles({ reset: true });
      loadStats();
    }
  });

  // App-level media players (audio + video) push progress events whenever
  // a chunk of playback lands upstream. Patch any loaded row in place so
  // the Files surface reflects watch/listen state without a full reload.
  window.addEventListener('media-player:progress', (e) => {
    const { fileId, progressPct } = e.detail || {};
    if (!fileId || progressPct == null) return;
    // Reflect the update inside the continue rail. patchRailProgress
    // resizes the bar in place when the row is still in_progress, and
    // removes the card entirely when the event signals the row left
    // the bucket (mark-watched → isFinished=true, reset → pct=0).
    patchRailProgress(fileId, progressPct, {
      isFinished: e.detail.isFinished === true,
    });
    const row = state.files.find(f => f.id === fileId);
    const detailRow = state.detailOverrideFile?.id === fileId ? state.detailOverrideFile : null;
    const nextMeta = {
      progress_pct: progressPct,
      current_time_s: e.detail.currentTimeS ?? row?.source_metadata?.current_time_s ?? detailRow?.source_metadata?.current_time_s ?? 0,
      duration_s: e.detail.durationS ?? row?.source_metadata?.duration_s ?? detailRow?.source_metadata?.duration_s ?? 0,
      is_finished: e.detail.isFinished ?? row?.source_metadata?.is_finished ?? detailRow?.source_metadata?.is_finished ?? false,
    };
    if (row) {
      row.source_metadata = {
        ...(row.source_metadata || {}),
        ...nextMeta,
      };
    }
    if (detailRow && detailRow !== row) {
      detailRow.source_metadata = {
        ...(detailRow.source_metadata || {}),
        ...nextMeta,
      };
    }
    const card = state.el.grid?.querySelector(`.files-card[data-id="${CSS.escape(fileId)}"]`);
    if (card) {
      let bar = card.querySelector('.files-card-progress span');
      if (!bar) {
        const wrap = document.createElement('div');
        wrap.className = 'files-card-progress';
        wrap.setAttribute('aria-hidden', 'true');
        bar = document.createElement('span');
        wrap.appendChild(bar);
        card.appendChild(wrap);
      }
      bar.style.width = `${Number(progressPct).toFixed(1)}%`;
    }
  });

  // Quick-access affordances on Discovery cards / mini player.
  // Detail shape: { chip?, search?, fileId? } — any subset is valid.
  //   chip   → switch the Files panel to that virtual chip (audiobooks,
  //            comics, movies, shows, podcasts). Implies cloud scope
  //            since all four chips are cloud-backed in the file_index.
  //   search → seed the search input and reload. Used for "more from
  //            this author" / "more in this series" navigation.
  //   fileId → after the load settles, scroll the matching row into
  //            view + select it (best-effort; no-op if not in page).
  // The intent here is the sibling of media-player:expand below, but
  // generalised so resume-listening cards, library cards, and the
  // mini-player author link can all dispatch through one path.
  //
  // Order matters: the search input must be SEEDED before openFiles()
  // and _setSource() run, because both of those call loadFiles({ reset:
  // true }) internally, and loadFiles reads the search input value at
  // call time (api.js:368). Setting the value after openFiles() means
  // the first loadFiles fetches with empty search, then races with the
  // explicit-search loadFiles below — a flaky outcome that surfaced as
  // the search bar appearing empty + grid showing all-audiobooks instead
  // of author-filtered results.
  window.addEventListener('files:open-with-filter', async (e) => {
    const detail = (e.detail || {});
    const chip = (detail.chip || '').trim();
    const search = (detail.search || '').toString().slice(0, 200);
    const fileId = (detail.fileId || '').toString();
    console.info('[files] open-with-filter', { chip, search, fileId, hasSearchEl: !!state.el.search });

    // Seed the search input FIRST so any subsequent loadFiles() reads
    // the right value. Dispatching an 'input' event lets debounced
    // search listeners observe the change, and matters for visual
    // libraries that reflect input value in CSS pseudo-classes.
    if (search && state.el.search) {
      state.el.search.value = search;
      state.el.search.dispatchEvent(new Event('input', { bubbles: true }));
      // Cancel the 300ms debounce that the input event just kicked off
      // — we'll trigger loadFiles ourselves below at the right point.
      clearTimeout(state.searchDebounce);
      state.searchDebounce = null;
    } else if (!search && state.el.search && state.el.search.value) {
      // No-search dispatch (chip-only navigation) — clear stale text
      // so the chip switch shows the full chip, not a leftover filter.
      state.el.search.value = '';
    }

    // Capture which overlay was visible before we close it — drives the
    // "Back to Browse" affordance below the close button so the user
    // can one-tap return to where they started. Read BEFORE
    // dismissOverlays since the close-cycle removes the visibility
    // class we'd be sniffing.
    const browsePanelEl = document.getElementById('browse-panel');
    const cameFromBrowse = browsePanelEl && !browsePanelEl.classList.contains('hidden');

    // Close other overlays so the Files panel comes to the front. Without
    // this, a click from inside the Browse panel (Discovery library card,
    // resume-listening author link, etc.) opens Files behind the still-
    // visible Browse overlay — invisible to the user. dismissOverlays
    // honors except='files' so Files itself isn't closed.
    window.dispatchEvent(new CustomEvent('augmentum:dismiss-overlays', {
      detail: { except: 'files' },
    }));

    if (!state.isOpen) openFiles({ focusSearch: false });
    if (state.selectMode) exitSelectMode();

    _setCameFrom(cameFromBrowse ? 'browse' : '');

    if (chip) {
      // All library chips live under cloud scope — switching scope
      // first ensures _setSource doesn't render an empty grid because
      // the local-scope query rejects the chip. Skip if already on
      // the right chip to avoid re-running loadFiles unnecessarily.
      if (state.currentScope !== 'cloud') _setScope('cloud');
      if (state.currentSource !== chip) {
        // _setSource ends with loadFiles({reset:true}) — that single
        // call hits the server with the seeded search + new chip.
        _setSource(chip);
      } else if (search) {
        // Already on the right chip; just refresh with the new search.
        deselectAll();
        loadFiles({ reset: true });
      }
    } else if (search) {
      // No chip change at all; just refresh the existing chip with the
      // seeded search.
      deselectAll();
      loadFiles({ reset: true });
    }

    if (fileId) {
      // Kick off the entry fetch in parallel with the chip-switch +
      // loadFiles so we don't add a serial round-trip before the panel
      // can render. By the time the timeout below fires, either:
      //   - the row landed in state.files (preferred — selectOnly +
      //     scroll lights up the detail panel via the standard list
      //     selection flow), or
      //   - it didn't (movies/shows libraries can be 1000+ entries; the
      //     clicked row is rarely in page 1 of an alphabetised cloud
      //     listing), in which case we fall through to the
      //     detailOverrideFile path: stuff the fetched entry into
      //     state.detailOverrideFile and call updateSelectionUI() so
      //     the detail panel renders directly from the fetched data.
      // Mirrors the video-player:expand pattern below, since both
      // surfaces share the same "click a card → open detail" intent.
      const entryPromise = fetchFileEntry(fileId);
      setTimeout(async () => {
        const row = state.files.find(f => f.id === fileId);
        if (row) {
          selectOnly(fileId);
          const card = state.el.grid?.querySelector(`.files-card[data-id="${CSS.escape(fileId)}"]`);
          card?.scrollIntoView({ block: 'center', behavior: 'smooth' });
          return;
        }
        // Not in the current page — render detail from the fetched entry
        // so the user sees the movie's info instead of just landing on
        // the chip's grid.
        try {
          const entry = await entryPromise;
          if (!entry) return;
          state.selection.clear();
          state.selection.add(fileId);
          state.detailOverrideFile = entry;
          updateSelectionUI();
        } catch (err) {
          console.warn('[files] fetchFileEntry failed for fileId', fileId, err);
        }
      }, 200);
    }
  });

  // Mini-player "Open" button wants to surface the book's detail view.
  // Open the Files panel scoped to its source and select the row if it's
  // already loaded. If the row isn't in the current page, we let the
  // user scroll — a deep-link helper lands with the Library panel.
  window.addEventListener('media-player:expand', async (e) => {
    const { fileId } = e.detail || {};
    if (!fileId) return;
    if (!state.isOpen) openFiles({ focusSearch: false });
    const entry = await fetchFileEntry(fileId);
    const entityKind = String(entry?.source_metadata?.entity_kind || '').toLowerCase();
    const targetChip = entityKind === 'podcast' ? PODCASTS_CHIP : MEDIA_CHIP;
    // Media-player expand: land the user on the relevant cloud-audio chip
    // so the playing row is visible in context.
    if (state.currentScope !== 'cloud') _setScope('cloud');
    if (state.currentSource !== targetChip) {
      _setSource(targetChip);
    }
    // Give the grid a tick to finish rendering, then try to focus the
    // row. If it's not in the current page we'll fall back to just
    // highlighting the source chip; users can scroll to find it.
    setTimeout(() => {
      const row = state.files.find(f => f.id === fileId);
      if (row) {
        selectOnly(fileId);
        const card = state.el.grid?.querySelector(`.files-card[data-id="${CSS.escape(fileId)}"]`);
        card?.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    }, 120);
  });

  // Resume-toast deep link: open a video file and auto-start playback.
  // Dispatched by media-resume.js on the next page load when the user
  // taps "Play" on the "Resume X?" toast. Same shape as
  // video-player:expand but with autoplay intent.
  window.addEventListener('files:open-and-play', async (e) => {
    const { fileId } = e.detail || {};
    if (!fileId) return;
    const entry = await fetchFileEntry(fileId);
    if (!entry) return;
    if (!state.isOpen) openFiles({ focusSearch: false });
    if (state.selectMode) exitSelectMode();
    if (state.currentScope !== 'cloud') _setScope('cloud');
    const targetChip = _videoChipForFile(entry);
    if (targetChip !== 'all' && state.currentSource !== targetChip) {
      _setSource(targetChip);
    }
    // Open the floating video player — this is the same call the
    // "Play" button in the row would make, so it picks up the file's
    // server-side resume position automatically.
    try {
      const preview = await import('./preview.js');
      if (preview.openVideoPreviewById) {
        await preview.openVideoPreviewById(fileId);
      }
    } catch (err) {
      console.warn('[files] resume-toast play failed:', err);
    }
  });

  // Floating video can surface the currently playing item back into Files.
  // If the row isn't in the current page, land on the correct cloud chip
  // and open the detail panel from a fetched entry instead of making the
  // user hunt for it manually.
  window.addEventListener('video-player:expand', async (e) => {
    const { fileId } = e.detail || {};
    if (!fileId) return;
    const entry = await fetchFileEntry(fileId);
    if (!entry) return;
    if (!state.isOpen) openFiles({ focusSearch: false });
    if (state.selectMode) exitSelectMode();
    if (state.currentScope !== 'cloud') _setScope('cloud');
    const targetChip = _videoChipForFile(entry);
    if (targetChip !== 'all' && state.currentSource !== targetChip) {
      _setSource(targetChip);
    }
    setTimeout(() => {
      const row = state.files.find(f => f.id === fileId);
      if (row) {
        selectOnly(fileId);
        const card = state.el.grid?.querySelector(`.files-card[data-id="${CSS.escape(fileId)}"]`);
        card?.scrollIntoView({ block: 'center', behavior: 'smooth' });
        return;
      }
      state.selection.clear();
      state.selection.add(fileId);
      state.detailOverrideFile = entry;
      updateSelectionUI();
    }, 160);
  });

  // --- TTS Recording Studio (local Audio section) -------------------
  state.el.ttsStudioBtn?.addEventListener('click', () => {
    import('../tts-studio.js').then(m => m.openTtsStudio()).catch(() => {});
  });
  // Refresh the grid when a recording lands, so it shows up immediately
  // (only matters while the Audio chip is the active view).
  window.addEventListener('augmentum:tts-recording-saved', () => {
    if (state.currentSource === 'audio') loadFiles({ reset: true });
  });

  // --- Upload: button + hidden file input + panel-wide drag-drop -----
  state.el.uploadBtn?.addEventListener('click', () => state.el.uploadInput?.click());
  state.el.uploadInput?.addEventListener('change', (e) => {
    if (e.target.files?.length) {
      _handleUpload(e.target.files);
      e.target.value = '';  // allow re-upload of the same file
    }
  });
  if (state.el.panel && state.el.dropOverlay) {
    let dragDepth = 0;
    const onEnter = (e) => {
      if (!e.dataTransfer?.types?.includes('Files')) return;
      e.preventDefault();
      dragDepth += 1;
      state.el.dropOverlay.classList.add('active');
    };
    const onLeave = () => {
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) state.el.dropOverlay.classList.remove('active');
    };
    state.el.panel.addEventListener('dragenter', onEnter);
    state.el.panel.addEventListener('dragover', (e) => {
      if (e.dataTransfer?.types?.includes('Files')) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
      }
    });
    state.el.panel.addEventListener('dragleave', onLeave);
    state.el.panel.addEventListener('drop', (e) => {
      if (!e.dataTransfer?.files?.length) return;
      e.preventDefault();
      dragDepth = 0;
      state.el.dropOverlay.classList.remove('active');
      _handleUpload(e.dataTransfer.files);
    });
  }

  state.el.tabs?.addEventListener('click', (e) => {
    const chip = e.target.closest('.files-chip');
    if (chip?.dataset.source) _setSource(chip.dataset.source);
  });

  // Scope chooser — tap opens the title-line popover with Local/Cloud
  // options + counts. Selecting one runs _setScope, which re-renders
  // the chip cloud and re-runs the list query. If the user's current
  // chip is orphaned by the new scope (e.g. Audiobooks doesn't exist
  // under Local), _setScope falls back to 'all'.
  state.el.scopeChooser?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (state.el.scopeChooser.getAttribute('aria-expanded') === 'true') {
      _closeScopeMenu();
    } else {
      _openScopeMenu();
    }
  });

  state.el.sortBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggleSortDropdown();
  });
  state.el.sortDropdown?.addEventListener('click', (e) => {
    const opt = e.target.closest('.files-sort-option');
    if (opt?.dataset.sort) { e.stopPropagation(); _setSort(opt.dataset.sort); }
  });

  document.addEventListener('click', (e) => {
    if (state.el.sortDropdown?.classList.contains('open') && !e.target.closest('.files-sort')) {
      _closeSortDropdown();
    }
    if (state.el.filterDropdown?.classList.contains('open') && !e.target.closest('.files-filter')) {
      _closeFilterDropdown();
    }
    if (state.contextMenu && !e.target.closest('.files-context-menu')) {
      hideContextMenu();
    }
  });
  // Escape closes the filter dropdown specifically — sort dropdown
  // already has its own coverage; consolidating into one keydown
  // listener here matches the other "modal-ish" surfaces in the panel.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (state.el.filterDropdown?.classList.contains('open')) {
      _closeFilterDropdown();
    }
  });

  // Long-press state — shared by the click and pointer handlers below.
  // Declared before the click handler so the closure doesn't read from
  // the TDZ if an event somehow arrives before _wireEvents returns.
  const _longPress = { timer: null, x: 0, y: 0, swallowNextClick: false };
  const _clearLongPress = () => {
    if (_longPress.timer) {
      clearTimeout(_longPress.timer);
      _longPress.timer = null;
    }
  };

  state.el.grid?.addEventListener('click', (e) => {
    // Long-press just finished — swallow the click it would otherwise
    // synthesize so we don't immediately toggle back the selection the
    // press just set.
    if (_longPress.swallowNextClick) {
      _longPress.swallowNextClick = false;
      e.stopPropagation();
      return;
    }
    const favBtn = e.target.closest('.files-card-fav');
    if (favBtn && favBtn.dataset.favId) {
      e.stopPropagation();
      toggleFavoriteAction(favBtn.dataset.favId);
      return;
    }
    const playlistBtn = e.target.closest('.files-card-playlist');
    if (playlistBtn && playlistBtn.dataset.playlistId) {
      e.stopPropagation();
      window.dispatchEvent(new CustomEvent('playlist:add-item', {
        detail: {
          type: 'file',
          fileId: playlistBtn.dataset.playlistId,
          name: playlistBtn.dataset.playlistName || '',
          kind: playlistBtn.dataset.playlistKind === 'video' ? 'video' : 'audio',
          thumbnail: '',
        },
      }));
      return;
    }
    const card = e.target.closest('.files-card');
    if (!card || !card.dataset.id) {
      // In select mode, a stray tap shouldn't throw away the batch the
      // user's building. Desktop still deselects on click-off.
      if (!state.selectMode) deselectAll();
      return;
    }
    const id = card.dataset.id;
    const index = parseInt(card.dataset.index, 10);
    if (state.selectMode || e.ctrlKey || e.metaKey) {
      toggleSelect(id);
      state.lastClickedIndex = index;
    } else if (e.shiftKey && state.lastClickedIndex >= 0) {
      selectRange(state.lastClickedIndex, index);
    } else {
      selectOnly(id);
      state.lastClickedIndex = index;
    }
    state.focusedIndex = index;
  });

  // Long-press → enter select mode + toggle the pressed card. Pointer
  // events cover touch + mouse + pen in one path. We cancel the timer on
  // any movement beyond the slop radius so scrolling never accidentally
  // flips into select mode.
  state.el.grid?.addEventListener('pointerdown', (e) => {
    // Long-press is a touch/pen convention — mouse users already have
    // Ctrl-click, Shift-click, and the toolbar Select toggle, and
    // accidentally holding a mouse button shouldn't jump into a mode.
    if (e.pointerType !== 'touch' && e.pointerType !== 'pen') return;
    const card = e.target.closest('.files-card');
    if (!card?.dataset.id) return;
    // Favorite star handles its own click — don't treat a press on it
    // as a long-press on the card.
    if (e.target.closest('.files-card-fav')) return;
    if (e.target.closest('.files-card-playlist')) return;
    _longPress.x = e.clientX;
    _longPress.y = e.clientY;
    _clearLongPress();
    _longPress.timer = setTimeout(() => {
      _longPress.timer = null;
      _longPress.swallowNextClick = true;
      setSelectMode(true);
      toggleSelect(card.dataset.id);
      if (navigator.vibrate) { try { navigator.vibrate(10); } catch { /* */ } }
    }, 450);
  });
  state.el.grid?.addEventListener('pointermove', (e) => {
    if (!_longPress.timer) return;
    const dx = e.clientX - _longPress.x;
    const dy = e.clientY - _longPress.y;
    if ((dx * dx + dy * dy) > 100) _clearLongPress();  // 10px slop
  });
  state.el.grid?.addEventListener('pointerup', _clearLongPress);
  state.el.grid?.addEventListener('pointercancel', _clearLongPress);
  state.el.grid?.addEventListener('pointerleave', _clearLongPress);

  // Toolbar Select toggle — the discoverable mobile path. Entering mode
  // without a selection still reveals the checkbox affordance so the
  // user can see what's about to happen when they tap cards.
  state.el.selectBtn?.addEventListener('click', () => {
    if (state.selectMode) exitSelectMode();
    else setSelectMode(true);
  });

  state.el.grid?.addEventListener('dblclick', (e) => {
    const card = e.target.closest('.files-card');
    if (!card?.dataset.id) return;
    const file = state.files.find(f => f.id === card.dataset.id);
    activateFile(file);
  });

  state.el.grid?.addEventListener('contextmenu', (e) => {
    const card = e.target.closest('.files-card');
    if (!card?.dataset.id) return;
    e.preventDefault();
    const file = state.files.find(f => f.id === card.dataset.id);
    if (!file) return;
    // Preserve an in-progress batch in select mode — right-clicking a
    // card outside the selection shouldn't blow it away.
    if (!state.selection.has(file.id) && !state.selectMode) selectOnly(file.id);
    showContextMenu(e.clientX, e.clientY, file);
  });

  state.el.grid?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      const card = e.target.closest('.files-card');
      if (card?.dataset.id) {
        e.preventDefault();
        const file = state.files.find(f => f.id === card.dataset.id);
        if (e.key === 'Enter') activateFile(file);
        else if (file) {
          selectOnly(file.id);
          state.focusedIndex = parseInt(card.dataset.index, 10);
        }
      }
    }
  });

  // Press-and-hold on the narration mic → voice picker; a plain tap is
  // handled by the click listener below (narrationTap swallows it if a
  // hold already fired).
  state.el.detail?.addEventListener('pointerdown', (e) => {
    const btn = e.target.closest('button[data-action="narration"]');
    if (btn) narrationHoldStart(btn.dataset.id, btn, btn.dataset.name || '');
  });
  for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) {
    state.el.detail?.addEventListener(ev, () => narrationHoldCancel());
  }

  state.el.detail?.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;
    const id = btn.dataset.id;
    if (action === 'download' && id) downloadFile(id);
    else if (action === 'summarize' && id) summarizeFile(id);
    else if (action === 'reference' && id) referenceInChat(id, btn.dataset.name || '');
    else if (action === 'read-aloud' && id) readAloudFile(id, btn);
    else if (action === 'narration' && id) narrationTap(id, btn, btn.dataset.name || '');
    else if (action === 'read-along' && id) openReadAlong(id, btn.dataset.name || '');
    else if (action === 'open-gallery' && id) openGallery(id);
    else if (action === 'expand-preview' && id) openMediaPreview(id, btn.dataset.kind || 'rendered');
    else if (action === 'project' && id) {
      const file = state.files.find(f => f.id === id);
      if (file) openProject(file);
    }
    else if (action === 'unpin' && id) unpinLibrivoxAction(id);
    else if (action === 'add-to-playlist' && id) {
      window.dispatchEvent(new CustomEvent('playlist:add-item', {
        detail: {
          type: 'file',
          fileId: id,
          name: btn.dataset.name || '',
          kind: btn.dataset.kind === 'video' ? 'video' : 'audio',
          thumbnail: '',
        },
      }));
    }
  });

  state.el.bulkBar?.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-action]');
    if (!btn) return;
    if (btn.dataset.action === 'bulk-download') bulkDownload();
    else if (btn.dataset.action === 'bulk-delete') bulkDelete();
    else if (btn.dataset.action === 'bulk-deselect') {
      // Deselect in select mode also exits the mode — hitting "Done" is
      // the natural way to leave. Desktop (no mode active) keeps the
      // pure "clear selection" meaning.
      if (state.selectMode) exitSelectMode();
      else deselectAll();
    }
    else if (btn.dataset.action === 'bulk-restore') restoreAll();
    else if (btn.dataset.action === 'bulk-empty-trash') emptyTrash();
  });

  document.addEventListener('keydown', _handleKeydown);
}

// --- Public API ------------------------------------------------------

export function initFiles() {
  _cacheDom();
  _wireEvents();
  setupScrollObserver(() => loadFiles({ reset: false }));
  setupPeekObserver();
  initComicsListeners();
  initLiveTvListeners();
  _restorePreferenceUI();
  updateViewToggle();
}

function _restorePreferenceUI() {
  // Restore the *visual* state from values loaded by state.js at
  // module-eval time. Keeps the "open Files and it looks how I left
  // it" contract without a render-before-DOM-cache race.
  const label = state.el.sortBtn?.querySelector('.files-sort-label');
  if (label) label.textContent = SORT_LABELS[state.currentSort] || 'Newest';
  state.el.sortDropdown?.querySelectorAll('.files-sort-option').forEach(opt => {
    opt.classList.toggle('active', opt.dataset.sort === state.currentSort);
  });
  // Scope toggle + chip visibility restored first so _setSource's chip
  // highlight doesn't fight with chip hiding.
  _renderScopeToggle();
  _renderChipsForScope();
  // If the restored chip is orphaned by the restored scope (e.g. user had
  // Audiobooks selected, scope was Cloud, then something changed), fall
  // back to 'all' so we don't show a highlighted-but-hidden chip.
  const currentChip = state.el.tabs?.querySelector(
    `.files-chip[data-source="${state.currentSource}"]`,
  );
  if (currentChip && _chipHiddenForScope(currentChip)) {
    state.currentSource = 'all';
  }
  state.el.tabs?.querySelectorAll('.files-chip').forEach(chip => {
    chip.classList.toggle('active', chip.dataset.source === state.currentSource);
  });
  _renderFilterButton();
  _renderSortOptions();
  // Mirror the chip-aware view-toggle visibility from _setSource so the
  // button hides immediately on first open if the saved chip was Comics
  // (rather than flashing in and then hiding when the user touches anything).
  state.el.viewToggle?.classList.toggle('hidden', state.currentSource === COMICS_CHIP);
  state.el.ttsStudioBtn?.classList.toggle('hidden', state.currentSource !== 'audio');
  // Pre-populate genre caches for the chip we're restoring to so the
  // dropdown is hot when the user opens it.
  if (state.currentSource === COMICS_CHIP) _refreshComicGenres();
  if (VIDEO_CLOUD_CHIPS.has(state.currentSource)) _refreshVideoGenres();
}

export function openFiles({ focusSearch = true } = {}) {
  if (!state.el.panel) return;
  state.el.panel.classList.remove('hidden');
  state.el.panel.offsetHeight; // force reflow for transition
  state.el.panel.classList.add('visible');
  state.isOpen = true;
  deselectAll();
  loadFiles({ reset: true });
  loadStats();
  // Focus the search on desktop only when the user explicitly opened the
  // panel. Programmatic opens (e.g. the mini-player deep-link below) pass
  // focusSearch=false so we don't hijack the context they're landing in.
  if (focusSearch && window.innerWidth >= 768) state.el.search?.focus();
}

export function closeFiles() {
  if (!state.el.panel) return;
  state.el.panel.classList.remove('visible');
  state.isOpen = false;
  _closeSortDropdown();
  hideContextMenu();
  // Clear the back-target so a future direct-open (Files button)
  // doesn't surface a stale "Back to Browse" pill from a prior session.
  // _navigateBack already cleared cameFromOverlay before closing, so
  // this only matters for the X/Esc close paths.
  if (state.cameFromOverlay) {
    state.cameFromOverlay = '';
    _updateBackButton();
  }
  setTimeout(() => {
    if (!state.isOpen) state.el.panel.classList.add('hidden');
  }, 300);
}

export function toggleFiles() {
  if (state.isOpen) closeFiles();
  else openFiles();
}
