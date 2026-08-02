/**
 * Files panel — "Continue watching/listening" rail.
 *
 * Renders a horizontal scrollable rail of in-progress media at the top
 * of cloud media chips (audiobooks/podcasts/shows/movies/music_videos).
 * Hidden when:
 *   - the active chip isn't a media chip
 *   - the user has an active search query (would compete with results)
 *   - there are no in-progress items
 *
 * Data: ``GET /api/files/search?source={chip}&media_status=in_progress
 *       &sort=progress&limit=20``. The ``progress`` sort maps to
 *       ``updated_at DESC`` server-side, which is the right "most
 *       recently played" proxy because every progress push bumps
 *       ``updated_at`` via update_source_metadata.
 *
 * Comics is intentionally excluded — that chip uses a series-level
 * grid (renderComicsGrid) with its own ``sort=continue`` affordance,
 * which already surfaces in-progress series at the top. Adding a
 * second rail above it would duplicate the same signal.
 */

import { escapeHtml } from '../app.js';
import { state, readBoolPref, saveBoolPref } from './state.js';
import { showContextMenu } from './actions.js';
import { cardOnlyHtml } from './render.js';
import { registerExtraFileSource } from './preview.js';

// Rail items aren't in state.files (especially for shows where the
// grid holds series rows but the rail surfaces episodes). Register a
// provider so preview.js's resolver can find them when the user
// clicks/right-clicks a rail card. The closure reads the current
// rail entries each call — re-renders update the source automatically
// without re-registering.
let _currentEntries = [];
registerExtraFileSource(() => _currentEntries);

// Chips that get the continue rail. Comics is intentionally excluded
// because its series grid is already sorted by `continue` (last-
// read first), which surfaces in-progress series at the top of the
// existing layout — a separate rail above would duplicate the same
// signal. The other media chips have grids sorted by name/recency
// instead, so a dedicated continue rail is the only way to surface
// in-progress items prominently for them.
const RAIL_CHIPS = new Set([
  'audiobooks', 'podcasts', 'shows', 'movies', 'music_videos',
]);

// How many items to surface in the rail. Twenty covers the common
// "I'm in the middle of a few things" case without making the API
// fetch a meaningful payload.
const RAIL_LIMIT = 20;

const VIDEO_CHIPS = new Set(['shows', 'movies', 'music_videos']);

// Dedup recent-render guard. Multiple triggers can stack (chip
// switch + initial load callback + media-player progress event). We
// keep a per-source token and ignore stale completions so a slow
// fetch returning AFTER the user moved on doesn't overwrite what's
// on screen.
let _renderToken = 0;

export function shouldShowRail(source) {
  return RAIL_CHIPS.has(source);
}

export async function renderContinueRail(source) {
  const rail = state.el.continueRail;
  if (!rail) return;

  if (!shouldShowRail(source)) {
    _hideRail(rail);
    return;
  }

  // Don't compete with search results. The user typed something —
  // they're looking for it, not browsing.
  if (state.el.search?.value?.trim()) {
    _hideRail(rail);
    return;
  }

  const myToken = ++_renderToken;

  const params = new URLSearchParams({
    source,
    media_status: 'in_progress',
    sort: 'progress',
    limit: String(RAIL_LIMIT),
  });
  // Shows chip defaults to entity_kind=series (its grid is series-
  // level), but progress is tracked per episode. Override so the
  // rail surfaces the episode the user is mid-watch on, not the
  // parent series (which never has a progress_pct itself). The
  // /api/files/search route accepts entity_kind as an explicit
  // override that wins over the chip's default entity_kinds set.
  if (source === 'shows') {
    params.set('entity_kind', 'episode');
  }

  let entries = [];
  try {
    const resp = await fetch(`/api/files/search?${params}`);
    if (myToken !== _renderToken) return; // user moved on
    if (!resp.ok) {
      _hideRail(rail);
      return;
    }
    const data = await resp.json();
    entries = Array.isArray(data?.files) ? data.files : [];
  } catch {
    _hideRail(rail);
    return;
  }

  if (myToken !== _renderToken) return;

  if (entries.length === 0) {
    _hideRail(rail);
    _currentEntries = [];
    return;
  }
  // Publish to the shared resolver so preview.js can find these
  // entries by id even when they aren't in state.files.
  _currentEntries = entries;

  // Reuse the EXACT same card markup (.files-card) used by the
  // grid below — same thumb sizing, title placement, progress bar,
  // tag pills, etc. The rail just wraps them in a horizontally
  // scrolling row instead of the grid's auto-fill columns.
  const heading = VIDEO_CHIPS.has(source) ? 'Continue watching' : 'Continue listening';
  // For shows, the rail surfaces episodes (entity_kind=episode) but
  // users think in series ("resume Breaking Bad"). Pre-transform so
  // the card displays "Breaking Bad" as the title with "S01E05 ·
  // Pilot" in the meta line, not "Pilot" with "S01E05 · Breaking
  // Bad". Original entry stays intact for click handling.
  const cardEntries = source === 'shows'
    ? entries.map(_showsDisplayEntry)
    : entries;
  const collapsed = readBoolPref('continueRailCollapsed', false);
  // Header is a button so the entire row is one tap target — chevron
  // alone would be too small on touch. aria-expanded reflects the
  // current state for screen readers; the chevron just visualizes it.
  // .files-continue-rail.collapsed hides the track via CSS while
  // leaving the header visible so the user always knows what's there
  // and can re-open it.
  rail.innerHTML = `
    <button type="button" class="files-continue-rail-header"
            aria-expanded="${collapsed ? 'false' : 'true'}"
            data-rail-toggle>
      <span>${escapeHtml(heading)}</span>
      <svg class="files-continue-rail-chevron" width="14" height="14"
           viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"
           aria-hidden="true">
        <polyline points="6 15 12 9 18 15"/>
      </svg>
    </button>
    <div class="files-continue-rail-track" role="list">
      ${cardEntries.map((f, i) => cardOnlyHtml(f, i)).join('')}
    </div>
  `;
  rail.classList.toggle('collapsed', collapsed);
  rail.classList.remove('hidden');

  // Toggle handler. Keeps the rail node + its content; just flips a
  // class so collapsed state is a CSS concern. State persists via
  // localStorage so a user who collapsed it to dig through the
  // catalog stays collapsed across chip switches and reloads.
  const toggleBtn = rail.querySelector('[data-rail-toggle]');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const wasCollapsed = rail.classList.contains('collapsed');
      const nextCollapsed = !wasCollapsed;
      rail.classList.toggle('collapsed', nextCollapsed);
      toggleBtn.setAttribute('aria-expanded', nextCollapsed ? 'false' : 'true');
      saveBoolPref('continueRailCollapsed', nextCollapsed);
    });
  }

  // Click delegation. The grid's own click handler is bound to
  // state.el.grid, not this rail — so we wire ``_resume`` here.
  // Single click → resume playback at the saved position, matching
  // the Plex/Netflix "continue watching" affordance.
  rail.querySelectorAll('.files-card').forEach(el => {
    const id = el.dataset.id;
    const entry = entries.find(e => e.id === id);
    if (!entry) return;
    el.addEventListener('click', (e) => {
      // Don't intercept clicks that landed on the favorite star,
      // selection check, or any other button/anchor inside the
      // card — those have their own meanings and should NOT open
      // the file. Bare-card clicks fall through to resume.
      if (e.target.closest('button, a')) return;
      _resume(entry);
    });
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        _resume(entry);
      }
    });
    // Right-click → the SAME shared menu the grid uses, so menu
    // items (mark watched, reset progress, rename, etc.) stay in
    // sync between surfaces. No rail-specific menu duplication.
    el.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      showContextMenu(e.clientX, e.clientY, entry);
    });
  });
}

/**
 * Open a rail entry. The per-kind routing rules that used to live
 * inline here (audio → progress-safe mini-player singleton, video →
 * selected_episode_id synthesis for the media-server guard) moved to
 * files/open-content.js during the 2026-07-18 class fix — see that
 * module's header for the full history, including the audiobook
 * progress-wipe that motivated the audio rule.
 */
function _resume(entry) {
  // Delegates to the canonical off-surface opener (open-content.js),
  // which grew out of this function's rules during the 2026-07-18
  // class fix: audio → media-player singleton, media-server leaf rows
  // → selected_episode_id synthesis, everything else → activateFile,
  // never a silent dead end. One implementation for every surface.
  import('./open-content.js').then(m => m.openContent(entry));
}

/**
 * Reflect a media-player progress update inside the rail without a
 * round-trip refetch. Three behaviors:
 *   - Card has progress 0 < pct < 1 and isn't finished → resize the
 *     progress bar in place.
 *   - Card just finished (isFinished=true) OR was reset to 0
 *     (progressPct=0) → it's no longer in the in_progress bucket;
 *     remove the card from the row, and hide the rail if it goes
 *     empty. The next chip switch / load re-fetches authoritatively.
 *
 * Listens for the same ``media-player:progress`` event the grid uses,
 * so user actions (live playback, mark-watched from context menu,
 * reset-progress from context menu) all flow through one event and
 * keep both surfaces consistent.
 */
export function patchRailProgress(fileId, progressPct, opts = {}) {
  const rail = state.el.continueRail;
  if (!rail) return;
  const card = rail.querySelector(`.files-card[data-id="${CSS.escape(fileId)}"]`);
  if (!card) return;

  const pct = Math.max(0, Math.min(1, progressPct || 0));
  const leftBucket = opts.isFinished === true || pct <= 0;
  if (leftBucket) {
    // Row is no longer "in progress" — drop it from the rail.
    card.remove();
    const remaining = rail.querySelectorAll('.files-card').length;
    if (remaining === 0) _hideRail(rail);
    return;
  }

  // Cards in the rail use the same .files-card class as the grid,
  // so the progress bar selector matches what render.js paints.
  let bar = card.querySelector('.files-card-progress span');
  if (!bar) {
    const wrap = document.createElement('div');
    wrap.className = 'files-card-progress';
    wrap.setAttribute('aria-hidden', 'true');
    bar = document.createElement('span');
    wrap.appendChild(bar);
    card.appendChild(wrap);
  }
  bar.style.width = `${(pct * 100).toFixed(1)}%`;
}

function _hideRail(rail) {
  rail.classList.add('hidden');
  rail.innerHTML = '';
  // Clear the resolver source so a stale entry can't satisfy a
  // late-arriving lookup after the rail is gone (e.g. an event
  // dispatched after chip switch).
  _currentEntries = [];
}

/**
 * Shallow-copy an episode entry with name + series_name swapped so the
 * grid card's title shows the SERIES and the meta line shows the
 * episode. The card builder reads ``f.name`` for the title and
 * ``source_metadata.series_name`` for the "S01E05 · {x}" meta tail —
 * swapping their values makes the card read like a Plex "continue
 * watching" tile without touching the shared card-builder logic.
 *
 * Falls through to the original entry when series_name isn't set
 * (e.g. an orphan episode whose parent series we don't know about) —
 * better to show the episode name than render an empty title.
 */
function _showsDisplayEntry(file) {
  const meta = file.source_metadata || {};
  const seriesName = (meta.series_name || '').trim();
  const episodeName = (file.name || '').trim();
  if (!seriesName) return file;
  return {
    ...file,
    name: seriesName,
    source_metadata: {
      ...meta,
      // The meta-line builder uses ``series_name`` after the SxxExx
      // marker; substituting the episode name there yields
      // "S01E05 · Pilot" without a custom render path.
      series_name: episodeName,
    },
  };
}

// ────────────────────────────────────────────────────────────────────
// Rail freshness — auto-refetch triggers.
//
// patchRailProgress updates the bar on EXISTING cards in place, but
// it never reorders cards or adds new ones. So when the user plays
// something new (in-session, in the same chip), the rail's order
// stays frozen until a chip-switch refresh. Worse, cast-driven
// playback doesn't fire ``media-player:progress`` at all — the cast
// surfaces push progress directly to the backend, bypassing the
// web app's event bus.
//
// Two listeners close the gap without a heavy push system:
//
//   1. ``visibilitychange`` → re-fetch when the user comes back to
//      the tab. Covers the dominant "I was casting on the TV, now
//      I'm looking at my Files panel" flow. Cheap (one search call).
//
//   2. ``media-player:progress`` → debounced re-fetch ~5s after the
//      last event. Covers in-session reordering (web playback that
//      crosses into a new item, or the cast surface eventually
//      bouncing a progress signal via the same event if it's wired
//      to the main page).
//
// Both gate on the current source being a RAIL_CHIPS member so the
// fetch is skipped when the rail isn't visible anyway.

let _refreshDebounceTimer = null;
const _PROGRESS_REFRESH_DEBOUNCE_MS = 5000;

function _refreshIfRailVisible() {
  const source = state?.currentSource;
  if (!source || !shouldShowRail(source)) return;
  if (state.el?.search?.value?.trim()) return;  // search mode hides rail
  renderContinueRail(source);
}

if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      _refreshIfRailVisible();
    }
  });
}

if (typeof window !== 'undefined') {
  window.addEventListener('media-player:progress', () => {
    if (_refreshDebounceTimer) clearTimeout(_refreshDebounceTimer);
    _refreshDebounceTimer = setTimeout(() => {
      _refreshDebounceTimer = null;
      _refreshIfRailVisible();
    }, _PROGRESS_REFRESH_DEBOUNCE_MS);
  });
}
