/**
 * consumption/grid.js — paginated section browse ("See all") for Media
 * and, later, the cast surfaces.
 *
 * Thin client over GET /api/cast/library/section/{slug}, which already
 * supports offset/limit paging, ?sort=, ?status= chips and per-section
 * FTS via ?q= (built for cast-control, unconsumed until now). This
 * module owns the toolbar + grid + load-more; tiles are the shared
 * renderTile component.
 */

import { escapeHtml } from '../app.js';
import { renderTile } from './tile.js';

const PAGE_SIZE = 60;

// Sections whose rows carry playback state get the status chips.
const STATUS_SLUGS = new Set(['movies', 'shows', 'audiobooks', 'recently_added']);
// Comics use a different server-side vocabulary (?filter= series status,
// name_asc sorts) — keep its toolbar minimal for now.
const SEARCH_SLUGS = new Set(['movies', 'shows', 'audiobooks', 'comics', 'recently_added', 'music_videos']);

const STATUS_CHIPS = [
  ['all', 'All'],
  ['not_started', 'Unwatched'],
  ['in_progress', 'In progress'],
  ['finished', 'Finished'],
];

const SORTS = [
  ['', 'Default'],
  ['newest', 'Newest'],
  ['name', 'A–Z'],
  ['year_desc', 'Year ↓'],
  ['year_asc', 'Year ↑'],
];

export function renderSectionGrid(container, section, {
  onTileActivate,
  onTileSecondary,
  variant = 'portrait',
  excludeKinds = null,   // e.g. ['image'] — caller-owned content policy
  keep = null,           // item => boolean — caller-owned scope predicate
} = {}) {
  const slug = section.id;
  const _excluded = excludeKinds && excludeKinds.length
    ? new Set(excludeKinds)
    : null;
  const _keep = typeof keep === 'function' ? keep : null;
  const state = { offset: 0, q: '', status: 'all', sort: '', loading: false, hasMore: false };

  container.innerHTML = `
    <div class="media-grid-toolbar">
      ${SEARCH_SLUGS.has(slug) ? `
        <input class="media-grid-search" type="search"
               placeholder="Search ${escapeHtml((section.title || slug).toLowerCase())}…"
               aria-label="Search this section" autocomplete="off" spellcheck="false">` : ''}
      ${STATUS_SLUGS.has(slug) ? `
        <div class="media-grid-chips" role="group" aria-label="Filter by status">
          ${STATUS_CHIPS.map(([val, label]) =>
            `<button class="media-grid-chip${val === 'all' ? ' is-active' : ''}" type="button" data-status="${val}">${label}</button>`,
          ).join('')}
        </div>` : ''}
      <select class="media-grid-sort" aria-label="Sort">
        ${SORTS.map(([val, label]) => `<option value="${val}">${label}</option>`).join('')}
      </select>
    </div>
    <div class="media-grid" role="list"></div>
    <div class="media-grid-status" hidden></div>
    <button class="media-grid-more" type="button" hidden>Load more</button>
  `;

  const grid = container.querySelector('.media-grid');
  const statusEl = container.querySelector('.media-grid-status');
  const moreBtn = container.querySelector('.media-grid-more');

  async function load({ reset = false } = {}) {
    if (state.loading) return;
    state.loading = true;
    if (reset) {
      state.offset = 0;
      grid.innerHTML = '';
    }
    statusEl.hidden = false;
    statusEl.textContent = 'Loading…';
    moreBtn.hidden = true;
    const params = new URLSearchParams({
      offset: String(state.offset),
      limit: String(PAGE_SIZE),
    });
    if (state.q) params.set('q', state.q);
    if (state.status && state.status !== 'all') params.set('status', state.status);
    if (state.sort) params.set('sort', state.sort);
    try {
      const resp = await fetch(
        `/api/cast/library/section/${encodeURIComponent(slug)}?${params}`,
        { credentials: 'same-origin' },
      );
      if (!resp.ok) {
        statusEl.textContent = `Couldn't load this section (HTTP ${resp.status}).`;
        return;
      }
      const body = await resp.json();
      const rawItems = Array.isArray(body.items) ? body.items : [];
      let items = rawItems;
      if (_excluded) items = items.filter((it) => !_excluded.has(it?.kind || ''));
      if (_keep) items = items.filter(_keep);
      for (const item of items) {
        grid.appendChild(renderTile(item, {
          onActivate: onTileActivate,
          onSecondary: onTileSecondary,
          variant,
        }));
      }
      // Advance by the SERVER's page size, not the filtered count, or
      // paging drifts and re-fetches overlap.
      state.offset += rawItems.length;
      state.hasMore = !!body.has_more && rawItems.length > 0;
      statusEl.hidden = grid.children.length > 0;
      statusEl.textContent = state.q
        ? 'Nothing matched that search.'
        : 'Nothing here yet.';
      moreBtn.hidden = !state.hasMore;
    } catch (err) {
      console.warn('[media-grid] section load failed:', err);
      statusEl.textContent = 'Network error — try again in a moment.';
    } finally {
      state.loading = false;
    }
  }

  // Toolbar wiring
  const searchInput = container.querySelector('.media-grid-search');
  if (searchInput) {
    let timer = 0;
    searchInput.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        state.q = searchInput.value.trim();
        load({ reset: true });
      }, 300);
    });
  }
  container.querySelectorAll('[data-status]').forEach((chip) => {
    chip.addEventListener('click', () => {
      container.querySelectorAll('[data-status]').forEach((c) =>
        c.classList.toggle('is-active', c === chip));
      state.status = chip.dataset.status;
      load({ reset: true });
    });
  });
  container.querySelector('.media-grid-sort').addEventListener('change', (ev) => {
    state.sort = ev.target.value;
    load({ reset: true });
  });
  moreBtn.addEventListener('click', () => load());

  load({ reset: true });
}
