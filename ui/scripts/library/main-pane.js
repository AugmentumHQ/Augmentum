/**
 * library/main-pane.js — middle column.
 *
 * Three view modes:
 *   • list  (default)  — dense rich rows, j/k keyboard nav
 *   • grid             — cover-focused, secondary info on hover
 *   • cover            — large cover-only tiles for visual browsing
 *
 * View mode is per-collection; the active mode is remembered locally
 * via localStorage so a power user who tunes one collection's view
 * doesn't fight the default on every visit.
 *
 * Selection model. The pane fires ``onItemSelect(item)`` when a row
 * gets focused or clicked; parent owns which item the detail pane
 * shows. Parent drives content via show(selection, options) — selection
 * shape matches the sidebar's ({ kind, id }), and options carries
 * search query + cached home payload.
 */

import { escapeHtml } from '../app.js';
import { deleteLibraryItem, fetchItems, getCollection } from './api.js';
import { renderBuildCard } from './build.js';
import { backfillAppPreviews, renderCover } from './cover.js';
import { GamesBrowse } from './games-browse.js';
import { formatsForLabel, friendlyFormat } from './types.js';

const VIEW_MODES = ['list', 'grid', 'cover'];
const SORT_OPTIONS = [
  { id: 'recent', label: 'Recently opened' },
  { id: 'name',   label: 'Name (A–Z)'      },
  { id: 'pinned', label: 'Pinned first'    },
  { id: 'size',   label: 'Largest first'   },
  { id: 'oldest', label: 'Oldest first'    },
];
const LS_VIEW_PREFIX = 'augmentum.library.viewMode.';
const PAGE_SIZE = 60;

const _viewIcon = {
  list:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="4" y1="7" x2="20" y2="7"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="17" x2="20" y2="17"/></svg>',
  grid:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="4" width="7" height="7"/><rect x="13" y="4" width="7" height="7"/><rect x="4" y="13" width="7" height="7"/><rect x="13" y="13" width="7" height="7"/></svg>',
  cover: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="3" width="16" height="18" rx="2"/></svg>',
};

const _pinIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2l2.5 6.5L21 9.7l-5 4.6L17.3 21 12 17.7 6.7 21 8 14.3 3 9.7l6.5-1.2z"/></svg>';

const _selectIcon = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><path d="M16 17l2 2 4-4"/></svg>';

const _checkIcon = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12l5 5L21 6"/></svg>';


export class MainPane {
  constructor(host, { onItemSelect, onBack, onBulkChange, getHome, getActiveBuilds } = {}) {
    this.host = host;
    this.onItemSelect = onItemSelect || (() => {});
    this.onBack = onBack || (() => {});
    this.onBulkChange = onBulkChange || (() => {});
    this.getHome = getHome || (() => ({ pinned: [], recent: [], continue: [] }));
    // Live builds in flight — rendered as a pinned strip atop the body so
    // they accumulate in realtime regardless of the active selection.
    this.getActiveBuilds = getActiveBuilds || (() => []);
    this.selection = { kind: '', id: '' };
    this.query = '';
    this.items = [];
    this.total = 0;
    this.offset = 0;
    this.sort = 'recent';
    this.viewMode = 'list';
    this.activeItemIndex = -1;
    this.loading = false;
    this.selectMode = false;
    this.selected = new Set();

    this._buildDom();
  }

  // ── DOM scaffolding ──────────────────────────────────────────────

  _buildDom() {
    this.host.innerHTML = '';
    this.host.classList.add('lib-main');

    const header = document.createElement('header');
    header.className = 'lib-main-header';
    header.innerHTML = `
      <button type="button" class="lib-back-btn lib-back-to-sidebar"
              aria-label="Back to collections" title="Collections">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
             stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M15 6l-6 6 6 6"/>
        </svg>
      </button>
      <div class="lib-main-title">
        <h2 class="lib-main-heading"></h2>
        <span class="lib-main-subtitle"></span>
      </div>
      <div class="lib-main-controls">
        <button type="button" class="lib-select-toggle" title="Select multiple">
          ${_selectIcon}
        </button>
        <label class="lib-main-sort">
          <span class="visually-hidden">Sort</span>
          <select class="lib-sort-select">
            ${SORT_OPTIONS.map(s => `<option value="${s.id}">${escapeHtml(s.label)}</option>`).join('')}
          </select>
        </label>
        <div class="lib-view-toggle" role="group" aria-label="View mode">
          ${VIEW_MODES.map(m => `
            <button type="button" class="lib-view-btn" data-view="${m}" aria-label="${m} view" title="${m} view">
              ${_viewIcon[m]}
            </button>
          `).join('')}
        </div>
      </div>
    `;
    this.host.appendChild(header);

    this._heading = header.querySelector('.lib-main-heading');
    this._subtitle = header.querySelector('.lib-main-subtitle');
    this._sortSelect = header.querySelector('.lib-sort-select');
    this._viewButtons = [...header.querySelectorAll('.lib-view-btn')];

    this._sortSelect.addEventListener('change', (ev) => {
      this.sort = ev.target.value;
      this._loadAndRender();
    });

    for (const btn of this._viewButtons) {
      btn.addEventListener('click', () => this.setViewMode(btn.dataset.view));
    }

    const back = header.querySelector('.lib-back-to-sidebar');
    back?.addEventListener('click', () => this.onBack());

    this._selectToggle = header.querySelector('.lib-select-toggle');
    this._selectToggle.addEventListener('click', () => this.toggleSelectMode());

    // Sticky footer used in select mode. Hidden by default; revealed by
    // _renderSelectionBar() when something is selected.
    this._footer = document.createElement('div');
    this._footer.className = 'lib-select-footer hidden';
    this._footer.innerHTML = `
      <span class="lib-select-count"></span>
      <div class="lib-select-actions">
        <button type="button" class="lib-select-cancel">Cancel</button>
        <button type="button" class="lib-select-delete">Delete</button>
      </div>
    `;
    this._footer.querySelector('.lib-select-cancel')
      .addEventListener('click', () => this._exitSelectMode());
    this._footer.querySelector('.lib-select-delete')
      .addEventListener('click', () => this._deleteSelected());

    this._body = document.createElement('div');
    this._body.className = 'lib-main-body';
    this._body.tabIndex = 0;
    this.host.appendChild(this._body);
    this.host.appendChild(this._footer);

    this._body.addEventListener('click', this._onBodyClick);
    this._body.addEventListener('keydown', this._onKeyDown);
  }

  // ── Bulk select ───────────────────────────────────────────────────

  toggleSelectMode() {
    this.selectMode ? this._exitSelectMode() : this._enterSelectMode();
  }

  _enterSelectMode() {
    this.selectMode = true;
    this.selected.clear();
    this._selectToggle.classList.add('active');
    this.host.classList.add('select-mode');
    this._renderBody();
    this._renderSelectionBar();
  }

  _exitSelectMode() {
    this.selectMode = false;
    this.selected.clear();
    this._selectToggle.classList.remove('active');
    this.host.classList.remove('select-mode');
    this._footer.classList.add('hidden');
    this._renderBody();
  }

  _toggleSelected(id) {
    if (this.selected.has(id)) this.selected.delete(id);
    else this.selected.add(id);
    this._renderSelectionBar();
    // Lightweight: just flip the row class instead of full re-render.
    const el = this._body.querySelector(`[data-id="${CSS.escape(id)}"]`);
    el?.classList.toggle('selected', this.selected.has(id));
  }

  _renderSelectionBar() {
    const n = this.selected.size;
    this._footer.classList.toggle('hidden', n === 0);
    this._footer.querySelector('.lib-select-count').textContent =
      n === 0 ? '' : `${n} selected`;
    const del = this._footer.querySelector('.lib-select-delete');
    del.disabled = n === 0;
    del.textContent = n === 0 ? 'Delete' : `Delete ${n}`;
  }

  async _deleteSelected() {
    const n = this.selected.size;
    if (n === 0) return;
    if (!confirm(`Delete ${n} item${n === 1 ? '' : 's'}? This cannot be undone.`)) return;
    const ids = [...this.selected];
    const results = await Promise.allSettled(ids.map(deleteLibraryItem));
    const ok = results.filter(r => r.status === 'fulfilled').length;
    // Remove successfully-deleted ids from the local list so the user
    // sees the change without waiting for a reload.
    const dead = new Set(
      ids.filter((_, i) => results[i].status === 'fulfilled'),
    );
    this.items = this.items.filter(it => !dead.has(it.id));
    this._exitSelectMode();
    this._renderHeader();
    this._renderBody();
    this.onBulkChange({ deleted: ok, total: n });
  }

  // ── Public entry point ────────────────────────────────────────────

  async show(selection, { query = '' } = {}) {
    this.selection = selection || { kind: '', id: '' };
    this.query = query || '';
    this.viewMode = _readPreferredView(this.selection);
    this._reflectViewToggle();
    this.activeItemIndex = -1;

    if (this.selection.kind === 'browse-games') {
      this._renderHeader();
      this._renderGamesBrowse();
      return;
    }
    if (this._gamesBrowse) {
      this._gamesBrowse.destroy?.();
      this._gamesBrowse = null;
    }
    await this._loadAndRender();
  }

  _renderGamesBrowse() {
    // GamesBrowse renders directly into the host (this._body) rather
    // than owning a wrapper element, so we can't "re-attach the host"
    // after clearing — host is the body itself, and appending body to
    // body throws HierarchyRequestError. Wiping innerHTML and calling
    // _buildShell() rebuilds the tabs/subtitle/filterbar/grid children
    // and rewires their listeners. Cached state on the instance
    // (active source, filters, bios status) is preserved.
    this._body.innerHTML = '';
    if (!this._gamesBrowse) {
      this._gamesBrowse = new GamesBrowse(this._body, {
        onPinned: () => this.onBulkChange({ pinned: 1 }),
        getSettings: async () => {
          try {
            const mod = await import('../settings.js');
            return mod.getSettings ? mod.getSettings() : {};
          } catch { return {}; }
        },
      });
    } else {
      this._gamesBrowse._buildShell();
    }
    // A deep-link may name which source tab it meant (selection.id),
    // e.g. Discover's "Open" on the console-emulation add-on wants the
    // emulator tab, not whichever source happens to sort first. show()
    // keeps activeSourceId when it's still enabled and otherwise falls
    // back to sources[0], so this is a request, not an override.
    if (this.selection.id) this._gamesBrowse.activeSourceId = this.selection.id;
    this._gamesBrowse.show();
  }

  setViewMode(mode) {
    if (!VIEW_MODES.includes(mode)) return;
    this.viewMode = mode;
    _writePreferredView(this.selection, mode);
    this._reflectViewToggle();
    this._renderBody();
  }

  setQuery(q) {
    this.query = q || '';
    // Phase 4 keeps it simple: when query changes, refetch through the
    // same path. Faster client-side filtering can replace this once
    // we're loading >200 items into a single view.
    this._loadAndRender();
  }

  // ── Loading + rendering ───────────────────────────────────────────

  async _loadAndRender() {
    this._renderHeader();
    this.loading = true;
    this._renderBody();  // render loading state

    try {
      this.items = await this._fetchSelection();
    } catch (err) {
      console.warn('[library] main-pane load failed', err);
      this.items = [];
    } finally {
      this.loading = false;
      this._renderBody();
    }
  }

  async _fetchSelection() {
    const { kind, id } = this.selection;
    const home = this.getHome();

    // Sections that the home payload already gives us — no extra fetch.
    if (kind === 'pinned')   return home.pinned || [];
    if (kind === 'recent')   return home.recent || [];
    if (kind === 'continue') return home.continue || [];

    if (kind === 'collection' && id) {
      const col = await getCollection(id);
      // The endpoint already inlines items in display order.
      return col.items || [];
    }

    const params = {
      q: this.query,
      sort: this.sort,
      limit: PAGE_SIZE,
      offset: 0,
    };
    if (kind === 'type' && id) {
      params.types = formatsForLabel(id);
    }
    if (kind === 'pinned') params.pinned = true;
    const body = await fetchItems(params);
    this.total = body.total;
    this.offset = (body.offset || 0) + (body.items || []).length;
    return body.items || [];
  }

  _renderHeader() {
    const { kind, id } = this.selection;
    let heading = 'Library';
    let subtitle = '';

    if (kind === 'pinned')       { heading = 'Pinned'; }
    else if (kind === 'recent')   { heading = 'Recent'; }
    else if (kind === 'continue') { heading = 'Continue where you left off'; }
    else if (kind === 'all')      { heading = 'All items'; }
    else if (kind === 'type')     { heading = id; }
    else if (kind === 'browse-games') { heading = 'Browse games'; subtitle = 'Public catalogs · pin one to keep it in your Library'; }
    else if (kind === 'collection') {
      heading = id ? this._collectionName(id) : 'Collection';
    }

    if (this.query) {
      subtitle = `Filtered by “${this.query}”`;
    } else if (this.total && (kind === 'all' || kind === 'type')) {
      subtitle = `${this.total} item${this.total === 1 ? '' : 's'}`;
    }

    this._heading.textContent = heading;
    this._subtitle.textContent = subtitle;
  }

  _collectionName(id) {
    // The sidebar already has the collection list cached; the parent
    // could pass it through but this lookup is cheap enough to live here
    // as a window probe with a sensible fallback.
    const cache = window.__libraryState?.collections || [];
    const hit = cache.find(c => c.id === id);
    return hit ? hit.name : 'Collection';
  }

  _reflectViewToggle() {
    for (const btn of this._viewButtons) {
      btn.classList.toggle('selected', btn.dataset.view === this.viewMode);
      btn.setAttribute('aria-pressed', btn.dataset.view === this.viewMode);
    }
    this._body.dataset.view = this.viewMode;
  }

  _renderBody() {
    this._body.innerHTML = '';
    this._mountBuildsStrip();  // live build cards stay pinned at the top

    if (this.loading) {
      const el = document.createElement('div');
      el.className = 'lib-main-state lib-state-loading';
      el.textContent = 'Loading…';
      this._body.appendChild(el);
      return;
    }

    if (!this.items.length && !this.selection.kind) {
      this._renderDashboard();
      return;
    }

    if (!this.items.length) {
      const el = document.createElement('div');
      el.className = 'lib-main-state lib-state-empty';
      el.innerHTML = `
        <div class="lib-empty-line">Nothing here yet.</div>
        <div class="lib-empty-hint">${
          this.query
            ? 'Try a different search.'
            : 'Pin an item or add one to this collection to get started.'
        }</div>
      `;
      this._body.appendChild(el);
      return;
    }

    const list = document.createElement('div');
    list.className = `lib-list lib-list-${this.viewMode}`;
    list.setAttribute('role', 'listbox');
    list.setAttribute('aria-label', this._heading.textContent || 'Library items');

    for (let i = 0; i < this.items.length; i++) {
      const it = this.items[i];
      list.appendChild(this._renderItem(it, i));
    }
    this._body.appendChild(list);
    // Kick the lazy backfill for any app cards without a captured
    // screenshot — the observer inside cover.js only fires once per
    // artifact and only when the cover is in view, so this is cheap.
    backfillAppPreviews(this._body);
  }

  // ── Live builds strip ─────────────────────────────────────────────
  //
  // Owned by the BuildController via getActiveBuilds(); rendered as the
  // first child of the body so in-flight builds stay visible across
  // selection changes. refreshBuilds() updates the strip in place (add/
  // remove a card) without re-fetching the item list; per-tick content
  // updates replace individual cards directly (BuildController._updateCard).

  _mountBuildsStrip() {
    const builds = this.getActiveBuilds() || [];
    if (!builds.length) return;
    const strip = document.createElement('div');
    strip.className = 'lib-builds-strip';
    strip.innerHTML = builds.map(renderBuildCard).join('');
    this._body.insertBefore(strip, this._body.firstChild);
  }

  refreshBuilds() {
    // GamesBrowse owns the body wholesale — don't fight it.
    if (this.selection.kind === 'browse-games') return;
    const builds = this.getActiveBuilds() || [];
    let strip = this._body.querySelector(':scope > .lib-builds-strip');
    if (!builds.length) {
      strip?.remove();
      return;
    }
    if (!strip) {
      strip = document.createElement('div');
      strip.className = 'lib-builds-strip';
      this._body.insertBefore(strip, this._body.firstChild);
    }
    strip.innerHTML = builds.map(renderBuildCard).join('');
  }

  _renderDashboard() {
    const home = this.getHome();
    const wrap = document.createElement('div');
    wrap.className = 'lib-dashboard';

    const sections = [
      { id: 'pinned',   title: 'Pinned',   items: home.pinned || [] },
      { id: 'continue', title: 'Continue', items: home.continue || [] },
      { id: 'recent',   title: 'Recent',   items: home.recent || [] },
    ];
    let drewAny = false;
    for (const sec of sections) {
      if (!sec.items.length) continue;
      drewAny = true;
      const block = document.createElement('section');
      block.className = 'lib-dash-section';
      block.innerHTML = `<h3 class="lib-dash-title">${escapeHtml(sec.title)}</h3>`;
      const strip = document.createElement('div');
      strip.className = 'lib-dash-strip';
      sec.items.slice(0, 8).forEach((it, i) => {
        strip.appendChild(this._renderCoverTile(it, i));
      });
      block.appendChild(strip);
      wrap.appendChild(block);
    }

    if (!drewAny) {
      wrap.innerHTML = `
        <div class="lib-main-state lib-state-empty">
          <div class="lib-empty-line">Your Library is quiet.</div>
          <div class="lib-empty-hint">
            Import a file, browse a game, or create your first artifact
            from the Sources panel on the left.
          </div>
        </div>
      `;
    }
    this._body.appendChild(wrap);
    backfillAppPreviews(this._body);
  }

  _renderItem(item, index) {
    if (this.viewMode === 'list') return this._renderRow(item, index);
    if (this.viewMode === 'grid') return this._renderGridCard(item, index);
    return this._renderCoverTile(item, index);
  }

  _renderRow(item, index) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'lib-row';
    row.setAttribute('role', 'option');
    row.dataset.id = item.id;
    row.dataset.index = String(index);
    if (index === this.activeItemIndex) row.classList.add('active');
    if (this.selected.has(item.id)) row.classList.add('selected');

    const subtitle = _rowSubtitle(item);
    const pinned = item.pinned
      ? `<span class="lib-row-pin" title="Pinned">${_pinIcon}</span>`
      : '';

    row.innerHTML = `
      <span class="lib-check" aria-hidden="true">${_checkIcon}</span>
      <span class="lib-row-cover">
        ${renderCover(item, { size: 'row' })}
      </span>
      <span class="lib-row-text">
        <span class="lib-row-title">${escapeHtml(item.display_name || item.filename || 'Untitled')}</span>
        <span class="lib-row-sub">${escapeHtml(subtitle)}</span>
      </span>
      ${pinned}
    `;
    return row;
  }

  _renderGridCard(item, index) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'lib-card';
    card.setAttribute('role', 'option');
    card.dataset.id = item.id;
    card.dataset.index = String(index);
    if (index === this.activeItemIndex) card.classList.add('active');
    if (this.selected.has(item.id)) card.classList.add('selected');
    card.innerHTML = `
      <span class="lib-card-cover">
        ${renderCover(item, { size: 'card' })}
        <span class="lib-check" aria-hidden="true">${_checkIcon}</span>
      </span>
      <span class="lib-card-title">${escapeHtml(item.display_name || item.filename || 'Untitled')}</span>
      <span class="lib-card-sub">${escapeHtml(_rowSubtitle(item))}</span>
    `;
    return card;
  }

  _renderCoverTile(item, index) {
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = 'lib-tile';
    tile.setAttribute('role', 'option');
    tile.dataset.id = item.id;
    tile.dataset.index = String(index);
    if (index === this.activeItemIndex) tile.classList.add('active');
    if (this.selected.has(item.id)) tile.classList.add('selected');
    tile.innerHTML = `
      <span class="lib-tile-cover">
        ${renderCover(item, { size: 'tile' })}
        <span class="lib-check" aria-hidden="true">${_checkIcon}</span>
      </span>
      <span class="lib-tile-title">${escapeHtml(item.display_name || item.filename || 'Untitled')}</span>
    `;
    return tile;
  }

  // ── Event handlers ────────────────────────────────────────────────

  _onBodyClick = (ev) => {
    const el = ev.target.closest('[data-id]');
    if (!el) return;
    ev.preventDefault();
    if (this.selectMode) {
      this._toggleSelected(el.dataset.id);
      return;
    }
    const idx = Number(el.dataset.index || -1);
    if (idx >= 0) this._activate(idx);
  };

  _onKeyDown = (ev) => {
    if (!this.items.length) return;
    let nextIdx = this.activeItemIndex;
    const isGridish = this.viewMode !== 'list';
    if (ev.key === 'j' || ev.key === 'ArrowDown') {
      nextIdx = Math.min(this.items.length - 1, this.activeItemIndex + 1);
    } else if (ev.key === 'k' || ev.key === 'ArrowUp') {
      nextIdx = Math.max(0, this.activeItemIndex - 1);
    } else if (isGridish && ev.key === 'ArrowRight') {
      nextIdx = Math.min(this.items.length - 1, this.activeItemIndex + 1);
    } else if (isGridish && ev.key === 'ArrowLeft') {
      nextIdx = Math.max(0, this.activeItemIndex - 1);
    } else if (ev.key === 'Enter') {
      if (this.activeItemIndex >= 0) {
        this._activate(this.activeItemIndex);
        return;
      }
    } else {
      return;
    }
    ev.preventDefault();
    if (nextIdx !== this.activeItemIndex) {
      this._setActiveIndex(nextIdx);
    }
  };

  _activate(index) {
    this._setActiveIndex(index);
    const item = this.items[index];
    if (item) this.onItemSelect(item);
  }

  _setActiveIndex(index) {
    const prev = this._body.querySelector('[data-index].active');
    if (prev) prev.classList.remove('active');
    this.activeItemIndex = index;
    const next = this._body.querySelector(`[data-index="${index}"]`);
    if (next) {
      next.classList.add('active');
      next.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }
}


// ── Local helpers ──────────────────────────────────────────────────

function _rowSubtitle(item) {
  const bits = [];
  const fmtLabel = friendlyFormat(item);
  if (fmtLabel) bits.push(fmtLabel);
  if (item.size_bytes) bits.push(_formatBytes(item.size_bytes));
  if (item.last_opened_at) bits.push(`opened ${_relativeTime(item.last_opened_at)}`);
  else if (item.created_at) bits.push(`added ${_relativeTime(item.created_at)}`);
  return bits.join(' · ');
}

function _formatBytes(n) {
  if (!n) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = Number(n);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return v >= 10 || i === 0 ? `${v.toFixed(0)} ${units[i]}` : `${v.toFixed(1)} ${units[i]}`;
}

function _relativeTime(iso) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const diffSec = Math.max(1, (Date.now() - t) / 1000);
  if (diffSec < 90) return 'just now';
  const diffMin = diffSec / 60;
  if (diffMin < 90) return `${Math.round(diffMin)}m ago`;
  const diffHr = diffMin / 60;
  if (diffHr < 36) return `${Math.round(diffHr)}h ago`;
  const diffDay = diffHr / 24;
  if (diffDay < 14) return `${Math.round(diffDay)}d ago`;
  const diffWk = diffDay / 7;
  if (diffWk < 12) return `${Math.round(diffWk)}w ago`;
  return new Date(t).toLocaleDateString();
}

function _viewKey(selection) {
  if (!selection || !selection.kind) return 'home';
  if (selection.id) return `${selection.kind}:${selection.id}`;
  return selection.kind;
}

function _readPreferredView(selection) {
  try {
    return localStorage.getItem(LS_VIEW_PREFIX + _viewKey(selection)) || 'list';
  } catch {
    return 'list';
  }
}

function _writePreferredView(selection, mode) {
  try {
    localStorage.setItem(LS_VIEW_PREFIX + _viewKey(selection), mode);
  } catch {
    // private mode, full storage, etc. — fine to drop.
  }
}
