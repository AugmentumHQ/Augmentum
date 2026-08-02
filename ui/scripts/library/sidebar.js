/**
 * library/sidebar.js — left rail.
 *
 * Sections, top-to-bottom:
 *   • Search input (live filter; ⌘K palette lands in Phase 6)
 *   • YOURS         — Pinned / Recent / Continue
 *   • COLLECTIONS   — "All" + auto-type virtuals (Apps/Docs/Games/...) +
 *                     user-defined collections
 *   • SOURCES       — Browse games / Import file / Create artifact
 *
 * The sidebar owns selection state visually (active row gets a brass
 * left-edge rule) but parent owns the source of truth. The parent
 * passes ``selection`` on each render() and listens for selection
 * changes via the ``onSelect`` handler.
 *
 *   selection = { kind, id }
 *     kind ∈ 'pinned' | 'recent' | 'continue' | 'all' | 'type' | 'collection'
 *     id is the discriminator within kind ('html', 'col_xyz', or '')
 *
 * Source actions fire ``onSource(action)`` with action ∈
 *   'browse-games' | 'import' | 'create' | 'new-collection'.
 *
 * The renderer is idempotent — passing the same payload twice produces
 * identical DOM; we only replace children that actually change.
 */

import { escapeHtml } from '../app.js';
import { LABEL_FALLBACK, labelForFormat } from './types.js';

// ── Atelier-flavored icon set (16x16 stroke, currentColor) ─────────
// Hand-tuned per-section so each row reads at a glance without
// rendering an emoji that pulls in the OS' color palette.
const ICONS = {
  pinned: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2l2.5 6.5L21 9.7l-5 4.6L17.3 21 12 17.7 6.7 21 8 14.3 3 9.7l6.5-1.2z"/></svg>',
  recent: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  continue: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 4l14 8-14 8z"/></svg>',
  all: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="6" rx="1"/><rect x="3" y="14" width="18" height="6" rx="1"/></svg>',
  html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M10 9.2v5.6l5-2.8z" fill="currentColor" stroke="none"/></svg>',
  game: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 4h6v5h5v6h-5v5H9v-5H4V9h5z"/></svg>',
  doc: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/></svg>',
  image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.5"/><path d="M3 17l5-5 4 4 3-3 6 6"/></svg>',
  sheet: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="1"/><path d="M3 10h18M3 16h18M9 4v16M15 4v16"/></svg>',
  slides: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="5" width="14" height="10" rx="1"/><rect x="7" y="9" width="14" height="10" rx="1"/></svg>',
  bundle: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7l9-4 9 4-9 4z"/><path d="M3 12l9 4 9-4M3 17l9 4 9-4"/></svg>',
  generic: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/></svg>',
  browseGames: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a13 13 0 0 1 0 18M12 3a13 13 0 0 0 0 18"/></svg>',
  import: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12M7 10l5 5 5-5"/><path d="M5 18h14"/></svg>',
  create: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
  buildApp: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 7l-1.5-1.5a3 3 0 0 0-4 4L10 11l-5 5 3 3 5-5 1.5 1.5a3 3 0 0 0 4-4L17 9"/></svg>',
  newCollection: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7h12l2 3h4v10H3z"/><path d="M14 14h4M16 12v4"/></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg>',
};

// Icon picked per label so the row reads at a glance.
const LABEL_ICONS = {
  Apps: 'html', Games: 'game', Documents: 'doc', Books: 'doc',
  Notes: 'doc', Slides: 'slides', Sheets: 'sheet', Images: 'image',
  Other: 'generic',
};

// ── Sidebar class ──────────────────────────────────────────────────

export class Sidebar {
  constructor(host, { onSelect, onSource, onSearch } = {}) {
    this.host = host;
    this.onSelect = onSelect || (() => {});
    this.onSource = onSource || (() => {});
    this.onSearch = onSearch || (() => {});
    this._searchTimer = null;
    this._lastQuery = '';
    this._wire();
  }

  _wire() {
    // Single delegated handler covers all nav rows. Saves us from
    // re-binding on every render and keeps the DOM diff cheap.
    this.host.addEventListener('click', (ev) => {
      const item = ev.target.closest('.lib-nav-item');
      if (!item) return;
      const target = item.dataset.target;
      const action = item.dataset.action;
      if (action) {
        this.onSource(action);
        return;
      }
      if (target) {
        const [kind, id = ''] = target.split(':');
        this.onSelect({ kind, id });
      }
    });
  }

  render(homePayload, collections, selection) {
    this.host.innerHTML = '';

    // Search input. Debounced 150ms so we don't thrash on every keystroke
    // but stay responsive enough that the user feels heard.
    const searchWrap = document.createElement('div');
    searchWrap.className = 'lib-search-wrap';
    searchWrap.innerHTML = `
      <span class="lib-search-icon" aria-hidden="true">${ICONS.search}</span>
      <input
        type="search"
        class="lib-search"
        placeholder="Search Library"
        autocomplete="off"
        spellcheck="false"
        value="${escapeHtml(this._lastQuery)}"
      />
    `;
    const searchInput = searchWrap.querySelector('.lib-search');
    searchInput.addEventListener('input', (ev) => {
      const q = ev.target.value;
      this._lastQuery = q;
      clearTimeout(this._searchTimer);
      this._searchTimer = setTimeout(() => this.onSearch(q), 150);
    });
    this.host.appendChild(searchWrap);

    const nav = document.createElement('nav');
    nav.className = 'lib-nav';

    // ── Yours ─────────────────────────────────────────────────
    nav.appendChild(_renderSection('Yours', [
      _row({
        target: 'pinned', icon: ICONS.pinned, label: 'Pinned',
        count: (homePayload.pinned || []).length,
        selected: selection?.kind === 'pinned',
      }),
      _row({
        target: 'recent', icon: ICONS.recent, label: 'Recent',
        count: (homePayload.recent || []).length,
        selected: selection?.kind === 'recent',
      }),
      _row({
        target: 'continue', icon: ICONS.continue, label: 'Continue',
        count: (homePayload.continue || []).length,
        selected: selection?.kind === 'continue',
      }),
    ]));

    // ── Collections (All + auto-type + user) ──────────────────
    const typeCounts = homePayload.type_counts || {};
    const totalCount = homePayload.total_count
      || Object.values(typeCounts).reduce((a, b) => a + b, 0);

    // Roll up format-level counts to the display labels. types.js owns
    // the format → label mapping; we just sum the buckets here.
    const labelCounts = new Map();
    for (const [fmt, count] of Object.entries(typeCounts)) {
      const label = labelForFormat(fmt) || LABEL_FALLBACK;
      labelCounts.set(label, (labelCounts.get(label) || 0) + count);
    }

    const typeRows = [];
    typeRows.push(_row({
      target: 'all', icon: ICONS.all, label: 'All',
      count: totalCount, selected: selection?.kind === 'all',
    }));
    // Stable alphabetical ordering so the rail doesn't reshuffle as
    // new types appear.
    for (const [label, count] of [...labelCounts.entries()].sort(
      (a, b) => a[0].localeCompare(b[0]))
    ) {
      const iconKey = LABEL_ICONS[label] || 'generic';
      // Selection id is the label itself — main pane reads
      // formatsForLabel() to expand it into the SQL filter.
      typeRows.push(_row({
        target: `type:${label}`, icon: ICONS[iconKey], label,
        count,
        selected: selection?.kind === 'type' && selection?.id === label,
      }));
    }

    // User-defined collections, sorted by sort_order (server provides it).
    const userCollections = (collections || []).slice().sort(
      (a, b) => (a.sort_order || 0) - (b.sort_order || 0),
    );
    for (const c of userCollections) {
      typeRows.push(_row({
        target: `collection:${c.id}`,
        icon: ICONS.bundle,
        label: c.name,
        count: c.count || 0,
        accent: c.accent_color || '',
        selected: selection?.kind === 'collection' && selection?.id === c.id,
        dynamic: c.kind === 'dynamic',
      }));
    }

    nav.appendChild(_renderSection('Collections', typeRows, {
      action: { kind: 'new-collection', icon: ICONS.newCollection,
                label: 'New collection' },
    }));

    // ── Sources ───────────────────────────────────────────────
    nav.appendChild(_renderSection('Sources', [
      _row({
        action: 'build-app',
        icon: ICONS.buildApp, label: 'Build an app…',
      }),
      _row({
        action: 'browse-games',
        icon: ICONS.browseGames, label: 'Browse games…',
      }),
      _row({
        action: 'import',
        icon: ICONS.import, label: 'Import file…',
      }),
      _row({
        action: 'create',
        icon: ICONS.create, label: 'Create artifact…',
      }),
    ]));

    this.host.appendChild(nav);
  }

  focusSearch() {
    const el = this.host.querySelector('.lib-search');
    el?.focus();
  }

  clearSearch() {
    const el = this.host.querySelector('.lib-search');
    if (el) el.value = '';
    this._lastQuery = '';
    this.onSearch('');
  }
}


// ── Internal helpers ───────────────────────────────────────────────

function _renderSection(title, children, opts = {}) {
  const section = document.createElement('section');
  section.className = 'lib-section';

  const header = document.createElement('h3');
  header.className = 'lib-section-title';
  header.innerHTML = `<span>${escapeHtml(title)}</span>`;
  if (opts.action) {
    const btn = document.createElement('button');
    btn.className = 'lib-section-add';
    btn.type = 'button';
    btn.dataset.action = opts.action.kind;
    btn.setAttribute('aria-label', opts.action.label);
    btn.innerHTML = opts.action.icon;
    header.appendChild(btn);
  }
  section.appendChild(header);

  for (const c of children) {
    section.appendChild(c);
  }
  return section;
}

function _row({
  target = '', action = '', icon = '', label = '', count = null,
  selected = false, accent = '', dynamic = false,
}) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'lib-nav-item';
  if (selected) btn.classList.add('selected');
  if (dynamic) btn.classList.add('dynamic');
  if (target) btn.dataset.target = target;
  if (action) btn.dataset.action = action;
  if (accent) btn.style.setProperty('--lib-row-accent', accent);

  const countSpan = (count === null || count === undefined)
    ? ''
    : `<span class="lib-nav-count">${escapeHtml(String(count))}</span>`;
  btn.innerHTML = `
    <span class="lib-nav-icon" aria-hidden="true">${icon}</span>
    <span class="lib-nav-label">${escapeHtml(label)}</span>
    ${countSpan}
  `;
  return btn;
}
