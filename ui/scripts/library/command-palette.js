/**
 * library/command-palette.js — ⌘K / Ctrl+K fuzzy launcher.
 *
 * Two result classes:
 *   • Items     — fuzzy match against ``display_name`` + ``filename``
 *                 of the user's artifacts (via /api/library/items).
 *   • Actions   — registered verbs (Pin, Cast, New collection, etc.)
 *                 that operate on the current state.
 *
 * Activation: ``open()`` with the current state + handlers. Keyboard
 * model is the canonical command-palette shape — Up/Down to move,
 * Enter to activate, Esc to dismiss. Focus is trapped inside the
 * palette so Tab can't escape; backdrop click also dismisses.
 *
 * The fuzzy score is intentionally cheap (subsequence + prefix bonus)
 * so the palette can re-render on every keystroke without a debounce
 * for catalogs in the low thousands. If the catalog grows past that
 * we can swap in fuse.js or worker-based search without touching the
 * caller.
 */

import { escapeHtml } from '../app.js';
import { fetchItems } from './api.js';
import { labelForFormat } from './types.js';

const MAX_ITEM_HITS = 30;
const ITEM_FETCH_LIMIT = 200;  // single warm fetch; in-process fuzzy after


export class CommandPalette {
  constructor({ onItemActivate, onAction }) {
    this.onItemActivate = onItemActivate || (() => {});
    this.onAction = onAction || (() => {});
    this.actions = [];
    this.allItems = [];     // last-fetched warm cache
    this.results = [];
    this.activeIndex = 0;
    this.open_ = false;
    this._buildDom();
    this._installShortcut();
  }

  // ── DOM ──────────────────────────────────────────────────────────

  _buildDom() {
    this.root = document.createElement('div');
    this.root.className = 'lib-cmdk hidden';
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-modal', 'true');
    this.root.setAttribute('aria-label', 'Command palette');
    this.root.innerHTML = `
      <div class="lib-cmdk-backdrop"></div>
      <div class="lib-cmdk-panel">
        <div class="lib-cmdk-input-wrap">
          <span class="lib-cmdk-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>
            </svg>
          </span>
          <input type="text" class="lib-cmdk-input"
                 placeholder="Search items, or run an action…"
                 autocomplete="off" spellcheck="false"/>
          <kbd class="lib-cmdk-hint">Esc</kbd>
        </div>
        <ul class="lib-cmdk-results" role="listbox"></ul>
      </div>
    `;
    this.input = this.root.querySelector('.lib-cmdk-input');
    this.list = this.root.querySelector('.lib-cmdk-results');
    document.body.appendChild(this.root);

    this.root.querySelector('.lib-cmdk-backdrop')
      .addEventListener('click', () => this.close());
    this.input.addEventListener('input', () => this._refresh());
    this.input.addEventListener('keydown', this._onKeyDown);
    this.list.addEventListener('click', this._onResultClick);
  }

  _installShortcut() {
    document.addEventListener('keydown', (ev) => {
      const meta = ev.metaKey || ev.ctrlKey;
      if (meta && ev.key.toLowerCase() === 'k') {
        // Only when Library2 is the visible overlay. Otherwise leave
        // ⌘K alone for the rest of the app.
        const overlay = document.querySelector('#library-shell-overlay');
        if (!overlay || overlay.classList.contains('hidden')) return;
        ev.preventDefault();
        this.toggle();
      }
    });
  }

  // ── Public API ───────────────────────────────────────────────────

  open(actions = []) {
    this.actions = actions;
    if (this.open_) {
      this.input.focus();
      this.input.select();
      return;
    }
    this.open_ = true;
    this.root.classList.remove('hidden');
    this.input.value = '';
    this.input.focus();
    this._warmCacheIfNeeded();
    this._refresh();
  }

  close() {
    if (!this.open_) return;
    this.open_ = false;
    this.root.classList.add('hidden');
  }

  toggle() {
    if (this.open_) this.close();
    else this.open(this.actions);
  }

  // ── Loading ──────────────────────────────────────────────────────

  async _warmCacheIfNeeded() {
    if (this.allItems.length) return;
    try {
      const body = await fetchItems({ limit: ITEM_FETCH_LIMIT, sort: 'recent' });
      this.allItems = body.items || [];
    } catch (err) {
      console.warn('[library] palette warm-cache failed', err);
    }
    if (this.open_) this._refresh();
  }

  // ── Filtering ────────────────────────────────────────────────────

  _refresh() {
    const q = this.input.value.trim().toLowerCase();
    const itemHits = q
      ? this._fuzzyItems(q)
      : this.allItems.slice(0, 8).map(it => ({ ...it, _kind: 'item', _score: 0 }));
    const actionHits = this.actions
      .map(a => ({ ...a, _score: q ? _scoreSubseq(a.label.toLowerCase(), q) : 0 }))
      .filter(a => !q || a._score > 0)
      .sort((a, b) => b._score - a._score);

    this.results = [
      ...actionHits.map(a => ({ ...a, _kind: 'action' })),
      ...itemHits,
    ];
    this.activeIndex = 0;
    this._renderResults();
  }

  _fuzzyItems(q) {
    const scored = [];
    for (const it of this.allItems) {
      const hay = (it.display_name || it.filename || '').toLowerCase();
      const s = _scoreSubseq(hay, q);
      if (s > 0) scored.push({ ...it, _kind: 'item', _score: s });
    }
    scored.sort((a, b) => b._score - a._score);
    return scored.slice(0, MAX_ITEM_HITS);
  }

  _renderResults() {
    this.list.innerHTML = '';
    if (!this.results.length) {
      const empty = document.createElement('li');
      empty.className = 'lib-cmdk-empty';
      empty.textContent = this.input.value
        ? 'Nothing matches that query.'
        : 'Type to search items, or pick an action.';
      this.list.appendChild(empty);
      return;
    }
    let lastKind = '';
    for (let i = 0; i < this.results.length; i++) {
      const r = this.results[i];
      if (r._kind !== lastKind) {
        const sep = document.createElement('li');
        sep.className = 'lib-cmdk-section';
        sep.textContent = r._kind === 'action' ? 'Actions' : 'Items';
        this.list.appendChild(sep);
        lastKind = r._kind;
      }
      const row = document.createElement('li');
      row.className = 'lib-cmdk-row';
      row.setAttribute('role', 'option');
      row.dataset.index = String(i);
      if (i === this.activeIndex) row.classList.add('active');

      if (r._kind === 'action') {
        row.innerHTML = `
          <span class="lib-cmdk-row-label">${escapeHtml(r.label)}</span>
          ${r.hint ? `<span class="lib-cmdk-row-hint">${escapeHtml(r.hint)}</span>` : ''}
        `;
      } else {
        const sub = labelForFormat(r.format);
        row.innerHTML = `
          <span class="lib-cmdk-row-label">${escapeHtml(r.display_name || r.filename || 'Untitled')}</span>
          <span class="lib-cmdk-row-hint">${escapeHtml(sub)}</span>
        `;
      }
      this.list.appendChild(row);
    }
  }

  // ── Handlers ─────────────────────────────────────────────────────

  _onKeyDown = (ev) => {
    if (ev.key === 'Escape') {
      ev.preventDefault();
      this.close();
      return;
    }
    if (ev.key === 'ArrowDown') {
      ev.preventDefault();
      this._moveSelection(+1);
    } else if (ev.key === 'ArrowUp') {
      ev.preventDefault();
      this._moveSelection(-1);
    } else if (ev.key === 'Enter') {
      ev.preventDefault();
      this._activate(this.activeIndex);
    }
  };

  _onResultClick = (ev) => {
    const row = ev.target.closest('.lib-cmdk-row');
    if (!row) return;
    const idx = Number(row.dataset.index);
    if (!Number.isNaN(idx)) this._activate(idx);
  };

  _moveSelection(delta) {
    const rows = this.list.querySelectorAll('.lib-cmdk-row');
    if (!rows.length) return;
    const next = (this.activeIndex + delta + rows.length) % rows.length;
    this.activeIndex = next;
    rows.forEach((el, i) => el.classList.toggle('active', i === next));
    const active = rows[next];
    if (active) active.scrollIntoView({ block: 'nearest' });
  }

  _activate(index) {
    const r = this.results[index];
    if (!r) return;
    this.close();
    if (r._kind === 'action') {
      this.onAction(r);
    } else {
      this.onItemActivate(r);
    }
  }
}


// ── Fuzzy scoring ──────────────────────────────────────────────────

/**
 * Cheap subsequence match. Returns:
 *   0   — no match (skip)
 *   100 — exact prefix
 *   80  — word-boundary subsequence (consecutive chars after a space/-/_)
 *   50  — contiguous substring anywhere
 *   30  — non-contiguous subsequence
 *
 * Tuned so prefix > word-start > contiguous > scattered. Good enough
 * for thousand-item catalogs; switch to fuse.js when the corpus grows.
 */
function _scoreSubseq(hay, needle) {
  if (!needle) return 0;
  if (hay.startsWith(needle)) return 100;
  const i = hay.indexOf(needle);
  if (i === 0) return 100;
  if (i > 0) {
    const prev = hay[i - 1];
    if (prev === ' ' || prev === '-' || prev === '_' || prev === '.') return 80;
    return 50;
  }
  // Non-contiguous subsequence check.
  let h = 0;
  for (let n = 0; n < needle.length; n++) {
    h = hay.indexOf(needle[n], h);
    if (h === -1) return 0;
    h += 1;
  }
  return 30;
}
