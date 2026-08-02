/* ==========================================================================
   Composer control — inline web-search sheet

   Expands a search surface from the TOP of the composer (just above the
   textbox) instead of floating a popover over the chat. Living in the normal
   document flow of .input-area means it never fights the overflow (⋯) menu or
   the chat behind it for z-index, needs no backdrop, and needs no off-screen
   re-anchoring. Replaces the old `.web-search-popover`.

   Interaction model:
     - Tap a result body            → attach that page to the chat, close.
     - Press-and-hold a result      → enter multi-select; tap toggles others;
                                       an "Add N" bar commits the batch.
     - Click the arrow (visit) chip → open the original page in a new tab.
     - Esc / ✕ / tap the button again / click outside → close.

   The toggle button (#web-search-btn) lives in the toolbar and, on mobile, may
   be re-parented into the ⋯ overflow menu. We resolve the SHEET via the
   document (it lives in the composer, not the toolbar) and we deliberately do
   NOT stopPropagation on the toggle click so that, on mobile, the click bubbles
   and lets the overflow menu close itself — the two never co-exist.
   ========================================================================== */

import { escapeHtml } from '../../app.js';

const _VISIT_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>`;
const _CHECK_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>`;
const _HOLD_MS = 450;   // press-and-hold threshold to enter multi-select
const _MOVE_TOL = 10;   // px of finger/pointer drift that cancels a hold

/**
 * Wire the web-search button + inline composer sheet.
 *
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root (holds the button).
 * @param {object|null}      surface    Owning surface (unused in Step 2).
 */
export function wireWebSearch(toolbarEl, surface) {
  const btn = toolbarEl?.querySelector('#web-search-btn') ?? null;
  // The sheet lives in the composer (.input-area), not the toolbar — and the
  // button can be re-parented into the ⋯ menu on mobile. Resolve via document.
  const sheet = document.getElementById('composer-search');
  if (!btn || !sheet) return;

  const input = sheet.querySelector('#composer-search-input');
  const results = sheet.querySelector('#composer-search-results');
  const closeBtn = sheet.querySelector('#composer-search-close');
  const multibar = sheet.querySelector('#composer-search-multibar');
  const selCountEl = sheet.querySelector('#composer-search-selcount');
  const addSelBtn = sheet.querySelector('#composer-search-addsel');
  const cancelSelBtn = sheet.querySelector('#composer-search-cancelsel');

  const HINT = '<div class="composer-search-hint">Type a query to find web pages</div>';

  let multiSelect = false;
  const selected = new Set();   // urls chosen during multi-select

  const isOpen = () => !sheet.classList.contains('hidden');

  // ---- open / close -------------------------------------------------------

  const resetSelection = () => {
    multiSelect = false;
    selected.clear();
    sheet.classList.remove('multi');
    multibar.classList.add('hidden');
    results.querySelectorAll('.composer-search-result.selected')
      .forEach(el => el.classList.remove('selected'));
  };

  const open = () => {
    if (isOpen()) return;
    sheet.classList.remove('hidden');
    btn.setAttribute('aria-expanded', 'true');
    btn.classList.add('active');
    // Focus after the expand paints so the keyboard rises smoothly on mobile.
    setTimeout(() => input?.focus(), 0);
  };

  const close = () => {
    if (!isOpen()) return;
    sheet.classList.add('hidden');
    btn.setAttribute('aria-expanded', 'false');
    btn.classList.remove('active');
    resetSelection();
  };

  const toggle = () => { isOpen() ? close() : open(); };

  // No stopPropagation: on mobile the button sits inside the ⋯ popover and we
  // WANT the click to bubble so the overflow menu closes itself.
  btn.addEventListener('click', () => toggle());
  closeBtn?.addEventListener('click', close);

  // Esc closes (keydown listens on the sheet; the input is inside it).
  sheet.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); close(); }
  });

  // Click anywhere outside the sheet (and not on the toggle) closes it.
  document.addEventListener('click', (e) => {
    if (!isOpen()) return;
    if (sheet.contains(e.target)) return;
    if (e.target.closest('#web-search-btn')) return;
    close();
  });

  // Switching mode can hide the toggle button (e.g. narrative uses its own
  // tools), so don't leave the sheet orphaned-open behind a vanished control.
  document.addEventListener('augmentum:mode-changed', close);

  // ---- attach helpers -----------------------------------------------------

  // window._attachWebPage is the existing app.js bridge (fetch + attach to the
  // active session). Read lazily at click-time so wiring order doesn't matter.
  const addPages = async (items) => {
    for (const it of items) {
      if (window._attachWebPage) await window._attachWebPage(it.url, it.title);
    }
  };

  // ---- multi-select -------------------------------------------------------

  const updateMultibar = () => {
    const n = selected.size;
    selCountEl.textContent = `${n} selected`;
    addSelBtn.textContent = `Add ${n}`;
    addSelBtn.disabled = n === 0;
  };

  const enterMultiSelect = (firstUrl, firstRow) => {
    multiSelect = true;
    sheet.classList.add('multi');
    multibar.classList.remove('hidden');
    if (firstUrl) { selected.add(firstUrl); firstRow?.classList.add('selected'); }
    updateMultibar();
  };

  const toggleSel = (url, row) => {
    if (selected.has(url)) selected.delete(url); else selected.add(url);
    row.classList.toggle('selected', selected.has(url));
    updateMultibar();
  };

  addSelBtn?.addEventListener('click', async () => {
    if (!selected.size) return;
    const items = [...results.querySelectorAll('.composer-search-result')]
      .filter(el => selected.has(el.dataset.url))
      .map(el => ({ url: el.dataset.url, title: el.dataset.title }));
    addSelBtn.disabled = true;
    addSelBtn.textContent = 'Adding…';
    await addPages(items);
    close();
  });

  cancelSelBtn?.addEventListener('click', resetSelection);

  // ---- search + render ----------------------------------------------------

  const wireResultRows = () => {
    results.querySelectorAll('.composer-search-result').forEach(row => {
      const url = row.dataset.url;
      const title = row.dataset.title;

      // Arrow chip → open the original page in a new tab (never attaches).
      row.querySelector('.composer-search-visit')?.addEventListener('click', (e) => {
        e.stopPropagation();
        window.open(url, '_blank', 'noopener,noreferrer');
      });

      // Press-and-hold (pointer events cover touch + mouse) → multi-select.
      let holdTimer = null;
      let held = false;          // a hold just fired — swallow the trailing click
      let sx = 0, sy = 0;
      const clearHold = () => { if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; } };

      row.addEventListener('pointerdown', (e) => {
        if (e.target.closest('.composer-search-visit')) return;
        held = false;
        sx = e.clientX; sy = e.clientY;
        holdTimer = setTimeout(() => {
          held = true;
          if (!multiSelect) enterMultiSelect(url, row);
          else toggleSel(url, row);
        }, _HOLD_MS);
      });
      row.addEventListener('pointermove', (e) => {
        if (holdTimer && (Math.abs(e.clientX - sx) > _MOVE_TOL ||
                          Math.abs(e.clientY - sy) > _MOVE_TOL)) clearHold();
      });
      row.addEventListener('pointerup', clearHold);
      row.addEventListener('pointercancel', clearHold);
      row.addEventListener('pointerleave', clearHold);

      // Tap body: multi-select → toggle; otherwise add this one + close.
      row.addEventListener('click', async (e) => {
        if (e.target.closest('.composer-search-visit')) return;
        if (held) { held = false; return; }   // the hold already handled this
        if (multiSelect) { toggleSel(url, row); return; }
        row.classList.add('loading');
        await addPages([{ url, title }]);
        close();
      });

      // Keyboard a11y: Enter/Space mirrors a body tap.
      row.addEventListener('keydown', async (e) => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        e.preventDefault();
        if (multiSelect) { toggleSel(url, row); return; }
        await addPages([{ url, title }]);
        close();
      });
    });
  };

  const doSearch = async () => {
    const query = input?.value.trim();
    if (!query) return;
    resetSelection();
    results.innerHTML = '<div class="composer-search-loading"><span class="composer-search-spinner"></span>Searching…</div>';
    try {
      const resp = await fetch(`/api/browse/search?q=${encodeURIComponent(query)}`);
      const data = await resp.json();
      const items = (data.results || []).slice(0, 8);
      if (!items.length) {
        results.innerHTML = '<div class="composer-search-hint">No results found</div>';
        return;
      }
      results.innerHTML = items.map((r, i) => {
        let host = '';
        try { host = new URL(r.url).hostname.replace(/^www\./, ''); } catch { /* unparseable URL */ }
        const title = r.title || host || r.url;
        return `<div class="composer-search-result" role="button" tabindex="0" data-url="${escapeHtml(r.url)}" data-title="${escapeHtml(title)}">
          <span class="composer-search-rank">${i + 1}</span>
          <span class="composer-search-info">
            <span class="composer-search-title">${escapeHtml(title)}</span>
            <span class="composer-search-domain">${escapeHtml(host)}</span>
          </span>
          <span class="composer-search-check">${_CHECK_ICON}</span>
          <button type="button" class="composer-search-visit" title="Open original page" aria-label="Open original page in new tab">${_VISIT_ICON}</button>
        </div>`;
      }).join('');
      wireResultRows();
      // Keep the freshly-rendered list scrolled to the top.
      results.scrollTop = 0;
    } catch {
      results.innerHTML = '<div class="composer-search-hint">Search failed — try again</div>';
    }
  };

  input?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); doSearch(); }
  });

  // Empty the query → restore the resting hint (and drop any selection).
  input?.addEventListener('input', () => {
    if (!input.value.trim() && !multiSelect) results.innerHTML = HINT;
  });
}
