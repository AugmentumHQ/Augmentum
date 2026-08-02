/**
 * librivox-browse.js — Browse panel for the free LibriVox public-domain library.
 *
 * Surfaces a live-search grid backed by /api/media/browse/librivox. Each card
 * shows a cover (with client-side fallback when archive.org returns nothing),
 * title, author, runtime, and a primary "+ Library" action. Pinning a result
 * POSTs to /api/media/pin, which promotes it into file_index so the rest of
 * the media UI (Files panel, player, progress) picks it up unchanged.
 *
 * Design notes:
 *   - All user-visible strings go through escapeHtml(); LibriVox descriptions
 *     sometimes contain HTML and would be XSS gold otherwise.
 *   - The fallback cover is generated from a deterministic hash of the
 *     archive identifier so the same book always renders the same gradient.
 *   - Pin button shows optimistic "Pinning…" state; errors revert the card.
 *   - The overlay is a full-screen dialog, not a drawer — the grid needs room.
 */

import { escapeHtml, showToast } from './app.js';

let _overlay = null;
let _state = {
  query:     '',   // free-text from the search input
  category:  '',   // subject: filter from the chip row
  page:      1,
  pageSize:  24,
  results:   [],
  hasMore:   false,
  loading:   false,
  // Recent-mode: true when the grid is showing LibriVox's "cataloged
  // since N days" feed. Set on overlay open (landing state); cleared the
  // moment the user types a query or picks a category chip. UI renders
  // a "Recently added on LibriVox" subheading while this is true.
  recent:    false,
  pinning:   new Set(),   // external_ids currently in flight
};

// Category chips — backed by archive.org's `subject:` filter over the
// librivoxaudio collection. Each entry is (display label, archive
// subject term). The subject term must be lowercase to match archive
// indexing; the label is free to use title-case. LibriVox's own feed
// `search=` / `genre_id=` are inert upstream (verified 2026-04-20), so
// these chips go through archive.org's real search index instead.
const CATEGORY_CHIPS = [
  { label: 'Fiction',         subject: 'fiction' },
  { label: 'Mystery',         subject: 'mystery' },
  { label: 'Horror',          subject: 'horror' },
  { label: 'Adventure',       subject: 'adventure' },
  { label: 'Romance',         subject: 'romance' },
  { label: 'Science Fiction', subject: 'science fiction' },
  { label: 'Fantasy',         subject: 'fantasy' },
  { label: 'Children',        subject: 'children' },
  { label: 'Poetry',          subject: 'poetry' },
  { label: 'Short Stories',   subject: 'short stories' },
  { label: 'Humor',           subject: 'humor' },
  { label: 'Drama',           subject: 'drama' },
  { label: 'Biography',       subject: 'biography' },
  { label: 'History',         subject: 'history' },
  { label: 'Philosophy',      subject: 'philosophy' },
  { label: 'Religion',        subject: 'religion' },
  { label: 'Nonfiction',      subject: 'nonfiction' },
];

export async function openLibrivoxBrowse() {
  if (!_overlay) _buildOverlay();
  _overlay.classList.add('visible');
  document.body.classList.add('lv-lock-scroll');
  // Landing state: fetch LibriVox's "recently cataloged" feed so the
  // overlay opens on something fresh rather than the archive.org
  // default alphabetical first page. Flips off as soon as the user
  // types in the search box or picks a category chip.
  if (_state.results.length === 0) {
    _state.recent = true;
    await _loadPage({ reset: true });
  } else {
    _renderGrid();
  }
}

export function closeLibrivoxBrowse() {
  if (!_overlay) return;
  _overlay.classList.remove('visible');
  document.body.classList.remove('lv-lock-scroll');
  // Fire a DOM event so surfaces that opened us (Files' Audiobooks chip
  // toggle) can revert their UI. Keeps the coupling one-way — this module
  // doesn't know or care who's listening.
  window.dispatchEvent(new CustomEvent('librivox-browse:closed'));
}

// --- Data loading ----------------------------------------------------

async function _loadPage({ reset = false } = {}) {
  if (_state.loading) return;
  _state.loading = true;
  if (reset) {
    _state.page = 1;
    _state.results = [];
    _state.hasMore = false;
  }
  _renderGrid();   // show loading indicator

  // Default landing state: fetch "recently added on LibriVox" instead
  // of archive.org's default first page. The moment the user types or
  // picks a chip, recent flips off and we fall back to the normal
  // browse path. Pagination doesn't apply to the recent feed — it's a
  // fixed 30-day window — so page stays pinned at 1 when recent.
  const useRecent = !_state.query && !_state.category && _state.recent;
  const params = new URLSearchParams({
    page:      String(_state.page),
    page_size: String(_state.pageSize),
  });
  if (useRecent)       params.set('recent', '1');
  if (_state.query)    params.set('q', _state.query);
  if (_state.category) params.set('category', _state.category);

  let resp, body;
  try {
    resp = await fetch(`/api/media/browse/librivox?${params.toString()}`);
    body = await resp.json().catch(() => ({}));
  } catch (err) {
    showToast(`Couldn't reach LibriVox: ${err.message || 'network error'}`, 'error', 4000);
    _state.loading = false;
    _renderGrid();
    return;
  }
  if (!resp.ok) {
    showToast(body.error || 'LibriVox is unavailable right now', 'error', 4000);
    _state.loading = false;
    _renderGrid();
    return;
  }

  const fresh = Array.isArray(body.results) ? body.results : [];
  _state.results = reset ? fresh : _state.results.concat(fresh);
  _state.hasMore = !!body.has_more;
  // Trust the server's acknowledgement: if we asked for recent and the
  // server said it served recent, keep recent mode on for the label.
  _state.recent = !!body.recent;
  _state.loading = false;
  _renderGrid();
}

// --- Pin / unpin -----------------------------------------------------

async function _pinItem(result) {
  if (_state.pinning.has(result.external_id)) return;
  _state.pinning.add(result.external_id);
  _renderGrid();

  const payload = {
    provider:    'librivox',
    external_id: result.external_id,
    name:        result.name || 'Untitled',
    author:      result.author || '',
    narrator:    result.narrator || '',
    description: result.description || '',
    cover_url:   result.cover_url || '',
    duration_ms: Number(result.duration_ms) || 0,
    license:     result.license || 'public-domain',
    extra:       result.extra || {},
  };

  let resp, body;
  try {
    resp = await fetch('/api/media/pin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    body = await resp.json().catch(() => ({}));
  } catch (err) {
    showToast(`Pin failed: ${err.message || 'network error'}`, 'error', 4000);
    _state.pinning.delete(result.external_id);
    _renderGrid();
    return;
  }

  _state.pinning.delete(result.external_id);

  if (!resp.ok || !body.file_id) {
    showToast(body.error || 'Pin failed', 'error', 4000);
    _renderGrid();
    return;
  }

  // Update the local result so the grid shows the pinned state without
  // a refetch. The file_id lets unpin work from the same card.
  for (const r of _state.results) {
    if (r.external_id === result.external_id) {
      r.pinned = true;
      r.pinned_file_id = body.file_id;
    }
  }
  showToast(
    body.already_pinned
      ? `Already in your library`
      : `Added "${result.name}" to your library`,
    'success', 2500,
  );
  // Tell the rest of the UI a pin happened so the Files panel refreshes.
  window.dispatchEvent(new CustomEvent('media-servers:changed'));
  _renderGrid();
}

async function _unpinItem(result) {
  const fileId = result.pinned_file_id;
  if (!fileId) return;
  if (_state.pinning.has(result.external_id)) return;
  _state.pinning.add(result.external_id);
  _renderGrid();

  let resp;
  try {
    resp = await fetch(`/api/media/pin/${encodeURIComponent(fileId)}`, {
      method: 'DELETE',
    });
  } catch (err) {
    showToast(`Remove failed: ${err.message || 'network error'}`, 'error', 4000);
    _state.pinning.delete(result.external_id);
    _renderGrid();
    return;
  }

  _state.pinning.delete(result.external_id);

  if (!resp.ok) {
    showToast('Failed to remove from library', 'error', 4000);
    _renderGrid();
    return;
  }

  for (const r of _state.results) {
    if (r.external_id === result.external_id) {
      r.pinned = false;
      r.pinned_file_id = null;
    }
  }
  window.dispatchEvent(new CustomEvent('media-servers:changed'));
  _renderGrid();
}

// --- Fallback cover generator ---------------------------------------

// Cache generated covers in memory so re-renders don't redraw.
const _fallbackCache = new Map();

function _fallbackCoverDataUrl(title, author, seed) {
  const key = `${seed}|${title}|${author}`;
  if (_fallbackCache.has(key)) return _fallbackCache.get(key);

  const canvas = document.createElement('canvas');
  canvas.width = 240;
  canvas.height = 360;
  const ctx = canvas.getContext('2d');

  // Deterministic hue from the seed so the same book always gets the
  // same color even across reloads.
  const hue = _hashToHue(seed || title || 'x');
  const grad = ctx.createLinearGradient(0, 0, 0, 360);
  grad.addColorStop(0, `hsl(${hue}, 45%, 32%)`);
  grad.addColorStop(1, `hsl(${(hue + 40) % 360}, 38%, 18%)`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 240, 360);

  // Subtle diagonal rule so it doesn't read as a solid color tile.
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (let y = -360; y < 360; y += 22) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(240, y + 240);
    ctx.stroke();
  }

  // Title + author, wrapped.
  ctx.fillStyle = 'rgba(255,255,255,0.96)';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  _drawWrapped(ctx, title || 'Untitled', 18, 140, 204, 22, 'bold 22px "Inter", system-ui, sans-serif');

  ctx.fillStyle = 'rgba(255,255,255,0.68)';
  _drawWrapped(ctx, author || 'Unknown', 18, 300, 204, 16, '14px "Inter", system-ui, sans-serif');

  const url = canvas.toDataURL('image/jpeg', 0.82);
  _fallbackCache.set(key, url);
  return url;
}

function _hashToHue(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) | 0;
  return Math.abs(h) % 360;
}

function _drawWrapped(ctx, text, x, y, maxWidth, lineHeight, font) {
  ctx.font = font;
  const words = String(text).split(/\s+/).filter(Boolean);
  let line = '';
  let cursorY = y;
  for (const w of words) {
    const test = line ? `${line} ${w}` : w;
    if (ctx.measureText(test).width > maxWidth && line) {
      ctx.fillText(line, x, cursorY);
      line = w;
      cursorY += lineHeight;
      if (cursorY > y + lineHeight * 4) {
        // Truncate after 4 lines to keep the cover readable.
        ctx.fillText(line + '…', x, cursorY);
        return;
      }
    } else {
      line = test;
    }
  }
  if (line) ctx.fillText(line, x, cursorY);
}

// --- Rendering --------------------------------------------------------

function _buildOverlay() {
  _overlay = document.createElement('div');
  _overlay.className = 'lv-overlay';
  _overlay.innerHTML = `
    <div class="lv-panel" role="dialog" aria-modal="true" aria-label="LibriVox Library">
      <div class="lv-header">
        <div>
          <div class="lv-title">LibriVox</div>
          <div class="lv-subtitle">Free public-domain audiobooks, read by volunteers.</div>
        </div>
        <button class="lv-close" title="Close (Esc)" aria-label="Close">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
      </div>
      <div class="lv-controls">
        <input type="search" id="lv-search" placeholder="Search title, author, or keyword…" autocomplete="off">
      </div>
      <div class="lv-chips" aria-label="Categories">
        <button class="lv-chip lv-chip-active" data-subject="">All</button>
        ${CATEGORY_CHIPS.map(c =>
          `<button class="lv-chip" data-subject="${escapeHtml(c.subject)}">${escapeHtml(c.label)}</button>`
        ).join('')}
      </div>
      <div class="lv-body" id="lv-grid"></div>
    </div>
  `;
  document.body.appendChild(_overlay);

  _overlay.addEventListener('click', (e) => {
    if (e.target === _overlay) closeLibrivoxBrowse();
  });
  _overlay.querySelector('.lv-close').addEventListener('click', closeLibrivoxBrowse);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _overlay?.classList.contains('visible')) {
      closeLibrivoxBrowse();
    }
  });

  // Debounced search — 300ms is the sweet spot between feeling instant
  // and not hammering archive.org on every keystroke. Search is
  // independent from the category chip: you can type "Lovecraft" while
  // Horror is selected and get Lovecraft-horror books.
  const searchEl = _overlay.querySelector('#lv-search');
  let searchTimer = null;
  searchEl.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      _state.query = searchEl.value.trim();
      // Typing exits recent-mode. When the query is cleared back to
      // empty we stay in regular browse (archive.org default) rather
      // than bouncing back into recent — avoids a jarring refetch
      // on backspace-to-empty.
      _state.recent = false;
      _loadPage({ reset: true });
    }, 300);
  });

  _overlay.querySelectorAll('[data-subject]').forEach(btn => {
    btn.addEventListener('click', () => {
      const subject = btn.dataset.subject || '';
      _state.category = subject;
      // Picking a chip also exits recent-mode. The "All" chip (empty
      // subject) without a query would otherwise be ambiguous — treat
      // it as "all, archive.org default" not "recent".
      _state.recent = false;
      _overlay.querySelectorAll('.lv-chip').forEach(c => c.classList.toggle(
        'lv-chip-active', c === btn,
      ));
      _loadPage({ reset: true });
    });
  });
}

function _renderGrid() {
  const grid = _overlay?.querySelector('#lv-grid');
  if (!grid) return;

  if (_state.loading && _state.results.length === 0) {
    grid.innerHTML = `
      <div class="lv-empty">
        <div class="lv-loading-dot"></div>
        <div class="lv-empty-title">Searching LibriVox…</div>
      </div>
    `;
    return;
  }

  if (!_state.loading && _state.results.length === 0) {
    const parts = [];
    if (_state.query)    parts.push(`matching "${escapeHtml(_state.query)}"`);
    if (_state.category) parts.push(`in ${escapeHtml(_state.category)}`);
    const hint = parts.length
      ? `No results ${parts.join(' ')}. Try a different term or category.`
      : `Archive.org didn't return any results — try a different search.`;
    grid.innerHTML = `
      <div class="lv-empty">
        <div class="lv-empty-title">Nothing found</div>
        <div class="lv-empty-sub">${hint}</div>
      </div>
    `;
    return;
  }

  const cards = _state.results.map(r => _cardHtml(r)).join('');
  const moreBtn = _state.hasMore
    ? `<div class="lv-more"><button class="btn" id="lv-load-more">Load more</button></div>`
    : '';
  // "Recently added" subheading only renders while recent-mode is
  // active (landing state, before any search/category). The label
  // disappears once the user narrows the view — at that point the
  // grid is a search result, not a feed, and a "recently added"
  // label would be misleading.
  const sectionHeader = _state.recent
    ? `<div class="lv-section-header">
         <span class="lv-section-title">Recently added on LibriVox</span>
         <span class="lv-section-sub">New arrivals from the last 30 days</span>
       </div>`
    : '';
  grid.innerHTML = `
    ${sectionHeader}
    <div class="lv-grid">${cards}</div>
    ${moreBtn}
  `;

  // Wire action buttons.
  grid.querySelectorAll('[data-pin-id]').forEach(btn => {
    btn.addEventListener('click', () => {
      const r = _state.results.find(x => x.external_id === btn.dataset.pinId);
      if (!r) return;
      if (r.pinned) _unpinItem(r);
      else _pinItem(r);
    });
  });
  grid.querySelectorAll('.lv-cover-img').forEach(img => {
    // Swap to fallback on error so archive.org placeholder images don't
    // leave a grey square behind.
    img.addEventListener('error', () => {
      const title = img.dataset.title || '';
      const author = img.dataset.author || '';
      const seed = img.dataset.seed || '';
      img.src = _fallbackCoverDataUrl(title, author, seed);
      img.onerror = null;
    });
  });
  const moreEl = grid.querySelector('#lv-load-more');
  if (moreEl) {
    moreEl.addEventListener('click', () => {
      _state.page += 1;
      _loadPage();
    });
  }
}

function _cardHtml(r) {
  const title = r.name || 'Untitled';
  const author = r.author || 'Unknown author';
  // Prefer LibriVox's human-formatted totaltime ("13:06:44") over
  // re-formatting duration_ms — their string handles edge cases better.
  const dur = (r.extra?.totaltime && r.extra.totaltime.trim())
    ? r.extra.totaltime.trim()
    : _fmtDuration((r.duration_ms || 0) / 1000);
  const sections = Number(r.extra?.num_sections || 0);
  const secLabel = sections > 0
    ? `${sections} ${sections === 1 ? 'section' : 'sections'}`
    : '';
  const year = (r.extra?.copyright_year || '').trim();
  const language = (r.extra?.language || '').trim();
  // Only surface non-English language as a chip; English is the default
  // and 90%+ of LibriVox, so tagging it would be noise.
  const nonEnglish = language && language.toLowerCase() !== 'english';
  const isPinning = _state.pinning.has(r.external_id);
  const isPinned = !!r.pinned;

  const actionLabel = isPinning
    ? (isPinned ? 'Removing…' : 'Adding…')
    : (isPinned ? '✓ In library' : '+ Library');
  const actionClass = isPinned ? 'lv-action-pinned' : 'lv-action-pin';

  const cover = r.cover_url || '';
  // data- attrs let the error handler rebuild a fallback without a closure.
  const coverHtml = cover
    ? `<img class="lv-cover-img"
              src="${escapeHtml(cover)}"
              alt=""
              loading="lazy"
              decoding="async"
              data-title="${escapeHtml(title)}"
              data-author="${escapeHtml(author)}"
              data-seed="${escapeHtml(r.external_id || '')}">`
    : `<img class="lv-cover-img"
              src="${_fallbackCoverDataUrl(title, author, r.external_id || '')}"
              alt=""
              data-title="${escapeHtml(title)}"
              data-author="${escapeHtml(author)}"
              data-seed="${escapeHtml(r.external_id || '')}">`;

  const topRightChips = [];
  if (year) {
    topRightChips.push(
      `<span class="lv-chip-small lv-chip-year" title="Published ${escapeHtml(year)}">${escapeHtml(year)}</span>`,
    );
  }
  if (nonEnglish) {
    topRightChips.push(
      `<span class="lv-chip-small lv-chip-lang" title="Language: ${escapeHtml(language)}">${escapeHtml(language)}</span>`,
    );
  }

  return `
    <article class="lv-card">
      <div class="lv-cover">
        ${coverHtml}
        <span class="lv-badge" title="Public domain — free to listen">PD</span>
        ${topRightChips.length ? `<div class="lv-chip-stack">${topRightChips.join('')}</div>` : ''}
      </div>
      <div class="lv-meta">
        <div class="lv-title-line" title="${escapeHtml(title)}">${escapeHtml(title)}</div>
        <div class="lv-author-line" title="${escapeHtml(author)}">${escapeHtml(author)}</div>
        <div class="lv-stats">
          ${dur ? `<span>${escapeHtml(dur)}</span>` : ''}
          ${dur && secLabel ? `<span aria-hidden="true">·</span>` : ''}
          ${secLabel ? `<span>${escapeHtml(secLabel)}</span>` : ''}
        </div>
      </div>
      <button class="lv-action ${actionClass}"
              data-pin-id="${escapeHtml(r.external_id)}"
              ${isPinning ? 'disabled' : ''}>
        ${escapeHtml(actionLabel)}
      </button>
    </article>
  `;
}

function _fmtDuration(seconds) {
  if (!seconds || seconds <= 0) return '';
  const s = Math.round(Number(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}
