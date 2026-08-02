/**
 * Read-along overlay for Gutenberg plaintext.
 *
 * One overlay exists at a time. Opening calls GET /api/media/gutenberg-text/
 * {file_id}; the handler returns the full body (typically 200-2000 KB).
 * We paragraph-wrap on double-newlines, render with word-wrap + a reader-
 * friendly serif, and leave scroll position to the browser.
 *
 * Future: audio-sync highlighting, bookmarks, chapter jumps. For now the
 * overlay is deliberately minimal — font-size control + close. Anything
 * more would need a real text↔audio alignment pipeline which we haven't
 * built.
 */

import { escapeHtml, showToast } from '../app.js';

const FONT_SIZES = [15, 17, 19, 22, 26];
const FONT_STORAGE_KEY = 'augmentum.readAlong.fontIndex';

let _overlay = null;
let _escHandler = null;

function _loadFontIndex() {
  const raw = localStorage.getItem(FONT_STORAGE_KEY);
  const n = raw === null ? 2 : parseInt(raw, 10);
  return Number.isFinite(n) && n >= 0 && n < FONT_SIZES.length ? n : 2;
}

function _saveFontIndex(n) {
  try {
    localStorage.setItem(FONT_STORAGE_KEY, String(n));
  } catch {
    // quota / privacy mode — drop silently, not worth warning for a pref
  }
}

function _paragraphsHtml(raw) {
  // Gutenberg plaintext separates paragraphs with blank lines. Splitting
  // on \n\n+ gets us logical paragraphs; line-within-paragraph breaks are
  // preserved so poetry and verse keep their shape.
  if (!raw) return '';
  return raw
    .split(/\n{2,}/)
    .map(p => p.trim())
    .filter(Boolean)
    .map(p => `<p class="read-along-p">${escapeHtml(p).replace(/\n/g, '<br>')}</p>`)
    .join('');
}

function _shellHtml(title) {
  return `
    <div class="read-along-backdrop" data-readalong-close></div>
    <div class="read-along-panel" role="dialog" aria-modal="true" aria-label="${escapeHtml(title)}">
      <header class="read-along-header">
        <div class="read-along-title" title="${escapeHtml(title)}">${escapeHtml(title)}</div>
        <div class="read-along-controls">
          <button type="button" class="read-along-fontbtn" data-readalong-font="-" title="Decrease font size" aria-label="Decrease font size">A-</button>
          <button type="button" class="read-along-fontbtn" data-readalong-font="+" title="Increase font size" aria-label="Increase font size">A+</button>
          <button type="button" class="read-along-close" data-readalong-close title="Close (Esc)" aria-label="Close">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
          </button>
        </div>
      </header>
      <div class="read-along-scroll" data-readalong-scroll>
        <div class="read-along-body" data-readalong-body>
          <div class="read-along-loading">
            <svg class="read-along-spin" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
            <span>Loading text…</span>
          </div>
        </div>
      </div>
    </div>
  `;
}

function _applyFontIndex(idx) {
  if (!_overlay) return;
  const panel = _overlay.querySelector('.read-along-panel');
  if (panel) panel.style.setProperty('--read-along-font-size', `${FONT_SIZES[idx]}px`);
}

function _close() {
  if (!_overlay) return;
  _overlay.remove();
  _overlay = null;
  if (_escHandler) {
    document.removeEventListener('keydown', _escHandler);
    _escHandler = null;
  }
  document.body.classList.remove('read-along-open');
}

function _wireOverlayEvents() {
  if (!_overlay) return;
  let fontIdx = _loadFontIndex();
  _applyFontIndex(fontIdx);

  _overlay.addEventListener('click', (e) => {
    const closeEl = e.target.closest('[data-readalong-close]');
    if (closeEl) { _close(); return; }
    const fontEl = e.target.closest('[data-readalong-font]');
    if (fontEl) {
      const dir = fontEl.dataset.readalongFont;
      if (dir === '+') fontIdx = Math.min(FONT_SIZES.length - 1, fontIdx + 1);
      else if (dir === '-') fontIdx = Math.max(0, fontIdx - 1);
      _applyFontIndex(fontIdx);
      _saveFontIndex(fontIdx);
    }
  });

  _escHandler = (e) => {
    if (e.key === 'Escape') { _close(); }
  };
  document.addEventListener('keydown', _escHandler);
}

async function _fetchAndRender(fileId) {
  const body = _overlay?.querySelector('[data-readalong-body]');
  if (!body) return;

  let resp;
  try {
    resp = await fetch(`/api/media/gutenberg-text/${encodeURIComponent(fileId)}`, {
      headers: { 'Accept': 'text/plain' },
    });
  } catch (err) {
    body.innerHTML = `<div class="read-along-error">Could not load text: ${escapeHtml(String(err.message || err))}</div>`;
    return;
  }

  if (resp.status === 202) {
    // Job still running — the outer panel already shows "Fetching…",
    // but someone could have clicked through before the button was
    // suppressed. Give an explicit message rather than a blank body.
    body.innerHTML = `
      <div class="read-along-pending">
        <p>Still fetching the Project Gutenberg text…</p>
        <p class="read-along-hint">Close this and reopen in a few seconds, or watch the "Fetching text…" chip in the book detail.</p>
      </div>`;
    return;
  }
  if (resp.status === 410) {
    const info = await resp.json().catch(() => ({}));
    body.innerHTML = `<div class="read-along-error">No text available for this recording${info?.reason ? `: ${escapeHtml(info.reason)}` : ''}.</div>`;
    return;
  }
  if (!resp.ok) {
    body.innerHTML = `<div class="read-along-error">Could not load text (HTTP ${escapeHtml(String(resp.status))}).</div>`;
    return;
  }

  const raw = await resp.text();
  const html = _paragraphsHtml(raw);
  if (!html) {
    body.innerHTML = `<div class="read-along-error">The fetched text was empty.</div>`;
    return;
  }
  body.innerHTML = html;
  // Restore focus into the scroll region so PageDown/PageUp work right
  // away without a click.
  _overlay?.querySelector('[data-readalong-scroll]')?.focus({ preventScroll: true });
}

export function openReadAlong(fileId, title) {
  if (_overlay) _close();
  // Companion presence: this book is now "what I'm reading".
  import('../architect-observer.js')
    .then(m => m.reportAttention('surface.media.reading_started', {
      label: title || '',
      kind: 'book',
      ref: String(fileId || ''),
    }))
    .catch(() => {});
  const host = document.createElement('div');
  host.className = 'read-along-overlay';
  host.setAttribute('data-readalong-root', '');
  host.innerHTML = _shellHtml(title || 'Read along');
  document.body.appendChild(host);
  document.body.classList.add('read-along-open');
  _overlay = host;
  _wireOverlayEvents();
  _fetchAndRender(fileId).catch(err => {
    showToast(`Could not open read-along: ${err.message || err}`, 'error');
    _close();
  });
}
