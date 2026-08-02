/* ==========================================================================
   Browse Module — AI-Assisted Web Reader + Notes
   Search, fetch, read articles, AI sidebar analysis, Milkdown notes editor
   ========================================================================== */

import { escapeHtml, extractErrorMessage, showToast, app } from './app.js';
import { getCurrentUser } from './auth.js';
import { getSettings, save as saveSettings } from './settings.js';
import * as NoteEditor from './note-editor.js';
import * as SlashMenu from './note-slash-menu.js';
import { createNotesEditor, prefetchNotesEditor } from './notes-editor.js';
import { SurfaceRegistry } from './surface-registry.js';
import { LayoutManager } from './layout-manager.js';
import { ViewStack } from './view-stack.js';
import { renderMathIn } from './math-renderer.js';
import { copyToClipboard } from './clipboard.js';
import { mountCastButton } from './cast-button.js';
import { renderMarkdown, highlightCodeDeferred } from './chat/markdown.js';
import { makeStreamRenderer } from './chat/stream-render.js';

/**
 * Mirror the current page's url + title onto the BrowseSurface (if browse
 * is mounted as a tab). This is how the surface tab's title updates to
 * match the page, and how workspace restore knows which URL to re-open.
 * Safe to call when there's no BrowseSurface — a no-op in that case.
 */
function _syncBrowseSurface(url, title) {
  const surface = SurfaceRegistry.ofType('browse')[0];
  if (!surface) return;
  if (url) surface.url = url;
  if (title) surface.pageTitle = title;
  surface.emit('surface:titleChanged', { title: surface.getTitle() });
}

/**
 * Syntax-highlight every <pre><code> block inside a container, deferred
 * to idle time so the initial paint isn't blocked. Uses the global hljs
 * (vendored via lib/highlight.js). Works on both language-classed
 * blocks (GitHub rendered READMEs set class="language-js") and
 * unclassed ones (Stack Exchange wraps inline code without a language
 * hint — hljs auto-detects). Idempotent via the dataset.highlighted
 * marker hljs sets itself.
 */
function _highlightArticleCodeDeferred(container) {
  const run = () => {
    if (typeof hljs === 'undefined' || !container) return;
    container.querySelectorAll('pre code').forEach(block => {
      if (block.dataset.highlighted) return;
      try { hljs.highlightElement(block); } catch { /* unsupported lang */ }
    });
  };
  if (typeof window.requestIdleCallback === 'function') {
    window.requestIdleCallback(run, { timeout: 500 });
  } else {
    setTimeout(run, 0);
  }
}

/**
 * Slug-ify a heading's text for use as an anchor id. Lowercases,
 * replaces non-word chars with hyphens, collapses repeats. Deduplicates
 * by appending a counter when the same slug appears twice in one doc.
 */
function _slugifyHeading(text, existing) {
  let slug = (text || '').toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .slice(0, 60);
  if (!slug) slug = 'section';
  if (!existing.has(slug)) { existing.add(slug); return slug; }
  let n = 2;
  while (existing.has(`${slug}-${n}`)) n++;
  const unique = `${slug}-${n}`;
  existing.add(unique);
  return unique;
}

/**
 * Build a sticky table-of-contents panel next to long articles. Walks
 * h1/h2/h3 inside the article body, assigns stable anchor ids, and
 * renders a nav list. Hidden (strip detached) when fewer than 3
 * meaningful headings exist — short pages don't need one. Re-built on
 * every article render.
 */
function _buildArticleToc(articleBody) {
  state.articleToc = [];
  const tocBtn = dom.readerView?.querySelector('[data-article-action="toc"]');
  if (tocBtn) tocBtn.style.display = 'none';
  if (!articleBody) return;

  const headings = articleBody.querySelectorAll('h1, h2, h3');
  // Filter out headings that look like utility/noise (single-word
  // nav headings, hidden section markers). Cheap heuristic: minimum
  // text length + not already aria-hidden.
  const usable = [];
  const slugs = new Set();
  for (const h of headings) {
    if (h.closest('[aria-hidden="true"]')) continue;
    const text = h.textContent.trim();
    if (text.length < 2) continue;
    // Assign an id if GitHub/MDN didn't already.
    if (!h.id) h.id = _slugifyHeading(text, slugs);
    else slugs.add(h.id);
    usable.push({ level: h.tagName, id: h.id, text });
  }
  if (usable.length < 3) return;

  state.articleToc = usable;
  if (tocBtn) tocBtn.style.display = '';
}

/**
 * Decorate every <pre> in an article body with a small copy button that
 * writes the block's text to the clipboard. Idempotent — skips <pre>s
 * already marked. Uses navigator.clipboard; falls back to a hidden
 * textarea + execCommand('copy') where the modern API is unavailable
 * (older Safari, some insecure contexts).
 */
function _addCopyButtonsToCodeBlocks(container) {
  if (!container) return;
  container.querySelectorAll('pre:not([data-has-copy-btn])').forEach(pre => {
    pre.dataset.hasCopyBtn = '1';
    // Make pre a positioning context for the floating button; avoid
    // breaking sites that already style pre.
    if (getComputedStyle(pre).position === 'static') {
      pre.style.position = 'relative';
    }
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'browse-code-copy-btn';
    btn.title = 'Copy code';
    btn.setAttribute('aria-label', 'Copy code');
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
      </svg>
    `;
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const text = pre.querySelector('code')?.innerText || pre.innerText || '';
      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(text);
        } else {
          const ta = document.createElement('textarea');
          ta.value = text;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          try { document.execCommand('copy'); } finally { ta.remove(); }
        }
        btn.classList.add('copied');
        btn.title = 'Copied!';
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.title = 'Copy code';
        }, 1500);
      } catch { /* ignore — user can still select manually */ }
    });
    pre.appendChild(btn);
  });
}

function _resolveArticleLinkTarget(href) {
  const raw = String(href || '').trim();
  if (!raw || raw.startsWith('#')) return null;
  if (/^(?:javascript|data):/i.test(raw)) return null;

  try {
    const appUrl = new URL(raw, window.location.origin);
    if (appUrl.origin === window.location.origin && appUrl.pathname === '/api/browse/fetch') {
      const proxied = appUrl.searchParams.get('url');
      if (proxied) return proxied;
    }
  } catch { /* malformed URL — caller falls back to the raw string */ }

  if (/^(?:mailto|tel):/i.test(raw)) return null;

  try {
    const base = /^https?:\/\//i.test(state.currentUrl || '')
      ? state.currentUrl
      : window.location.href;
    const resolved = new URL(raw, base).href;
    return /^https?:\/\//i.test(resolved) ? resolved : null;
  } catch {
    return null;
  }
}

function _openArticleLink(targetUrl, event = null) {
  if (!targetUrl) return;
  const mode = getSettings().browseLinkOpenMode || 'current';
  if (event?.ctrlKey || event?.metaKey || event?.shiftKey || mode === 'reader-tab') {
    openNewPageTab(targetUrl);
    return;
  }
  if (mode === 'external') {
    window.open(targetUrl, '_blank', 'noopener,noreferrer');
    return;
  }
  browseFetch(targetUrl);
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  panelOpen: false,
  activeTab: 'browse', // 'browse' | 'notes' — outer view tab, NOT page tab
  // Browse state — active page tab's live data; snapshots of non-active
  // tabs live in state.pageTabs[i].snapshot. Keeping the old field names
  // means every existing function that reads state.history / state.currentUrl
  // / etc. continues working without change; switching tabs just swaps
  // these fields in from another tab's snapshot.
  history: [],       // [{url, title, scrollPos}]
  historyIdx: -1,    // -1 = search view
  currentUrl: null,
  currentContent: null,  // plain text for AI actions
  currentHtml: null,
  currentVideos: [],     // detected video embeds with transcripts
  currentPageType: 'article', // detected content type for AI prompt selection
  _videoAskContext: null, // one-shot video context for ask bar
  // Page-tab registry. Always has ≥1 tab after init. Single-tab state
  // keeps the strip hidden so users who never open a second tab don't
  // see UI they don't need. Tabs persist to localStorage; full HTML is
  // not persisted so the first switch after reload re-fetches.
  pageTabs: [],
  activePageTabIdx: 0,
  searchQuery: '',
  searchResults: null,
  activeCategory: 'general',
  // Active provider chip (id matches _PROVIDER_CHIPS). Null = no scoping.
  // Persisted so the user's last provider carries across sessions.
  activeProvider: (() => {
    try { return localStorage.getItem('augmentum_browse_provider') || null; }
    catch { return null; }
  })(),
  // Video-surface filters — only meaningful when the active category is
  // videos (or every result is from a known video host). Persisted so
  // a user's "Past month / Newest" preference carries between sessions.
  // Empty string = filter off / default sort = relevance.
  videoTimeRange: (() => {
    try { return localStorage.getItem('augmentum_browse_video_timerange') || ''; }
    catch { return ''; }
  })(),
  videoSortBy: (() => {
    try { return localStorage.getItem('augmentum_browse_video_sort') || ''; }
    catch { return ''; }
  })(),
  videoDuration: (() => {
    try { return localStorage.getItem('augmentum_browse_video_duration') || ''; }
    catch { return ''; }
  })(),
  // Reader typography prefs — populated lazily on first popover open to
  // avoid touching localStorage on module load for users who never
  // open the reader. _loadReaderPrefs fills in from storage + defaults.
  readerPrefs: null,
  // Headings for the current article's TOC popover. Filled by
  // _buildArticleToc after each renderArticle; consumed by
  // _openTocPopover. Empty array = no toggle button shown.
  articleToc: [],
  // Split mode preference is persisted across sessions; restored on
  // panel open if the viewport is wide enough (>=1201px). Below that
  // breakpoint split has no CSS support and is force-disabled.
  splitMode: (() => {
    try { return localStorage.getItem('augmentum_browse_split') === '1'; }
    catch { return false; }
  })(),
  notesHistoryCollapsed: (() => {
    try { return localStorage.getItem('augmentum_browse_notes_history_collapsed') === '1'; }
    catch { return false; }
  })(),
  aiAbort: null,      // AbortController
  // Notes state
  notes: [],          // metadata stubs from server
  activeNoteId: null,
  activeNote: null,   // full note object
  milkdownEditor: null,
  milkdownLoaded: false,
  milkdownLoading: false,
  noteSaveTimer: null, // debounce timer
};

// ---------------------------------------------------------------------------
// Page tabs — browser-style tabs inside the browse reader.
//
// Storage model: the ACTIVE tab's data lives in the flat state.history /
// state.currentUrl / etc. fields (so existing code keeps working). Each
// other tab stores a `snapshot` copy of those fields. On switch we save
// the active fields back to the outgoing tab then load the incoming
// tab's snapshot into the same fields.
// ---------------------------------------------------------------------------
const _PAGE_TABS_STORAGE_KEY = 'augmentum_browse_page_tabs';

function _newBlankPageTab() {
  return {
    id: 't_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    title: 'New Tab',
    favicon: '',
    url: '',
    snapshot: null,
  };
}

function _activePageTab() {
  return state.pageTabs[state.activePageTabIdx] || null;
}

function _canAutoFocusEditable() {
  return !document.documentElement.classList.contains('touch-keyboard');
}

function _autoFocusSearchInput({ select = false } = {}) {
  if (!_canAutoFocusEditable() || !dom.searchInput) return;
  try { dom.searchInput.focus({ preventScroll: true }); }
  catch { dom.searchInput.focus(); }
  if (select) dom.searchInput.select();
}

function _faviconForUrl(url) {
  try {
    const host = new URL(url).hostname;
    return `/api/browse/image?url=${encodeURIComponent(`https://www.google.com/s2/favicons?domain=${host}&sz=32`)}`;
  } catch { return ''; }
}

/** First significant letter of a pack name, used as the favicon fallback
 * glyph. Skips leading "the/a/an" articles so "The Wikipedia" yields "W"
 * rather than "T". Returns "?" if no letter is available. */
function _packFirstLetter(name) {
  const trimmed = String(name || '').replace(/^(?:the|a|an)\s+/i, '').trim();
  const letter = trimmed.charAt(0);
  return (letter || '?').toUpperCase();
}

/** Emit a per-pack favicon ``<img>`` for a ZIM pack. The image hits our
 * own ``/_illustration`` endpoint, which 404s gracefully when the pack
 * lacks an illustration entry (common on older ZIMs). The companion
 * ``_attachPackFaviconFallbacks`` swaps a letter pill in place of the
 * broken img — never leave the broken-image icon visible.
 *
 * ``size`` matches the libzim illustration size (48 for the landing
 * grid, 96 for the active-pack header so it stays crisp at 2x DPI).
 * The rendered footprint is governed by CSS (.browse-pack-favicon /
 * .browse-pack-favicon.large); the inline width/height attrs prevent
 * pre-load reflow without dictating final display size.
 */
function _packFaviconHtml(packId, name, size = 48, extraClass = '') {
  const letter = _packFirstLetter(name);
  const cls = `browse-pack-favicon${extraClass ? ` ${extraClass}` : ''}`;
  return `<img class="${cls}"`
    + ` src="/api/knowledge/zim/${encodeURIComponent(packId)}/_illustration?size=${size}"`
    + ` width="${size}" height="${size}"`
    + ` data-fallback-letter="${escapeHtml(letter)}"`
    + ` alt="" loading="lazy">`;
}

/** Format a ZIM ``Counter`` metadata dict as a compact stat line:
 * ``"1.5M articles · 523k images · 12k other"``. Buckets MIME types
 * into articles (text/html), images, audio, video, other. Skips empty
 * buckets. Uses Intl.NumberFormat for the SI suffix so 1500000 → "1.5M".
 */
function _formatPackCounterCompact(counter) {
  if (!counter || typeof counter !== 'object') return '';
  const buckets = { articles: 0, images: 0, audio: 0, video: 0, other: 0 };
  for (const [mime, count] of Object.entries(counter)) {
    if (typeof count !== 'number') continue;
    if (mime.startsWith('text/html')) buckets.articles += count;
    else if (mime.startsWith('image/')) buckets.images += count;
    else if (mime.startsWith('audio/')) buckets.audio += count;
    else if (mime.startsWith('video/')) buckets.video += count;
    else buckets.other += count;
  }
  const fmt = new Intl.NumberFormat(undefined, {
    notation: 'compact', maximumFractionDigits: 1,
  });
  const parts = [];
  if (buckets.articles) parts.push(`${fmt.format(buckets.articles)} articles`);
  if (buckets.images) parts.push(`${fmt.format(buckets.images)} images`);
  if (buckets.audio) parts.push(`${fmt.format(buckets.audio)} audio`);
  if (buckets.video) parts.push(`${fmt.format(buckets.video)} video`);
  if (buckets.other) parts.push(`${fmt.format(buckets.other)} other`);
  return parts.join(' · ');
}

/** Format a ZIM ``Date`` metadata string as a friendly "Mar 2025" /
 * "2024-12-03" — preserves source granularity. Empty / unparseable
 * input passes through verbatim. */
function _formatPackDate(raw) {
  if (!raw) return '';
  const s = String(raw).trim();
  if (!s) return '';
  // ISO month: 2024-03
  if (/^\d{4}-\d{2}$/.test(s)) {
    try {
      const d = new Date(`${s}-01T00:00:00Z`);
      return d.toLocaleDateString(undefined, { month: 'long', year: 'numeric' });
    } catch { return s; }
  }
  // Full ISO date
  if (/^\d{4}-\d{2}-\d{2}/.test(s)) {
    try {
      const d = new Date(s);
      return d.toLocaleDateString(undefined, {
        year: 'numeric', month: 'short', day: 'numeric',
      });
    } catch { return s; }
  }
  return s;
}

/** Fetch and render the per-pack metadata sidebar. Idempotent — subsequent
 * opens skip the fetch and reuse the cached HTML. Errors render an inline
 * "Couldn't load metadata" notice rather than throwing.
 */
async function _renderZimMetaPanel(panel, packId, name) {
  if (!panel) return;
  // Cached state on the element itself so we don't refetch on toggle.
  if (panel.dataset.loaded === '1') return;
  panel.dataset.loaded = '1';

  panel.innerHTML = `
    <div class="browse-zim-meta-loading">
      <div class="browse-zim-meta-skeleton"></div>
      <div class="browse-zim-meta-skeleton"></div>
      <div class="browse-zim-meta-skeleton short"></div>
    </div>
  `;

  let data = null;
  try {
    const resp = await fetch(`/api/knowledge/zim/${encodeURIComponent(packId)}/_meta`);
    if (resp.ok) data = await resp.json();
  } catch { /* fall through to error state */ }

  if (!data) {
    panel.innerHTML = `
      <div class="browse-zim-meta-error">Couldn't load metadata for this pack.</div>
    `;
    panel.dataset.loaded = '0';  // allow retry
    return;
  }

  const headerName = escapeHtml(data.Name || data.Title || name || packId);
  const desc = data.Description || '';
  const lang = data.language && data.language.code ? data.language : null;
  const creator = data.Creator || '';
  const publisher = data.Publisher || '';
  const dateStr = _formatPackDate(data.Date);
  const flavour = data.Flavour || '';
  const tags = Array.isArray(data.tags) ? data.tags : [];
  const counter = _formatPackCounterCompact(data.Counter);

  const langBadge = lang
    ? `<span class="browse-zim-meta-lang" title="${escapeHtml(lang.code)}">${escapeHtml(lang.name || lang.code)}</span>`
    : '';

  const attrLines = [];
  if (creator) attrLines.push(`<dt>Creator</dt><dd>${escapeHtml(creator)}</dd>`);
  if (publisher) attrLines.push(`<dt>Publisher</dt><dd>${escapeHtml(publisher)}</dd>`);
  if (dateStr) attrLines.push(`<dt>Date</dt><dd>${escapeHtml(dateStr)}</dd>`);
  if (flavour) attrLines.push(`<dt>Flavour</dt><dd>${escapeHtml(flavour)}</dd>`);

  panel.innerHTML = `
    <div class="browse-zim-meta-header">
      ${_packFaviconHtml(packId, headerName, 96, 'large')}
      <div class="browse-zim-meta-identity">
        <h3 class="browse-zim-meta-name">${headerName}</h3>
        ${langBadge}
      </div>
    </div>
    ${desc ? `<p class="browse-zim-meta-desc">${escapeHtml(desc)}</p>` : ''}
    ${attrLines.length ? `<dl class="browse-zim-meta-attr">${attrLines.join('')}</dl>` : ''}
    ${tags.length ? `
      <div class="browse-zim-meta-section">
        <div class="browse-zim-meta-label">Tags</div>
        <div class="browse-zim-meta-tags">
          ${tags.map(t => `<span class="browse-zim-meta-tag">${escapeHtml(t)}</span>`).join('')}
        </div>
      </div>
    ` : ''}
    ${counter ? `
      <div class="browse-zim-meta-section">
        <div class="browse-zim-meta-label">Contents</div>
        <div class="browse-zim-meta-counter">${escapeHtml(counter)}</div>
      </div>
    ` : ''}
  `;
  _attachPackFaviconFallbacks(panel);
}

/** Bind a one-shot ``error`` handler on every ``.browse-pack-favicon``
 * under ``scope`` (defaults to document) that swaps the broken img for
 * a letter-pill span. Idempotent — re-running on a node tree skips
 * already-bound imgs via ``data-fallback-bound``.
 *
 * The letter pill is purely CSS-styled (.browse-pack-favicon-fallback);
 * the markup carries only the letter and the size class. Re-runs after
 * any render path that emits ``_packFaviconHtml``.
 */
function _attachPackFaviconFallbacks(scope) {
  const root = scope || document;
  root.querySelectorAll('.browse-pack-favicon').forEach(img => {
    if (img.dataset.fallbackBound === '1') return;
    img.dataset.fallbackBound = '1';
    img.addEventListener('error', () => {
      const letter = img.dataset.fallbackLetter || '?';
      const span = document.createElement('span');
      // Mirror sizing classes (`.large`) so the fallback occupies the same
      // footprint as the missing img — no layout pop on swap.
      const sizeClass = img.classList.contains('large') ? ' large' : '';
      span.className = `browse-pack-favicon-fallback${sizeClass}`;
      span.textContent = letter;
      span.setAttribute('aria-hidden', 'true');
      img.replaceWith(span);
    }, { once: true });
  });
}

function _savePageTabSnapshot() {
  const t = _activePageTab();
  if (!t) return;
  t.snapshot = {
    history: state.history,
    historyIdx: state.historyIdx,
    currentUrl: state.currentUrl,
    currentContent: state.currentContent,
    currentHtml: state.currentHtml,
    currentVideos: state.currentVideos,
    currentPageType: state.currentPageType,
    scrollPos: dom.contentArea?.scrollTop || 0,
  };
  const entry = state.history[state.historyIdx];
  t.url = state.currentUrl || '';
  t.title = entry?.title || t.url || 'New Tab';
  t.favicon = t.url ? _faviconForUrl(t.url) : '';
}

function _loadPageTabSnapshot() {
  const t = _activePageTab();
  if (!t) return;
  const s = t.snapshot || {};
  state.history = s.history || [];
  state.historyIdx = (s.historyIdx !== undefined && s.historyIdx !== null) ? s.historyIdx : -1;
  state.currentUrl = s.currentUrl || null;
  state.currentContent = s.currentContent || null;
  state.currentHtml = s.currentHtml || null;
  state.currentVideos = s.currentVideos || [];
  state.currentPageType = s.currentPageType || 'article';
}

function _renderCachedPageTab() {
  const t = _activePageTab();
  if (!t) return false;
  const s = t.snapshot;
  if (!s || !s.currentHtml) return false;
  // Reassemble the minimum data shape renderArticle expects. Missing
  // fields (word count, sitename, favicon) degrade gracefully — the
  // article stays readable.
  renderArticle({
    html: s.currentHtml,
    text: s.currentContent || '',
    url: s.currentUrl,
    title: (s.history || [])[s.historyIdx]?.title || t.title,
    videos: s.currentVideos || [],
    page_type: s.currentPageType,
  });
  updateNavButtons();
  if (dom.searchInput) dom.searchInput.value = s.currentUrl || '';
  // Restore scroll position once the browser has laid out the content.
  const sp = s.scrollPos || 0;
  if (sp && dom.contentArea) {
    requestAnimationFrame(() => {
      if (dom.contentArea) dom.contentArea.scrollTop = sp;
    });
  }
  return true;
}

function openNewPageTab(url = '') {
  _savePageTabSnapshot();
  const tab = _newBlankPageTab();
  state.pageTabs.push(tab);
  state.activePageTabIdx = state.pageTabs.length - 1;
  _loadPageTabSnapshot();  // resets state.history etc. to blank
  if (url) {
    tab.url = url;
    browseFetch(url);
  } else {
    showSearchView();
    if (dom.searchInput) {
      dom.searchInput.value = '';
      _autoFocusSearchInput();
    }
  }
  _renderPageTabStrip();
  _savePageTabsToStorage();
}

function switchToPageTab(idx) {
  if (idx < 0 || idx >= state.pageTabs.length) return;
  if (idx === state.activePageTabIdx) return;
  _savePageTabSnapshot();
  state.activePageTabIdx = idx;
  _loadPageTabSnapshot();
  const t = _activePageTab();
  if (t.snapshot && t.snapshot.currentHtml) {
    _renderCachedPageTab();
  } else if (t.url) {
    // Tab restored from storage without a cached body — refetch fresh.
    browseFetch(t.url);
  } else {
    showSearchView();
    if (dom.searchInput) dom.searchInput.value = '';
  }
  _renderPageTabStrip();
  _savePageTabsToStorage();
}

function closePageTab(idx) {
  if (idx < 0 || idx >= state.pageTabs.length) return;
  if (state.pageTabs.length === 1) {
    // Last tab — clear rather than remove so we always have one tab to
    // render into. Matches Chrome's behaviour (close last tab → blank
    // new tab, not window close).
    state.pageTabs[0] = _newBlankPageTab();
    state.activePageTabIdx = 0;
    _loadPageTabSnapshot();
    showSearchView();
    if (dom.searchInput) dom.searchInput.value = '';
    _renderPageTabStrip();
    _savePageTabsToStorage();
    return;
  }
  const wasActive = idx === state.activePageTabIdx;
  state.pageTabs.splice(idx, 1);
  if (wasActive) {
    state.activePageTabIdx = Math.min(idx, state.pageTabs.length - 1);
    _loadPageTabSnapshot();
    const t = _activePageTab();
    if (t.snapshot && t.snapshot.currentHtml) {
      _renderCachedPageTab();
    } else if (t.url) {
      browseFetch(t.url);
    } else {
      showSearchView();
      if (dom.searchInput) dom.searchInput.value = '';
    }
  } else if (state.activePageTabIdx > idx) {
    state.activePageTabIdx -= 1;
  }
  _renderPageTabStrip();
  _savePageTabsToStorage();
}

function _savePageTabsToStorage() {
  try {
    // Persist only the identity + URL + title. Full HTML bodies are
    // recovered via re-fetch on next open; persisting them would bloat
    // localStorage past its 5 MB ceiling very quickly on heavy users.
    const lite = state.pageTabs.map(t => ({
      id: t.id,
      title: t.title || 'New Tab',
      favicon: t.favicon || '',
      url: t.url || '',
    }));
    localStorage.setItem(_PAGE_TABS_STORAGE_KEY, JSON.stringify({
      tabs: lite,
      active: state.activePageTabIdx,
    }));
  } catch { /* quota or disabled — silent */ }
}

function _loadPageTabsFromStorage() {
  try {
    const raw = localStorage.getItem(_PAGE_TABS_STORAGE_KEY);
    if (!raw) return false;
    const data = JSON.parse(raw);
    if (!Array.isArray(data.tabs) || data.tabs.length === 0) return false;
    state.pageTabs = data.tabs.map(t => ({
      id: t.id || ('t_' + Math.random().toString(36).slice(2, 10)),
      title: t.title || 'Tab',
      favicon: t.favicon || '',
      url: t.url || '',
      snapshot: null,  // re-fetched lazily on first switch-to
    }));
    state.activePageTabIdx = Math.max(
      0, Math.min(data.active || 0, state.pageTabs.length - 1),
    );
    return true;
  } catch { return false; }
}

function _renderPageTabStrip() {
  if (!dom.pageTabStrip) return;
  // Hide chrome when only one tab exists — Chrome-style. User's single-
  // tab flow looks identical to pre-tabs behaviour.
  const showStrip = state.pageTabs.length > 1;
  dom.pageTabStrip.classList.toggle('hidden', !showStrip);
  if (!showStrip) {
    dom.pageTabStrip.innerHTML = '';
    return;
  }
  const parts = state.pageTabs.map((t, i) => {
    const active = i === state.activePageTabIdx;
    const title = escapeHtml(t.title || 'New Tab');
    const favicon = t.favicon
      ? `<img class="browse-page-tab-favicon" src="${escapeHtml(t.favicon)}" alt="" loading="lazy" onerror="this.style.display='none'">`
      : '<span class="browse-page-tab-favicon-placeholder" aria-hidden="true">\u25CF</span>';
    return (
      `<div class="browse-page-tab${active ? ' active' : ''}" data-tab-idx="${i}" role="tab" tabindex="0" title="${title}" aria-selected="${active}">`
      + favicon
      + `<span class="browse-page-tab-title">${title}</span>`
      + `<button class="browse-page-tab-close" data-close-idx="${i}" title="Close tab (Ctrl+W)" aria-label="Close tab">&times;</button>`
      + `</div>`
    );
  }).join('');
  dom.pageTabStrip.innerHTML = parts
    + `<button class="browse-page-tab-new" id="browse-page-tab-new" title="New tab (Ctrl+T)" aria-label="New tab">+</button>`;

  dom.pageTabStrip.querySelectorAll('.browse-page-tab').forEach(el => {
    el.addEventListener('click', (e) => {
      if (e.target.closest('.browse-page-tab-close')) return;
      switchToPageTab(parseInt(el.dataset.tabIdx, 10));
    });
    // Middle-click closes, matching every major browser.
    el.addEventListener('auxclick', (e) => {
      if (e.button === 1) {
        e.preventDefault();
        closePageTab(parseInt(el.dataset.tabIdx, 10));
      }
    });
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        switchToPageTab(parseInt(el.dataset.tabIdx, 10));
      }
    });
  });
  dom.pageTabStrip.querySelectorAll('[data-close-idx]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      closePageTab(parseInt(btn.dataset.closeIdx, 10));
    });
  });
  dom.pageTabStrip.querySelector('#browse-page-tab-new')?.addEventListener('click', () => {
    openNewPageTab();
  });
}

// DOM cache
let dom = {};

// ---------------------------------------------------------------------------
// Discovery Signal Emitter
// ---------------------------------------------------------------------------
/** Emit a discovery signal to the backend. Fire-and-forget. */
function _emitSignal(signalType, data = {}) {
  if (!window.appSettings?.discoveryEnabled) return;
  const body = { signal_type: signalType, ...data };
  fetch('/api/discovery/signal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).catch(() => {}); // fire and forget
}

// ---------------------------------------------------------------------------
// Scroll-jump button visibility — driven by content-area scroll position +
// total scrollable height. Each button reveals itself only when the
// corresponding jump is meaningful: top-arrow appears once the user has
// scrolled down, bottom-arrow appears whenever there's content below the
// viewport. Short pages (no overflow) show neither.
// ---------------------------------------------------------------------------
function updateScrollJumpButtons() {
  const area = dom?.contentArea;
  const topBtn = dom?.scrollTopBtn;
  const bottomBtn = dom?.scrollBottomBtn;
  if (!area || !topBtn || !bottomBtn) return;
  // 24px buffer so the buttons don't strobe near the extremes.
  const threshold = 24;
  const scrollable = area.scrollHeight - area.clientHeight > threshold;
  const atTop = area.scrollTop <= threshold;
  const atBottom = area.scrollHeight - area.clientHeight - area.scrollTop <= threshold;
  topBtn.classList.toggle('visible', scrollable && !atTop);
  bottomBtn.classList.toggle('visible', scrollable && !atBottom);
}

// ---------------------------------------------------------------------------
// Simple markdown → HTML renderer (for AI output)
// ---------------------------------------------------------------------------
// renderSimpleMarkdown was removed 2026-06-16 — browse AI blocks now render
// through the shared chat markdown (compact mode) + incremental stream engine
// for visual + behavioral parity with the chat surface.

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
export function initBrowse() {
  const panel = document.getElementById('browse-panel');
  if (!panel) return;

  // Inject the page-tab strip container at the top of #browse-body-browse
  // (inside the browse body but outside the scrolling content area so it
  // stays pinned while the article scrolls). Done here so index.html
  // doesn't need a new element and this feature is fully contained.
  let pageTabStrip = panel.querySelector('#browse-page-tab-strip');
  if (!pageTabStrip) {
    const bodyBrowse = panel.querySelector('#browse-body-browse');
    if (bodyBrowse) {
      pageTabStrip = document.createElement('div');
      pageTabStrip.id = 'browse-page-tab-strip';
      pageTabStrip.className = 'browse-page-tab-strip hidden';
      pageTabStrip.setAttribute('role', 'tablist');
      pageTabStrip.setAttribute('aria-label', 'Browser tabs');
      bodyBrowse.insertBefore(pageTabStrip, bodyBrowse.firstChild);
    }
  }

  dom = {
    panel,
    pageTabStrip,
    searchInput: panel.querySelector('#browse-search-input'),
    searchSubmit: panel.querySelector('#browse-search-submit'),
    backBtn: panel.querySelector('#browse-back-btn'),
    fwdBtn: panel.querySelector('#browse-fwd-btn'),
    homeBtn: panel.querySelector('#browse-home-btn'),
    closeBtn: panel.querySelector('#browse-close-btn'),
    splitBtn: panel.querySelector('#browse-split-btn'),
    bodyContainer: panel.querySelector('#browse-body-container'),
    contentArea: panel.querySelector('.browse-content-area'),
    searchView: panel.querySelector('#browse-search-view'),
    readerView: panel.querySelector('#browse-reader-view'),
    readerAiBlocks: panel.querySelector('#browse-reader-ai-blocks'),
    readerAskBar: panel.querySelector('#browse-reader-ask-bar'),
    readerAskInput: panel.querySelector('#browse-reader-ask-input'),
    readerAskBtn: panel.querySelector('#browse-reader-ask-btn'),
    // Tabs
    tabBrowse: panel.querySelector('#browse-tab-browse'),
    tabNotes: panel.querySelector('#browse-tab-notes'),
    tabDiscovery: panel.querySelector('#browse-tab-discovery'),
    bodyBrowse: panel.querySelector('#browse-body-browse'),
    bodyNotes: panel.querySelector('#browse-body-notes'),
    bodyDiscovery: document.getElementById('discovery-container'),
    // Scroll-jump (to top / to bottom) floating controls.
    scrollJump: panel.querySelector('#browse-scroll-jump'),
    scrollTopBtn: panel.querySelector('#browse-scroll-top-btn'),
    scrollBottomBtn: panel.querySelector('#browse-scroll-bottom-btn'),
    // Notes — list
    notesItems: panel.querySelector('#browse-notes-items'),
    newNoteBtn: panel.querySelector('#browse-new-note-btn'),
    notesHistoryToggle: panel.querySelector('#browse-notes-history-toggle'),
    notesEditorArea: panel.querySelector('#browse-notes-editor-area'),
    notesEmpty: panel.querySelector('#browse-notes-empty'),
    // Notes — whisper bar
    noteWhisperBar: panel.querySelector('#note-whisper-bar'),
    noteBackBtn: panel.querySelector('#note-whisper-back'),
    noteWhisperTags: panel.querySelector('#note-whisper-tags'),
    noteAddTagBtn: panel.querySelector('#note-whisper-add-tag'),
    noteTagInput: panel.querySelector('#note-whisper-tag-input'),
    noteFormatBtn: panel.querySelector('#note-whisper-format'),
    noteSaveStatus: panel.querySelector('#note-whisper-save'),
    noteExportBtn: panel.querySelector('#note-whisper-export'),
    noteListenBtn: panel.querySelector('#note-whisper-listen'),
    noteMoreBtn: panel.querySelector('#note-whisper-more'),
    noteMoreMenu: panel.querySelector('#note-whisper-menu'),
    // Notes — writing surface
    noteInlineTitle: panel.querySelector('#note-inline-title'),
    noteMetaWords: panel.querySelector('#note-meta-words'),
    noteMetaTime: panel.querySelector('#note-meta-time'),
    noteMetaSource: panel.querySelector('#note-meta-source'),
    noteEditorBody: panel.querySelector('#note-editor-body'),
    noteScroll: panel.querySelector('#note-scroll'),
    noteAiBlocks: panel.querySelector('#note-ai-blocks'),
    noteAskInput: panel.querySelector('#note-ask-input'),
    noteAskBtn: panel.querySelector('#note-ask-btn'),
    noteAiToolsBtn: panel.querySelector('#note-ai-tools-btn'),
    // AI popover (browse reader)
    // (`#browse-ai-popover` was removed when the notes AI flow moved to
    // the ask-bar tools button. Keeping the key off the dom map so
    // stragglers surface as clear undefined rather than dead chains.)
    // Copy button (browse sidebar only)
    aiCopyBtn: panel.querySelector('#browse-ai-copy-btn'),
    // Search bar (for context indicator)
    searchBar: panel.querySelector('.browse-search-bar'),
    searchHistory: panel.querySelector('#browse-search-history'),
  };

  // Page tabs — restore from storage if we have any, else seed with one
  // blank tab so _activePageTab() always returns a valid object. Render
  // strip immediately so restored multi-tab state is visible before the
  // user interacts.
  const restored = _loadPageTabsFromStorage();
  if (!restored || state.pageTabs.length === 0) {
    state.pageTabs = [_newBlankPageTab()];
    state.activePageTabIdx = 0;
  }
  _renderPageTabStrip();

  // Tab switching (outer browse / notes / discovery — NOT page tabs)
  panel.querySelectorAll('.browse-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Search
  dom.searchInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      _closeSearchHistory();
      _submitHeaderSearch();
    }
    if (e.key === 'Escape') _closeSearchHistory();
  });
  dom.searchSubmit?.addEventListener('click', () => {
    _closeSearchHistory();
    _submitHeaderSearch();
  });
  // Realtime filtering when on notes tab
  dom.searchInput?.addEventListener('input', () => {
    if (state.activeTab === 'notes') {
      state.notesFilter = dom.searchInput.value.trim();
      renderNotesList();
    }
  });
  // Search history dropdown — show on click (not focus), hide on outside
  // click. Focus happens automatically when the panel opens so the user can
  // start typing immediately; popping the history dropdown on that auto-
  // focus is jarring. Requiring a tap matches Chrome/Firefox URL-bar UX.
  dom.searchInput?.addEventListener('click', () => {
    if (state.activeTab === 'browse') _showSearchHistory();
  });
  document.addEventListener('mousedown', (e) => {
    const hist = dom.searchHistory;
    if (hist && hist.classList.contains('open')
        && !hist.contains(e.target)
        && e.target !== dom.searchInput) {
      _closeSearchHistory();
    }
  });

  // Navigation
  dom.backBtn?.addEventListener('click', browseBack);
  dom.fwdBtn?.addEventListener('click', browseForward);
  dom.homeBtn?.addEventListener('click', browseHome);
  dom.closeBtn?.addEventListener('click', closeBrowsePanel);

  // Split-screen toggle (desktop only)
  dom.splitBtn?.addEventListener('click', _toggleSplitMode);
  dom.notesHistoryToggle?.addEventListener('click', _toggleNotesHistory);

  // Scroll-jump (to top / to bottom). Visibility is driven by the
  // existing scroll listener via updateScrollJumpButtons(); reader
  // load/navigation paths also call it after layout settles.
  dom.scrollTopBtn?.addEventListener('click', () => {
    dom.contentArea?.scrollTo({ top: 0, behavior: 'smooth' });
  });
  dom.scrollBottomBtn?.addEventListener('click', () => {
    if (!dom.contentArea) return;
    dom.contentArea.scrollTo({
      top: dom.contentArea.scrollHeight,
      behavior: 'smooth',
    });
  });
  dom.contentArea?.addEventListener('scroll', () => {
    updateScrollJumpButtons();
  }, { passive: true });
  // Layout changes (article rendered, window resize, split toggle) don't
  // emit scroll events, so the scroll listener alone misses the
  // scrollHeight transition. ResizeObserver picks those up cheaply.
  if (typeof ResizeObserver !== 'undefined' && dom.contentArea) {
    try {
      const ro = new ResizeObserver(() => updateScrollJumpButtons());
      ro.observe(dom.contentArea);
      // Also observe the first child so we catch content swaps that
      // don't change the container's own size (the article body grew
      // but the scroll area's viewport stayed the same).
      const inner = dom.contentArea.firstElementChild;
      if (inner) ro.observe(inner);
    } catch { /* ResizeObserver unavailable in old browsers — no auto-resize */ }
  }
  // Initial pass.
  updateScrollJumpButtons();

  // Reader ask bar
  dom.readerAskInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const q = dom.readerAskInput.value.trim();
      if (q) { browseAiAction('ask', q); dom.readerAskInput.value = ''; }
    }
  });
  dom.readerAskBtn?.addEventListener('click', () => {
    const q = dom.readerAskInput?.value.trim();
    if (q) { browseAiAction('ask', q); dom.readerAskInput.value = ''; }
  });

  // Reader AI block actions (event delegation — same pattern as notes)
  dom.readerAiBlocks?.addEventListener('click', (e) => {
    const btn = e.target.closest('.browse-ai-block-btn');
    if (!btn) return;
    const block = btn.closest('.browse-ai-block');
    if (!block) return;
    const action = btn.dataset.action;
    const content = block.dataset.markdown || '';
    if (action === 'copy') {
      copyToClipboard(content)
        .then((ok) => showToast(ok ? 'Copied to clipboard' : 'Copy failed', ok ? 'success' : 'error'));
    } else if (action === 'remove') {
      block.style.opacity = '0';
      block.style.transform = 'translateY(-8px)';
      block.style.transition = 'all 0.2s ease';
      setTimeout(() => block.remove(), 200);
    }
  });

  // Link interception in article body
  dom.contentArea?.addEventListener('click', (e) => {
    const link = e.target.closest('.browse-article-body a');
    if (link) {
      const href = link.getAttribute('href');
      const targetUrl = _resolveArticleLinkTarget(href);
      if (targetUrl) {
        e.preventDefault();
        // Ctrl/Cmd/shift opens a reader tab; otherwise follow user prefs.
        _openArticleLink(targetUrl, e);
      }
    }
  });
  // Middle-click opens links in a new tab too (standard browser UX).
  dom.contentArea?.addEventListener('auxclick', (e) => {
    if (e.button !== 1) return;
    const link = e.target.closest('.browse-article-body a');
    if (!link) return;
    const href = link.getAttribute('href');
    const targetUrl = _resolveArticleLinkTarget(href);
    if (!targetUrl) return;
    e.preventDefault();
    if (targetUrl) openNewPageTab(targetUrl);
  });

  // Continue the click delegation from above — split into a second
  // listener to keep the new-tab logic above self-contained.
  dom.contentArea?.addEventListener('click', (e) => {

    // Article images — open in lightbox on click
    const articleImg = e.target.closest('.browse-article-body img');
    if (articleImg && articleImg.src) {
      e.preventDefault();
      const modal = document.getElementById('image-lightbox-modal');
      const img = document.getElementById('lightbox-img');
      const meta = document.getElementById('lightbox-meta');
      if (modal && img) {
        img.src = articleImg.src;
        img.alt = articleImg.alt || 'Article image';
        if (meta) meta.textContent = articleImg.alt || '';
        // Hide action buttons (they're for generated images, not article images)
        const actions = modal.querySelector('.lightbox-actions');
        if (actions) actions.style.display = 'none';
        modal.classList.add('visible');
      }
      return;
    }

    // Article action buttons (delegated)
    const saveBtn = e.target.closest('[data-article-action="save"]');
    if (saveBtn) browseSave();

    const saveNoteBtn = e.target.closest('[data-article-action="save-note"]');
    if (saveNoteBtn) saveArticleAsNote();

    const discussBtn = e.target.closest('[data-article-action="discuss"]');
    if (discussBtn) discussInChat();

    const typoBtn = e.target.closest('[data-article-action="typography"]');
    if (typoBtn) { _openTypographyPopover(typoBtn); return; }

    const tocBtn = e.target.closest('[data-article-action="toc"]');
    if (tocBtn) { _openTocPopover(tocBtn); return; }

    const readAloudBtn = e.target.closest('[data-article-action="read-aloud"]');
    if (readAloudBtn) { _toggleReadAloud(readAloudBtn); return; }

    const bookmarkBtn = e.target.closest('[data-article-action="bookmark"]');
    if (bookmarkBtn) { _toggleBookmark(bookmarkBtn); return; }

    // Article AI tool buttons
    const aiBtn = e.target.closest('[data-article-ai]');
    if (aiBtn) {
      const action = aiBtn.dataset.articleAi;
      if (action === 'translate') {
        const lang = prompt('Translate to which language?', 'Spanish');
        if (lang) browseAiAction(action, lang);
      } else {
        browseAiAction(action);
      }
    }

    // Retry button
    const retryBtn = e.target.closest('.browse-retry-btn');
    if (retryBtn && state.currentUrl) browseFetch(state.currentUrl);
  });

  // Notes: back button (mobile — return to list)
  dom.noteBackBtn?.addEventListener('click', closeNoteEditor);

  // Surface image paste/drop upload failures (dispatched by notes-editor.js)
  // instead of failing silently in the console.
  document.addEventListener('note-image-attach-error', () => {
    showToast('Couldn’t attach image', 'error');
  });

  // Notes: new note
  dom.newNoteBtn?.addEventListener('click', createNewNote);

  // Provenance chip — toggle "companion-created only" on the notes list.
  const originChip = document.getElementById('browse-notes-origin-chip');
  originChip?.addEventListener('click', () => {
    state.notesOriginCompanion = !state.notesOriginCompanion;
    originChip.classList.toggle('active', state.notesOriginCompanion);
    originChip.setAttribute('aria-pressed', String(state.notesOriginCompanion));
    renderNotesList();
  });

  // Notes: whisper bar export / more actions
  dom.noteExportBtn?.addEventListener('click', downloadNote);
  dom.notesEditorArea?.addEventListener('click', (e) => {
    const sourceLink = e.target.closest('.note-meta-source a');
    if (!sourceLink) return;
    const url = sourceLink.getAttribute('href') || '';
    if (!/^https?:\/\//i.test(url)) return;
    e.preventDefault();
    if (e.ctrlKey || e.metaKey || e.shiftKey) {
      openNewPageTab(url);
      return;
    }
    switchTab('browse');
    browseFetch(url);
  });

  // Read the current note aloud. Pulls markdown via the note-editor
  // module so Milkdown's state (not just DOM text) is the source of
  // truth, then strips markdown syntax to prose before handing to TTS.
  dom.noteListenBtn?.addEventListener('click', async () => {
    try {
      const [{ readAloud }, noteEditor] = await Promise.all([
        import('./read-aloud.js'),
        import('./note-editor.js'),
      ]);
      const md = (noteEditor.getMarkdown?.() || '').trim();
      if (!md) {
        showToast('Note is empty.', 'info', 1500);
        return;
      }
      // Strip markdown syntax to leave readable prose. Same cleanup
      // the chat TTS pipeline uses on model output.
      const { ttsCleanText } = await import('./chat/tts.js');
      const prose = ttsCleanText ? ttsCleanText(md) : md;
      readAloud(prose, dom.noteListenBtn);
    } catch (err) {
      showToast('Read-aloud failed: ' + (err?.message || err), 'error');
    }
  });
  dom.noteMoreBtn?.addEventListener('click', _showNoteMoreMenu);
  _wireNoteMoreMenu();

  // Initialize NoteEditor orchestrator
  NoteEditor.init({
    dom: {
      inlineTitle: dom.noteInlineTitle,
      editorBody: dom.noteEditorBody,
      whisperBar: dom.noteWhisperBar,
      whisperTags: dom.noteWhisperTags,
      addTagBtn: dom.noteAddTagBtn,
      tagInput: dom.noteTagInput,
      formatBtn: dom.noteFormatBtn,
      saveStatus: dom.noteSaveStatus,
      metaWords: dom.noteMetaWords,
      metaTime: dom.noteMetaTime,
      metaSource: dom.noteMetaSource,
      scroll: dom.noteScroll,
    },
    onSave: _handleNoteEditorSave,
    slashCallbacks: {
      onImage: _slashGenerateImage,
      // Slash-menu filtering ("/ai") is not a real question; the menu
      // strips that before calling us. Real query text still runs
      // immediately, otherwise focus the persistent ask bar so the user
      // can compose in place — same flow as the bubble menu + Mod-J.
      onAi: (q) => {
        const query = (q || '').trim();
        if (query) { noteAiAction('ask', query); return; }
        dom.noteAskInput?.focus();
        dom.noteAskInput?.scrollIntoView({ block: 'center', behavior: 'smooth' });
      },
      onKnowledge: (q) => {
        const query = (q || '').trim();
        if (query) {
          noteAiAction('ask', `From your knowledge packs: ${query}`);
          return;
        }
        if (dom.noteAskInput) {
          dom.noteAskInput.value = 'From your knowledge packs: ';
          dom.noteAskInput.focus();
          dom.noteAskInput.scrollIntoView({ block: 'center', behavior: 'smooth' });
          dom.noteAskInput.dispatchEvent(new Event('input', { bubbles: true }));
        }
      },
    },
  });

  // Expose Milkdown loader for NoteEditor. Track the last created Crepe
  // globally so we always destroy the right instance. Serialize calls
  // behind a chain promise: Crepe's ProseMirror plugins register keyed
  // state against a module-global registry, and concurrent destroy+
  // create sequences can interleave and trigger "Adding different
  // instances of a keyed plugin (plugin$N)". The queue guarantees one
  // editor-lifecycle at a time.
  let _editorChain = Promise.resolve();
  window.__loadNoteEditor = function(container, markdown, onChange) {
    const run = async () => {
      if (window.__activeCrepeInstance) {
        try {
          if (window.__activeCrepeInstance.destroy) await window.__activeCrepeInstance.destroy();
        } catch { /* ignore */ }
        window.__activeCrepeInstance = null;
        state.milkdownEditor = null;
        await new Promise(r => setTimeout(r, 0));
      }
      const editor = await loadMilkdownEditor(container, markdown, onChange);
      window.__activeCrepeInstance = editor;
      return editor;
    };
    // Chain on the previous init so callers always get the correct
    // instance for their container. Swallow rejections on the chain
    // itself so one failure doesn't poison all future loads.
    const next = _editorChain.then(run, run);
    _editorChain = next.catch(() => {});
    return next;
  };

  // Legacy `#browse-ai-popover` + `#browse-note-ai-toggle` + the
  // `_ensureAiFab` floating robot button were removed on 2026-04-24 —
  // the notes AI flow now lives on the `#note-ai-tools-btn` pill in
  // the ask bar. The block here used to contain the popover toggle +
  // outside-click handlers + `.browse-ai-popover-item` delegation +
  // `_positionAiPopover` helper, all dead.

  // Persistent ask bar. Button starts disabled in HTML; re-enable it
  // whenever the input has non-whitespace content so both click and
  // Enter paths are viable. Reset on submit.
  const _syncAskBtn = () => {
    if (!dom.noteAskBtn) return;
    const v = dom.noteAskInput?.value.trim() || '';
    dom.noteAskBtn.disabled = v.length === 0;
  };
  // Render (or tear down) the "Context: …" chip that sits above the
  // ask bar when the user invoked AI on a selection. The chip shows
  // a truncated preview of the passage so it's clear that context
  // will be sent along with whatever question the user types. Click
  // the × to drop the context.
  const _renderAskContextChip = () => {
    const input = dom.noteAskInput;
    const bar = input?.parentElement;
    if (!bar) return;
    let chip = bar.parentElement?.querySelector('.note-ask-context-chip');
    const context = input?.dataset.context || '';
    if (!context) {
      if (chip) chip.remove();
      return;
    }
    if (!chip) {
      chip = document.createElement('div');
      chip.className = 'note-ask-context-chip';
      chip.innerHTML = `
        <span class="note-ask-context-label">Context</span>
        <span class="note-ask-context-text"></span>
        <button class="note-ask-context-clear" type="button" aria-label="Remove context">&times;</button>
      `;
      bar.parentElement.insertBefore(chip, bar);
      chip.querySelector('.note-ask-context-clear').addEventListener('click', () => {
        delete input.dataset.context;
        input.placeholder = 'Ask about this note...';
        _renderAskContextChip();
        input.focus();
      });
    }
    const preview = context.length > 140 ? context.slice(0, 138) + '…' : context;
    chip.querySelector('.note-ask-context-text').textContent = preview;
  };
  // Submit: combines any stashed "context" passage (set when the user
  // invokes AI on a selection) with the question they just typed, so
  // the model sees both. Clears context on submit.
  const _submitAsk = () => {
    if (!dom.noteAskInput) return;
    const question = dom.noteAskInput.value.trim();
    if (!question) return;
    const context = (dom.noteAskInput.dataset.context || '').trim();
    const payload = context
      ? `Context passage from the note:\n"${context}"\n\nQuestion: ${question}`
      : question;
    noteAiAction('ask', payload);
    dom.noteAskInput.value = '';
    delete dom.noteAskInput.dataset.context;
    dom.noteAskInput.placeholder = 'Ask about this note...';
    _renderAskContextChip();
    _syncAskBtn();
  };
  dom.noteAskInput?.addEventListener('input', _syncAskBtn);
  dom.noteAskInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      _submitAsk();
    } else if (e.key === 'Escape') {
      // Clear a stashed context passage; a second Escape blurs.
      if (dom.noteAskInput.dataset.context) {
        delete dom.noteAskInput.dataset.context;
        dom.noteAskInput.placeholder = 'Ask about this note...';
        _renderAskContextChip();
      } else {
        dom.noteAskInput.blur();
      }
    }
  });
  dom.noteAskBtn?.addEventListener('click', () => {
    if (!dom.noteAskBtn.disabled) _submitAsk();
  });
  _syncAskBtn();   // initial state

  // AI tools popover — nine backend actions grouped by intent, rendered
  // above the ask bar. Read-only actions stream into an inline AI block;
  // insertable actions show an "Insert into note" button on the result
  // (handled by noteAiAction). Translate prompts for a language first.
  _wireNoteAiToolsPopover();

  // Scroll-to-top / scroll-to-bottom jumper — visible only when the
  // note content overflows the viewport. The bottom button jumps to
  // the ask bar at the end of the scroll container.
  _wireNoteScrollJumper();
  // Note editor: image delete overlay on click
  dom.noteEditorBody?.addEventListener('click', (e) => {
    const img = e.target.closest('#note-editor-body img');
    // Remove any existing image toolbar
    document.querySelectorAll('.note-img-toolbar').forEach(t => t.remove());
    if (!img) return;

    const storedPrompt = img.title || '';

    const toolbar = document.createElement('div');
    toolbar.className = 'note-img-toolbar';
    toolbar.innerHTML = `
      ${storedPrompt ? `
      <button class="note-img-tb-btn" data-action="reroll" title="Generate a new variation">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M21 12a9 9 0 1 1-3-6.7"/><polyline points="21 4 21 10 15 10"/></svg>
        Re-roll
      </button>` : ''}
      <button class="note-img-tb-btn" data-action="delete" title="Remove image">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        Remove
      </button>
      <button class="note-img-tb-btn" data-action="fullsize" title="View full size">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
        View
      </button>
    `;

    // Position above the image
    img.style.position = 'relative';
    img.parentElement.style.position = 'relative';
    img.parentElement.insertBefore(toolbar, img);

    toolbar.addEventListener('click', async (ev) => {
      ev.stopPropagation();
      const action = ev.target.closest('[data-action]')?.dataset.action;
      if (action === 'reroll' && storedPrompt) {
        toolbar.remove();
        showToast(`Re-rolling: "${storedPrompt.slice(0, 40)}\u2026"`, 'info');
        try {
          const resp = await fetch('/api/image/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: storedPrompt, seed: -1 }),
          });
          if (!resp.ok) { showToast('Re-roll failed', 'error'); return; }
          const data = await resp.json();
          if (data.url) {
            img.src = data.url;
            debounceNoteSave();
            showToast('New variation', 'success');
          }
        } catch { showToast('Re-roll failed', 'error'); }
      } else if (action === 'delete') {
        // Remove the image node from ProseMirror by removing from DOM + triggering save
        img.remove();
        toolbar.remove();
        debounceNoteSave();
      } else if (action === 'fullsize') {
        // Open in lightbox
        const modal = document.getElementById('image-lightbox-modal');
        const lbImg = document.getElementById('lightbox-img');
        if (modal && lbImg) {
          lbImg.src = img.src;
          lbImg.alt = img.alt || '';
          const actions = modal.querySelector('.lightbox-actions');
          if (actions) actions.style.display = 'none';
          modal.classList.add('visible');
        }
        toolbar.remove();
      }
    });
  });

  // Close image toolbar on click elsewhere
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.note-img-toolbar') && !e.target.closest('#note-editor-body img')) {
      document.querySelectorAll('.note-img-toolbar').forEach(t => t.remove());
    }
  });

  // Inline AI block actions (event delegation on blocks container)
  dom.noteAiBlocks?.addEventListener('click', (e) => {
    const btn = e.target.closest('.browse-ai-block-btn');
    if (!btn) return;
    const block = btn.closest('.browse-ai-block');
    if (!block) return;
    const action = btn.dataset.action;
    const content = block.dataset.markdown || '';
    if (action === 'insert') {
      insertAiBlockIntoNote(content);
      block.remove();
      debounceNoteSave();
    } else if (action === 'copy') {
      copyToClipboard(content)
        .then((ok) => showToast(ok ? 'Copied to clipboard' : 'Copy failed', ok ? 'success' : 'error'));
    } else if (action === 'remove') {
      block.style.opacity = '0';
      block.style.transform = 'translateY(-8px)';
      block.style.transition = 'all 0.2s ease';
      setTimeout(() => { block.remove(); debounceNoteSave(); }, 200);
    }
  });

  // Browse AI sidebar copy button
  dom.aiCopyBtn?.addEventListener('click', () => {
    if (state._browseAiFullText) {
      copyToClipboard(state._browseAiFullText).then((ok) => showToast(ok ? 'Copied to clipboard' : 'Copy failed', ok ? 'success' : 'error'));
    }
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (!state.panelOpen) return;

    if (e.key === 'Escape') {
      closeBrowsePanel();
      return;
    }

    // Ctrl/Cmd+L → focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
      e.preventDefault();
      dom.searchInput?.focus();
      dom.searchInput?.select();
      return;
    }

    // Ctrl/Cmd+T → new page tab
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 't') {
      e.preventDefault();
      openNewPageTab();
      return;
    }

    // Ctrl/Cmd+W → close active page tab. Guarded so typing 'w' into
    // a note or URL bar isn't swallowed.
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'w') {
      // Let Ctrl+W in an editor do its native thing (delete-word in
      // some shells). We only take over when focus is NOT in a text
      // input so the URL bar / note editor stay functional.
      const tag = (e.target?.tagName || '').toLowerCase();
      const editable = tag === 'input' || tag === 'textarea' || e.target?.isContentEditable;
      if (!editable) {
        e.preventDefault();
        closePageTab(state.activePageTabIdx);
        return;
      }
    }

    // Ctrl/Cmd+Tab → next tab; add Shift for previous. Wraps both ways.
    if ((e.ctrlKey || e.metaKey) && e.key === 'Tab' && state.pageTabs.length > 1) {
      e.preventDefault();
      const n = state.pageTabs.length;
      const next = e.shiftKey
        ? (state.activePageTabIdx - 1 + n) % n
        : (state.activePageTabIdx + 1) % n;
      switchToPageTab(next);
      return;
    }

    // Alt+Arrow → back/forward
    if (e.altKey && e.key === 'ArrowLeft') {
      e.preventDefault();
      browseBack();
    }
    if (e.altKey && e.key === 'ArrowRight') {
      e.preventDefault();
      browseForward();
    }
  });

  // Chat → Browse bridge: open URL from chat in reader
  document.addEventListener('augmentum:browse-url', (e) => {
    const url = e.detail?.url;
    if (url) {
      if (!state.panelOpen) openBrowsePanel();
      switchTab('browse');
      browseFetch(url);
    }
  });

  // Orb → Browse inner tab bridge. Notes and ForYou orbs both target the
  // browse surface; this lets orb-nav.js drive the inner tab so dragging
  // the Notes orb visibly switches to notes instead of silent-focusing
  // whatever browse panel was last on (audit §4.4).
  document.addEventListener('augmentum:switch-browse-tab', (e) => {
    const tab = e.detail?.tab;
    if (tab === 'browse' || tab === 'notes' || tab === 'discovery') {
      switchTab(tab);
    }
  });

  // Command composer / voice → Browse bridge: run a search query
  document.addEventListener('augmentum:browse-search', (e) => {
    const query = (e.detail?.query || '').trim();
    if (!query) return;
    if (!state.panelOpen) openBrowsePanel();
    switchTab('browse');
    browseSearch(query);
  });

  // Discovery → Browse bridge: open URL from history/recommendations
  window.addEventListener('discovery:open-url', (e) => {
    const url = e.detail?.url;
    if (url) {
      if (!state.panelOpen) openBrowsePanel();
      switchTab('browse');
      browseFetch(url);
    }
  });

  // Note editor toolbars → AI action. The format-bar (desktop selection
  // popover) and mobile toolbar dispatch note-ai-action; route it through
  // the same noteAiAction pipeline the inline note-ai UI uses.
  document.addEventListener('note-ai-action', (e) => {
    const action = e.detail?.action;
    if (!action) return;
    if (action === 'ai-menu') {
      // Mobile "AI" button — route through the same ask-bar flow the
      // desktop bubble menu uses. Selection (if any) becomes a
      // context chip via input.dataset.context; the user types their
      // actual question in the bar, and the submit path combines
      // both. This avoids the model seeing the passage as the literal
      // prompt and answering "is it present?" style.
      const selectedText = (e.detail?.selectedText || '').trim();
      const input = dom.noteAskInput;
      if (input) {
        if (selectedText) {
          input.dataset.context = selectedText;
          input.placeholder = 'Ask about the highlighted passage…';
        } else {
          delete input.dataset.context;
          input.placeholder = 'Ask about this note...';
        }
        input.focus();
        input.scrollIntoView({ block: 'center', behavior: 'smooth' });
        if (typeof _renderAskContextChip === 'function') _renderAskContextChip();
      }
      return;
    }
    const selectedText = e.detail?.selectedText || '';
    noteAiAction(action, selectedText || undefined);
  });

  // Landing page — quick action chips. Snapshot the original markup
  // BEFORE _wireLandingChips appends the dynamic bookmarks/packs
  // sections so _restoreLanding rebuilds the same lazy-injected state
  // every time, without accumulating duplicate sections.
  if (dom.searchView && _savedLandingHTML === null) {
    _savedLandingHTML = dom.searchView.innerHTML;
  }
  _wireLandingChips();

  updateNavButtons();

  // Cross-surface: receive "Save to Note" from media cards
  window.addEventListener('media:save-to-note', async (e) => {
    const video = e.detail;
    if (!video) return;
    const title = video.title || 'Untitled Video';
    const url = video.url || '';
    const channel = video.channel || '';
    const content = `# ${title}\n\n` +
      (channel ? `**Channel:** ${channel}\n\n` : '') +
      (url ? `**URL:** ${url}\n\n` : '') +
      `Saved from ${video.platform || 'video'} playback.`;

    try {
      const resp = await fetch('/api/browse/notes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title,
          content,
          source_url: url,
          tags: ['video', video.platform || 'media'].filter(Boolean),
        }),
      });
      if (resp.ok) {
        console.debug('Video saved to notes:', title);
      }
    } catch (err) {
      console.warn('Failed to save video to notes:', err);
    }
  });

  // Cross-surface: "Save to Files" from media cards.  Bookmarks the URL
  // into file_index so the user can re-watch from the Files panel — no
  // download, just a pointer.
  window.addEventListener('media:save-to-files', async (e) => {
    const video = e.detail;
    if (!video?.url) return;
    try {
      const { saveBookmark } = await import('./files/api.js');
      const { showToast } = await import('./app.js');
      const data = await saveBookmark({
        url: video.url,
        title: video.title || 'Untitled video',
        thumbnail: video.thumbnail || video.thumbnailUrl || '',
        channel: video.channel || video.author || '',
        duration: video.duration ?? video.length ?? null,
        platform: video.platform || '',
        video_id: video.video_id || video.videoId || '',
      });
      if (data?.ok) showToast(`Saved "${video.title || 'video'}" to Files`, 'success');
      else showToast('Failed to save to Files', 'error');
    } catch (err) {
      console.warn('Failed to bookmark video:', err);
    }
  });
}

// ---------------------------------------------------------------------------
// Landing Page — quick search chips + recent searches
// ---------------------------------------------------------------------------
// Captured at first wire so _restoreLanding can put the search view's
// original markup back after a focused view (bookmarks, per-pack home)
// is dismissed. Without this the "back" buttons would just unhide an
// already-overwritten container.
let _savedLandingHTML = null;

/** Rebuild the landing markup after a focused view has overwritten it.
 *
 * Called by the back-out paths in _openBookmarksView and _openPackView.
 * Restores the saved HTML snapshot then re-wires chip clicks and re-
 * injects the bookmarks + knowledge-packs sections. Safe to call when
 * the snapshot doesn't exist yet (early init); it just no-ops.
 */
function _restoreLanding() {
  if (!_savedLandingHTML || !dom.searchView) return;
  dom.searchView.innerHTML = _savedLandingHTML;
  _wireLandingChips();
}

function _wireLandingChips() {
  const landing = document.getElementById('browse-landing');
  if (!landing) return;
  landing.querySelectorAll('.browse-landing-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.dataset.query;
      if (dom.searchInput) dom.searchInput.value = query;
      browseSearch(query);
    });
  });
  _renderLandingRecents();
  // Lazily inject a bookmarks strip below the recent searches and
  // render into it. Injection is idempotent so repeated landing
  // renders don't stack duplicate containers.
  let bookmarksEl = document.getElementById('browse-landing-bookmarks');
  if (!bookmarksEl) {
    bookmarksEl = document.createElement('div');
    bookmarksEl.id = 'browse-landing-bookmarks';
    bookmarksEl.className = 'browse-landing-bookmarks';
    const recents = landing.querySelector('#browse-landing-recents');
    if (recents && recents.parentElement) {
      recents.parentElement.insertBefore(bookmarksEl, recents.nextSibling);
    } else {
      landing.appendChild(bookmarksEl);
    }
  }
  _renderLandingBookmarks();

  // Knowledge packs section — same lazy-injection pattern as bookmarks.
  // Container always exists once injected; _renderLandingPacks empties
  // it (no-op visually) when the user has no packs installed, so the
  // landing layout stays identical to the pre-feature state.
  let packsEl = document.getElementById('browse-landing-packs');
  if (!packsEl) {
    packsEl = document.createElement('div');
    packsEl.id = 'browse-landing-packs';
    packsEl.className = 'browse-landing-packs';
    if (bookmarksEl && bookmarksEl.parentElement) {
      bookmarksEl.parentElement.insertBefore(packsEl, bookmarksEl.nextSibling);
    } else {
      landing.appendChild(packsEl);
    }
  }
  _renderLandingPacks();

  // Notify cross-cutting features (currently: language-learning) that the
  // landing chips row has been (re)rendered. The Learning chip lazy-injects
  // here if a language pack is installed.
  document.dispatchEvent(new CustomEvent('augmentum:browse-landing-ready'));
}

/** Render the "Knowledge Packs" landing section. Empty when no packs
 * are installed — preserves the original landing layout for users who
 * have never downloaded one. Each card opens the pack home view.
 */
// Live-refresh the Browse landing knowledge-packs grid when the shared
// library changes server-side (knowledge_routes.py emits knowledge.changed).
// Only re-render while the Browse panel is open — no manual refresh in the PWA.
window.addEventListener('system-event:knowledge.changed', () => {
  if (state.panelOpen) _renderLandingPacks();
});

async function _renderLandingPacks() {
  const container = document.getElementById('browse-landing-packs');
  if (!container) return;
  let packs = [];
  let failedConversions = [];
  try {
    const resp = await fetch('/api/knowledge/packs');
    if (resp.ok) {
      const data = await resp.json();
      packs = Array.isArray(data.packs) ? data.packs.filter(p => p.active !== false) : [];
      failedConversions = Array.isArray(data.failed_conversions) ? data.failed_conversions : [];
    }
  } catch {
    container.innerHTML = '';
    return;
  }
  // Dedupe: when both an augpack and a ZIM exist for the same pack_id
  // (e.g. user has the converted .augpack alongside the original .zim),
  // surface the ZIM card — it's browseable and the search dispatcher
  // runs both legs anyway. Without this dedupe the user sees two cards
  // for one pack and can't tell why one is "search-only" and the other
  // isn't.
  const byId = new Map();
  for (const p of packs) {
    const id = p.pack_id || p.id || '';
    if (!id) continue;
    const existing = byId.get(id);
    if (!existing || p.type === 'zim') {
      byId.set(id, p);
    }
  }
  packs = Array.from(byId.values());
  // Failed conversions surface even when there are no working packs —
  // a user whose first pack install crashed needs to see the recovery
  // affordance on the landing instead of a totally empty section.
  if (!packs.length && !failedConversions.length) {
    container.innerHTML = '';
    return;
  }

  const formatBytes = (n) => {
    if (!n || n <= 0) return '';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let v = n;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(v >= 10 ? 0 : 1)} ${units[i]}`;
  };

  const formatCount = (n) => {
    if (!n) return '';
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M articles`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K articles`;
    return `${n} articles`;
  };

  // Failed conversions block — rendered above the grid so users see
  // the recovery affordance before the working packs. Each entry shows
  // Resume + Discard. Resume is enabled iff fc.resumable (paired .zim
  // present, chunks committed, dim matches current model); otherwise
  // we show a tooltip explaining why and only Discard is actionable.
  const failedHtml = failedConversions.length ? `
    <div class="browse-landing-pack-failed">
      ${failedConversions.map(fc => {
        const stage = fc.last_stage || 'unknown';
        const progress = fc.last_total > 0
          ? `${fc.last_progress?.toLocaleString()} of ${fc.last_total.toLocaleString()}`
          : (fc.chunks_committed > 0 ? `${fc.chunks_committed.toLocaleString()} chunks embedded` : '');
        const ageHrs = Math.round((fc.stale_seconds || 0) / 3600);
        const ageLabel = ageHrs > 0 ? `${ageHrs}h ago` : 'recently';
        const resumable = fc.resumable === true;
        const resumeTitle = resumable
          ? `Continue from chunk ${(fc.chunks_committed || 0).toLocaleString()} — re-uses already-embedded vectors, no work lost.`
          : (fc.not_resumable_reason || 'Cannot resume');
        // Default the batch-size input to the current global setting if
        // we have it on window, else fall back to 32 (worker default).
        const defaultBatch = (window.augmentumConfig && window.augmentumConfig.knowledge_embedding_batch_size) || 32;
        return `
          <div class="browse-landing-pack-failed-card" data-failed-pack="${escapeHtml(fc.pack_id)}">
            <div class="browse-landing-pack-failed-icon">⚠</div>
            <div class="browse-landing-pack-failed-body">
              <div class="browse-landing-pack-failed-name">${escapeHtml(fc.pack_id)}</div>
              <div class="browse-landing-pack-failed-meta">
                Conversion stalled at <em>${escapeHtml(stage)}</em>${progress ? ` (${escapeHtml(progress)})` : ''} — ${escapeHtml(ageLabel)}
              </div>
              ${fc.last_error ? `<div class="browse-landing-pack-failed-error">${escapeHtml(fc.last_error)}</div>` : ''}
              ${resumable ? `
                <div class="browse-landing-pack-failed-batch" title="Lower this if the original failure was OOM. The model loads ~6GB base; each batch slot adds proportionally.">
                  <label>Batch size:</label>
                  <input type="number" min="1" max="2048" value="${defaultBatch}" data-batch-input="${escapeHtml(fc.pack_id)}" />
                </div>
              ` : ''}
              <div class="browse-landing-pack-failed-progress" data-progress-slot="${escapeHtml(fc.pack_id)}" style="display:none">
                <div class="browse-landing-pack-failed-progress-bar"><div class="browse-landing-pack-failed-progress-fill"></div></div>
                <div class="browse-landing-pack-failed-progress-label"></div>
                <button class="browse-landing-pack-failed-progress-cancel"
                        data-progress-cancel="${escapeHtml(fc.pack_id)}"
                        style="display:none"
                        title="Cancel this in-flight install/resume">Cancel</button>
              </div>
            </div>
            <div class="browse-landing-pack-failed-actions">
              <button class="browse-landing-pack-failed-resume"
                      data-resume="${escapeHtml(fc.pack_id)}"
                      ${resumable ? '' : 'disabled'}
                      title="${escapeHtml(resumeTitle)}">Resume</button>
              <button class="browse-landing-pack-failed-discard"
                      data-discard="${escapeHtml(fc.pack_id)}"
                      title="Remove the .augpack shell + progress file. Original .zim is preserved.">Discard</button>
            </div>
          </div>
        `;
      }).join('')}
    </div>
  ` : '';

  container.innerHTML = `
    <p class="browse-landing-recents-label">Knowledge Packs <span class="browse-landing-packs-count">${packs.length}</span></p>
    ${failedHtml}
    <div class="browse-landing-packs-grid">
      ${packs.map(p => {
        const id = p.pack_id || p.id || '';
        const name = p.name || id;
        const desc = p.description || '';
        const count = formatCount(p.chunk_count);
        const size = formatBytes(p.file_size);
        const isZim = p.type === 'zim';
        // ZIM packs without an augpack sidecar can be opted into vector
        // search via the per-pack Embed action. Install always lands a
        // pack as ZIM-only (auto-embed-on-install was removed 2026-05-07);
        // every freshly-installed pack starts in this state.
        const canEmbed = isZim && p.has_vector_index === false;
        const homePath = isZim ? (p.main_entry_path || '') : '';
        const homeUrl = homePath ? `zim:${id}/${homePath}` : '';
        // Inline meta — single editorial line, middot separator. Tighter
        // than two stacked spans and reads as type-set typography rather
        // than a stat block.
        const metaParts = [count, size].filter(Boolean).map(escapeHtml);
        const metaLine = metaParts.length
          ? metaParts.join(' <span class="browse-landing-pack-meta-sep" aria-hidden="true">·</span> ')
          : '';
        // ZIM packs render their bundled illustration entry as the card
        // glyph — actual pack branding (Wikipedia "W", MDWiki cross,
        // Project Gutenberg colophon, etc.). Falls back to a letter pill
        // on packs without an illustration (older ZIMs) via the onerror
        // handler wired by ``_attachPackFaviconFallbacks`` post-render.
        // Augpacks keep the quincunx monogram — they have no illustration
        // entry and the glyph still communicates "search-only corpus".
        const typeGlyph = isZim
          ? _packFaviconHtml(id, name, 48)
          : `<svg viewBox="0 0 16 16" fill="currentColor" width="14" height="14" aria-hidden="true"><circle cx="4" cy="4" r="1.2"/><circle cx="12" cy="4" r="1.2"/><circle cx="8" cy="8" r="1.2"/><circle cx="4" cy="12" r="1.2"/><circle cx="12" cy="12" r="1.2"/></svg>`;
        const typeLabel = isZim ? 'Browseable archive' : 'Search-only corpus';
        // Wrap card + optional progress strip so embed feedback flows
        // beneath the card without distorting grid-row height when idle.
        return `
          <div class="browse-landing-pack-wrapper" data-pack-wrapper="${escapeHtml(id)}">
            <div class="browse-landing-pack" data-pack-type="${isZim ? 'zim' : 'augpack'}">
              <button class="browse-landing-pack-main" data-pack-id="${escapeHtml(id)}" title="${escapeHtml(desc || name)}">
                <span class="browse-landing-pack-icon" title="${escapeHtml(typeLabel)}">${typeGlyph}</span>
                <span class="browse-landing-pack-body">
                  <span class="browse-landing-pack-name">${escapeHtml(name)}</span>
                  ${desc ? `<span class="browse-landing-pack-desc">${escapeHtml(desc)}</span>` : ''}
                  ${metaLine ? `<span class="browse-landing-pack-meta">${metaLine}</span>` : ''}
                </span>
              </button>
              <div class="browse-landing-pack-actions">
                ${canEmbed ? `
                  <button class="browse-landing-pack-action browse-landing-pack-embed"
                          data-embed-pack="${escapeHtml(id)}"
                          title="Build a vector index for semantic search across this pack. Runs once in the background, then chat queries get a semantic-similarity leg alongside keyword."
                          aria-label="Embed ${escapeHtml(name)} for vector search">
                    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" width="14" height="14" aria-hidden="true"><circle cx="4" cy="12" r="1.6" fill="currentColor"/><path d="m4.5 11.5 7-7M11.5 4.5v3.4M11.5 4.5H8.1"/></svg>
                  </button>
                ` : ''}
                <button class="browse-landing-pack-action"
                        data-pack-search="${escapeHtml(id)}"
                        title="Search ${escapeHtml(name)}"
                        aria-label="Search ${escapeHtml(name)}">
                  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="4.5"/><path d="m13.5 13.5-3-3"/></svg>
                </button>
                ${homeUrl ? `
                  <button class="browse-landing-pack-action"
                          data-pack-home="${escapeHtml(homeUrl)}"
                          title="Open ${escapeHtml(name)} home page"
                          aria-label="Open ${escapeHtml(name)} home page">
                    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" width="14" height="14" aria-hidden="true"><path d="M2.5 8 8 3l5.5 5"/><path d="M4 7v6.5h8V7"/></svg>
                  </button>
                ` : ''}
              </div>
            </div>
            ${canEmbed ? `
              <div class="browse-landing-pack-embed-row" data-embed-row="${escapeHtml(id)}" hidden>
                <div class="browse-landing-pack-embed-progress" data-embed-progress="${escapeHtml(id)}" style="display:none">
                  <div class="browse-landing-pack-embed-progress-bar"><div class="browse-landing-pack-embed-progress-fill"></div></div>
                  <div class="browse-landing-pack-embed-progress-label"></div>
                </div>
                <div class="browse-landing-pack-embed-error" data-embed-error="${escapeHtml(id)}" style="display:none"></div>
              </div>
            ` : ''}
          </div>
        `;
      }).join('')}
    </div>
  `;
  _attachPackFaviconFallbacks(container);
  const _openPackById = (id) => {
    const pack = packs.find(p => (p.pack_id || p.id) === id);
    if (pack) _openPackView(pack);
  };
  container.querySelectorAll('.browse-landing-pack-main').forEach(btn => {
    btn.addEventListener('click', () => _openPackById(btn.dataset.packId));
  });
  container.querySelectorAll('.browse-landing-pack-action[data-pack-search]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      _openPackById(btn.dataset.packSearch);
    });
  });
  container.querySelectorAll('.browse-landing-pack-action[data-pack-home]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const url = btn.dataset.packHome;
      if (url) browseFetch(url);
    });
  });

  // Embed buttons: opt-in vector index conversion for installed ZIM packs.
  // POSTs to /packs/{pack_id}/embed which returns a job_id, then we attach
  // an EventSource to the same SSE progress stream the failed-conversion
  // resume flow uses. On success, re-render the landing so the pack card
  // updates (has_vector_index becomes true → Embed button hides).
  container.querySelectorAll('.browse-landing-pack-embed').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const packId = btn.dataset.embedPack;
      if (!packId || btn.disabled) return;

      const wrapper = btn.closest('.browse-landing-pack-wrapper');
      const embedRow = wrapper?.querySelector(`[data-embed-row="${CSS.escape(packId)}"]`);
      const progressSlot = wrapper?.querySelector(`[data-embed-progress="${CSS.escape(packId)}"]`);
      const errorSlot = wrapper?.querySelector(`[data-embed-error="${CSS.escape(packId)}"]`);
      const labelEl = progressSlot?.querySelector('.browse-landing-pack-embed-progress-label');

      btn.disabled = true;
      btn.classList.remove('is-error');
      btn.classList.add('is-loading');
      if (errorSlot) { errorSlot.textContent = ''; errorSlot.style.display = 'none'; }
      if (embedRow) embedRow.hidden = false;
      if (progressSlot) progressSlot.style.display = '';
      if (labelEl) labelEl.textContent = 'starting…';

      try {
        const resp = await fetch(`/api/knowledge/packs/${encodeURIComponent(packId)}/embed`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(extractErrorMessage(err, `HTTP ${resp.status}`));
        }
        const { job_id: jobId } = await resp.json();
        if (!jobId) throw new Error('Server did not return a job_id');

        _streamEmbedProgress(jobId, packId, wrapper);
      } catch (err) {
        showToast(`Embed failed: ${err.message}`, 'error');
        btn.disabled = false;
        btn.classList.remove('is-loading');
        btn.classList.add('is-error');
        if (errorSlot) {
          errorSlot.textContent = err.message;
          errorSlot.style.display = '';
        }
        if (progressSlot) progressSlot.style.display = 'none';
      }
    });
  });

  // Discard buttons for failed conversions. Confirmation dialog because
  // it does delete files (the empty shell + progress.json). Original
  // .zim is preserved, but worth being explicit so users don't worry
  // they're losing pack data.
  container.querySelectorAll('.browse-landing-pack-failed-discard').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const packId = btn.dataset.discard;
      if (!packId) return;
      const ok = window.confirm(
        `Discard the failed conversion shell for "${packId}"?\n\nThe original .zim file is preserved.`
      );
      if (!ok) return;
      btn.disabled = true;
      btn.textContent = 'Discarding…';
      try {
        const resp = await fetch(`/api/knowledge/discard-failed/${encodeURIComponent(packId)}`, {
          method: 'POST',
        });
        if (resp.ok) {
          showToast(`Discarded ${packId} conversion shell`, 'success');
          // Re-render so the failed entry disappears + the now-unshadowed
          // ZIM (if any) appears as a real card.
          _renderLandingPacks();
        } else {
          const err = await resp.json().catch(() => ({}));
          showToast(`Discard failed: ${err.detail || resp.status}`, 'error');
          btn.disabled = false;
          btn.textContent = 'Discard';
        }
      } catch (err) {
        showToast(`Discard failed: ${err.message}`, 'error');
        btn.disabled = false;
        btn.textContent = 'Discard';
      }
    });
  });

  // Resume buttons. POST starts a background convert job in --resume mode;
  // we then attach an EventSource to /install/{job_id}/progress and update
  // the inline progress bar until the job completes or errors. On success,
  // re-render the landing so the now-finished pack appears as a real card.
  container.querySelectorAll('.browse-landing-pack-failed-resume').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const packId = btn.dataset.resume;
      if (!packId || btn.disabled) return;

      // Pull the user's batch-size override from the inline input. Worker
      // and route both clamp to [1, 2048]; we still parse defensively here
      // so a blank field doesn't post NaN.
      const batchInput = container.querySelector(`[data-batch-input="${CSS.escape(packId)}"]`);
      const batchSize = batchInput ? parseInt(batchInput.value, 10) : null;
      const requestBody = (Number.isFinite(batchSize) && batchSize > 0)
        ? { batch_size: batchSize }
        : {};

      const card = btn.closest('.browse-landing-pack-failed-card');
      const discardBtn = card?.querySelector('.browse-landing-pack-failed-discard');
      const progressSlot = card?.querySelector(`[data-progress-slot="${CSS.escape(packId)}"]`);
      const errorSlot = card?.querySelector('.browse-landing-pack-failed-error');

      btn.disabled = true;
      btn.textContent = 'Resuming…';
      if (discardBtn) discardBtn.disabled = true;
      if (batchInput) batchInput.disabled = true;
      if (errorSlot) errorSlot.style.display = 'none';

      try {
        const resp = await fetch(`/api/knowledge/resume-failed/${encodeURIComponent(packId)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(extractErrorMessage(err, `HTTP ${resp.status}`));
        }
        const { job_id: jobId } = await resp.json();
        if (!jobId) throw new Error('Server did not return a job_id');

        // Show the progress bar slot and start streaming.
        if (progressSlot) progressSlot.style.display = '';
        _streamResumeProgress(jobId, packId, card);
      } catch (err) {
        showToast(`Resume failed: ${err.message}`, 'error');
        btn.disabled = false;
        btn.textContent = 'Resume';
        if (discardBtn) discardBtn.disabled = false;
        if (batchInput) batchInput.disabled = false;
      }
    });
  });
}

/** Subscribe to the install-progress SSE stream and update the inline
 * progress bar on the failed-conversion card. Re-uses the same endpoint
 * as fresh installs since resume jobs are tracked in the same install_jobs
 * dict server-side. On completion, re-renders the landing to swap the
 * failed card for a real pack tile. */
function _streamResumeProgress(jobId, packId, cardEl) {
  if (!cardEl) return;
  const fillEl = cardEl.querySelector('.browse-landing-pack-failed-progress-fill');
  const labelEl = cardEl.querySelector('.browse-landing-pack-failed-progress-label');
  const resumeBtn = cardEl.querySelector('.browse-landing-pack-failed-resume');
  const discardBtn = cardEl.querySelector('.browse-landing-pack-failed-discard');
  const batchInput = cardEl.querySelector(`[data-batch-input="${CSS.escape(packId)}"]`);
  const errorSlot = cardEl.querySelector('.browse-landing-pack-failed-error');
  const cancelBtn = cardEl.querySelector(`[data-progress-cancel="${CSS.escape(packId)}"]`);

  // Show the cancel button for the duration of this stream and wire its
  // click handler. POST /api/knowledge/install/{job_id}/cancel — server
  // sets the cancel flag on the install_job; the SSE will close with
  // status=cancelled which is handled below.
  if (cancelBtn) {
    cancelBtn.style.display = '';
    cancelBtn.disabled = false;
    cancelBtn.onclick = async () => {
      if (cancelBtn.disabled) return;
      cancelBtn.disabled = true;
      cancelBtn.textContent = 'Cancelling…';
      try {
        await fetch(`/api/knowledge/install/${encodeURIComponent(jobId)}/cancel`, {
          method: 'POST',
          credentials: 'same-origin',
        });
      } catch (err) {
        showToast(`Cancel failed: ${err.message}`, 'error');
        cancelBtn.disabled = false;
        cancelBtn.textContent = 'Cancel';
      }
    };
  }

  let evtSource;
  try {
    evtSource = new EventSource(`/api/knowledge/install/${encodeURIComponent(jobId)}/progress`);
  } catch (err) {
    showToast(`Could not subscribe to resume progress: ${err.message}`, 'error');
    return;
  }

  evtSource.onmessage = (e) => {
    let evt;
    try { evt = JSON.parse(e.data); } catch { return; }
    const stage = evt.stage || 'working';
    const current = Number(evt.current) || 0;
    const total = Number(evt.total) || 0;
    const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;
    if (fillEl) fillEl.style.width = `${pct}%`;
    if (labelEl) {
      if (total > 0) {
        labelEl.textContent = `${stage} — ${current.toLocaleString()} / ${total.toLocaleString()} (${pct}%)`;
      } else {
        labelEl.textContent = stage;
      }
    }

    if (evt.status === 'complete') {
      evtSource.close();
      if (cancelBtn) cancelBtn.style.display = 'none';
      showToast(`Resumed ${packId} successfully`, 'success');
      // Full re-render: the partial augpack is now a real pack and should
      // appear in the grid; the failed-conversion entry should be gone.
      _renderLandingPacks();
    } else if (evt.status === 'error' || evt.status === 'cancelled' || evt.status === 'failed') {
      evtSource.close();
      if (cancelBtn) cancelBtn.style.display = 'none';
      const msg = evt.error || `Resume ${evt.status}`;
      showToast(`Resume failed: ${msg}`, 'error');
      // Restore button states + surface the new error in the existing slot.
      if (resumeBtn) {
        resumeBtn.disabled = false;
        resumeBtn.textContent = 'Resume';
      }
      if (discardBtn) discardBtn.disabled = false;
      if (batchInput) batchInput.disabled = false;
      if (errorSlot) {
        errorSlot.textContent = msg;
        errorSlot.style.display = '';
      }
      // Hide the progress bar so the card returns to its idle look.
      const slot = cardEl.querySelector('.browse-landing-pack-failed-progress');
      if (slot) slot.style.display = 'none';
    }
  };

  evtSource.onerror = () => {
    // Network blip — the server-side job keeps running. Close and let the
    // user re-enter the page if they want a fresh subscription. We don't
    // restore button states here because the resume IS in flight.
    evtSource.close();
    if (labelEl) labelEl.textContent = 'Progress stream interrupted (job continues in background)';
  };
}

/** Subscribe to the install-progress SSE stream for an opt-in embed job
 * (started by clicking "Embed for vector search" on a ZIM-only pack card).
 * Mirrors _streamResumeProgress but operates on the wrapper around a
 * regular pack card rather than the failed-conversion card. */
function _streamEmbedProgress(jobId, packId, wrapperEl) {
  if (!wrapperEl) return;
  const fillEl = wrapperEl.querySelector('.browse-landing-pack-embed-progress-fill');
  const labelEl = wrapperEl.querySelector('.browse-landing-pack-embed-progress-label');
  const embedBtn = wrapperEl.querySelector(`[data-embed-pack="${CSS.escape(packId)}"]`);
  const errorSlot = wrapperEl.querySelector(`[data-embed-error="${CSS.escape(packId)}"]`);
  const progressSlot = wrapperEl.querySelector(`[data-embed-progress="${CSS.escape(packId)}"]`);

  let evtSource;
  try {
    evtSource = new EventSource(`/api/knowledge/install/${encodeURIComponent(jobId)}/progress`);
  } catch (err) {
    showToast(`Could not subscribe to embed progress: ${err.message}`, 'error');
    return;
  }

  evtSource.onmessage = (e) => {
    let evt;
    try { evt = JSON.parse(e.data); } catch { return; }
    const stage = evt.stage || 'working';
    const current = Number(evt.current) || 0;
    const total = Number(evt.total) || 0;
    const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;
    if (fillEl) fillEl.style.width = `${pct}%`;
    if (labelEl) {
      if (total > 0) {
        labelEl.textContent = `${stage} — ${current.toLocaleString()} / ${total.toLocaleString()} (${pct}%)`;
      } else {
        labelEl.textContent = stage;
      }
    }

    if (evt.status === 'complete') {
      evtSource.close();
      showToast(`Embedded ${packId} for vector search`, 'success');
      // Re-render the landing — the pack now has has_vector_index=true,
      // so the Embed button vanishes from its card.
      _renderLandingPacks();
    } else if (evt.status === 'error' || evt.status === 'cancelled' || evt.status === 'failed') {
      evtSource.close();
      const msg = evt.error || `Embed ${evt.status}`;
      showToast(`Embed failed: ${msg}`, 'error');
      if (embedBtn) {
        embedBtn.disabled = false;
        embedBtn.classList.remove('is-loading');
        embedBtn.classList.add('is-error');
      }
      if (errorSlot) {
        errorSlot.textContent = msg;
        errorSlot.style.display = '';
      }
      if (progressSlot) progressSlot.style.display = 'none';
    }
  };

  evtSource.onerror = () => {
    evtSource.close();
    if (labelEl) labelEl.textContent = 'progress stream interrupted (job continues in background)';
  };
}

/** Open the per-pack home view — replaces searchView contents with a
 * focused experience scoped to one pack. Mirrors _openBookmarksView's
 * pattern: header with back button, scoped search input, recently-
 * viewed list (filtered from history), result rendering on demand.
 */
function _openPackView(pack) {
  const container = dom.searchView;
  if (!container) return;
  const packId = pack.pack_id || pack.id || '';
  const name = pack.name || packId;
  const desc = pack.description || '';
  const isZim = pack.type === 'zim';
  const homePath = isZim ? (pack.main_entry_path || '') : '';
  const homeUrl = homePath ? `zim:${packId}/${homePath}` : '';

  // Build "recently viewed in this pack" from session history. Filters
  // entries whose URL starts with zim:<packId>/. Augpack chunks aren't
  // browseable yet, so for non-ZIM packs this stays empty.
  const recents = [];
  if (isZim) {
    const seen = new Set();
    for (let i = state.history.length - 1; i >= 0 && recents.length < 8; i--) {
      const e = state.history[i];
      if (!e || !e.url || !e.url.startsWith(`zim:${packId}/`)) continue;
      if (seen.has(e.url)) continue;
      seen.add(e.url);
      recents.push(e);
    }
  }

  container.innerHTML = `
    <div class="browse-pack-view">
      <div class="browse-pack-header">
        <button class="browse-pack-back" data-action="pack-close" aria-label="Back to Browse">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><polyline points="15 18 9 12 15 6"/></svg>
        </button>
        <div class="browse-pack-title-block">
          <div class="browse-pack-icon">${isZim ? _packFaviconHtml(packId, name, 96, 'large') : '📦'}</div>
          <div>
            <h2 class="browse-pack-title">${escapeHtml(name)}</h2>
            ${desc ? `<p class="browse-pack-desc">${escapeHtml(desc)}</p>` : ''}
          </div>
        </div>
      </div>
      <div class="browse-pack-search">
        <div class="browse-pack-search-wrap">
          <input type="search" class="browse-pack-search-input" placeholder="Search ${escapeHtml(name)}…" autocomplete="off" autocorrect="off" autocapitalize="none" spellcheck="false">
          <div class="browse-pack-suggest" role="listbox" aria-label="Search suggestions" hidden></div>
        </div>
        ${homeUrl ? `
          <button class="browse-pack-home" data-pack-home="${escapeHtml(homeUrl)}" title="Open ${escapeHtml(name)} home page">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16" aria-hidden="true"><path d="M3 12 12 4l9 8"/><path d="M5 10v10h14V10"/></svg>
            <span>Home</span>
          </button>
        ` : ''}
        ${isZim ? `
          <button class="browse-pack-random" data-pack-random title="Random article (press R)" aria-label="Open a random article">
            <svg class="browse-pack-random-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" width="16" height="16" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="3"/>
              <circle cx="8" cy="8" r="1.2" fill="currentColor" stroke="none"/>
              <circle cx="16" cy="8" r="1.2" fill="currentColor" stroke="none"/>
              <circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none"/>
              <circle cx="8" cy="16" r="1.2" fill="currentColor" stroke="none"/>
              <circle cx="16" cy="16" r="1.2" fill="currentColor" stroke="none"/>
            </svg>
            <span>Random</span>
          </button>
        ` : ''}
      </div>
      <div class="browse-pack-results" id="browse-pack-results">
        ${recents.length ? `
          <p class="browse-landing-recents-label">Recently viewed</p>
          <div class="browse-pack-recents">
            ${recents.map(r => `
              <button class="browse-pack-recent" data-pack-url="${escapeHtml(r.url)}">
                <span class="browse-pack-recent-title">${escapeHtml(r.title || r.url)}</span>
              </button>
            `).join('')}
          </div>
        ` : `
          <div class="browse-pack-empty">
            <p>${isZim ? `${homeUrl ? 'Open the home page, search, ' : 'Search '}${escapeHtml(name)}${homeUrl ? ',' : ''} or click a citation in chat to dive in.` : 'This pack is searchable from chat. Browseable view is coming for non-ZIM packs.'}</p>
          </div>
        `}
      </div>
    </div>
  `;

  _attachPackFaviconFallbacks(container);

  // Random article — fetches /_random which returns a 302 redirect to
  // a random HTML article URL. ``response.url`` carries the resolved
  // path after the redirect; we parse it back into a ``zim:`` shape so
  // it flows through the existing navigation stack (history, page tab).
  // 503 means libzim couldn't find a valid candidate in the retry budget
  // (image-only pack, or transient bad luck) — toast and let the user
  // try again rather than surfacing a hard error.
  const _openRandomArticle = async () => {
    const btn = container.querySelector('[data-pack-random]');
    if (btn) btn.classList.add('is-loading');
    try {
      const resp = await fetch(
        `/api/knowledge/zim/${encodeURIComponent(packId)}/_random`,
      );
      if (resp.status === 503) {
        showToast('Try again — random pick missed', 'info');
        return;
      }
      if (!resp.ok) {
        showToast('Could not load a random article', 'error');
        return;
      }
      const match = resp.url.match(/\/api\/knowledge\/zim\/([^\/]+)\/(.+?)(?:\?|$)/);
      if (!match) return;
      const finalPackId = decodeURIComponent(match[1]);
      const entryPath = match[2];
      _renderZimArticle(`zim:${finalPackId}/${entryPath}`);
    } catch {
      showToast('Could not load a random article', 'error');
    } finally {
      if (btn) btn.classList.remove('is-loading');
    }
  };
  container.querySelector('[data-pack-random]')?.addEventListener('click', _openRandomArticle);
  // Keyboard shortcut: ``R`` opens a random article. Gated to skip when
  // an input/textarea/contenteditable has focus (so typing "R" in the
  // search bar doesn't fire). Document-scoped because the pack view's
  // container doesn't keep focus during browsing; ``document.contains``
  // self-cleans on re-renders so we don't leak listeners across navs.
  const _randomKeyHandler = (e) => {
    if (!isZim) return;
    if (e.key !== 'r' && e.key !== 'R') return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const target = e.target;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return;
    if (!document.contains(container)) {
      document.removeEventListener('keydown', _randomKeyHandler);
      return;
    }
    e.preventDefault();
    _openRandomArticle();
  };
  if (isZim) document.addEventListener('keydown', _randomKeyHandler);

  // Wire actions. Back button rebuilds the landing (showSearchView only
  // toggles visibility — without _restoreLanding the searchView would
  // still contain the pack view markup we just wrote).
  container.querySelector('[data-action="pack-close"]')?.addEventListener('click', () => {
    if (state.searchResults) {
      // User had search results before opening the pack view — restore
      // those, not the landing.
      renderSearchResults();
    } else {
      _restoreLanding();
    }
    showSearchView();
  });

  const searchInput = container.querySelector('.browse-pack-search-input');
  const suggestEl = container.querySelector('.browse-pack-suggest');
  if (searchInput) {
    // Two-tier search UX:
    //   * Keystroke → debounced typeahead dropdown (libzim
    //     SuggestionSearcher; instant, narrow, title-prefix). Only
    //     fires for ZIM packs — augpacks have no equivalent index.
    //   * Enter / blur-with-no-suggestion-pick → full pack search
    //     (existing _doPackSearch path; runs the FTS+vector merge).
    // The dropdown is dismissed on item-click, Escape, or blur.
    // ``suggestSeq`` discards stale responses when the user keeps
    // typing while an in-flight request resolves out of order.
    let typeaheadDebounce = null;
    let suggestSeq = 0;

    const hideSuggest = () => {
      if (suggestEl) {
        suggestEl.hidden = true;
        suggestEl.innerHTML = '';
      }
    };

    const fireFullSearch = () => {
      hideSuggest();
      const q = searchInput.value.trim();
      if (!q) {
        // Re-render the recents-only state when query cleared.
        _openPackView(pack);
        return;
      }
      _doPackSearch(packId, q);
    };

    const fireTypeahead = async () => {
      // Augpacks have no per-pack browseable surface yet — typeahead
      // would point at non-existent entries. Skip silently.
      if (!isZim || !suggestEl) {
        hideSuggest();
        return;
      }
      const q = searchInput.value.trim();
      if (q.length < 2) {
        hideSuggest();
        return;
      }
      const seq = ++suggestSeq;
      try {
        const url = `/api/knowledge/zim/${encodeURIComponent(packId)}`
          + `/_suggest?q=${encodeURIComponent(q)}&limit=8`;
        const resp = await fetch(url);
        if (!resp.ok) { hideSuggest(); return; }
        const data = await resp.json();
        // Race guard: a newer keystroke already kicked off a fresh
        // request — drop this stale response so the user never sees
        // suggestions for a query they've moved past.
        if (seq !== suggestSeq) return;
        const items = Array.isArray(data?.suggestions) ? data.suggestions : [];
        if (!items.length) { hideSuggest(); return; }
        suggestEl.innerHTML = items.map((s) => `
          <button class="browse-pack-suggest-item" type="button" role="option"
                  data-zim-path="${escapeHtml(s.path)}"
                  title="${escapeHtml(s.title)}">
            <span class="browse-pack-suggest-title">${escapeHtml(s.title)}</span>
            <span class="browse-pack-suggest-path">${escapeHtml(s.path)}</span>
          </button>
        `).join('');
        suggestEl.hidden = false;
        suggestEl.querySelectorAll('.browse-pack-suggest-item').forEach((btn) => {
          btn.addEventListener('mousedown', (e) => {
            // mousedown (not click) so the input's blur — which would
            // otherwise hide the dropdown before the click resolves —
            // doesn't race the navigation.
            e.preventDefault();
            const p = btn.dataset.zimPath;
            if (!p) return;
            hideSuggest();
            searchInput.value = '';
            browseFetch(`zim:${packId}/${p}`);
          });
        });
      } catch {
        hideSuggest();
      }
    };

    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        fireFullSearch();
        return;
      }
      if (e.key === 'Escape') {
        if (suggestEl && !suggestEl.hidden) {
          // First Escape dismisses the dropdown only; second Escape
          // (with dropdown already hidden) closes the pack view.
          e.stopPropagation();
          hideSuggest();
          return;
        }
        showSearchView();
        if (state.searchResults) renderSearchResults();
      }
    });

    searchInput.addEventListener('input', () => {
      if (typeaheadDebounce) clearTimeout(typeaheadDebounce);
      // 180ms balances responsiveness with avoiding a request per
      // keystroke during typing bursts. libzim's suggester is fast
      // (microseconds for prefix lookup) but the round-trip itself
      // dominates, so the debounce is mostly about deduping.
      typeaheadDebounce = setTimeout(fireTypeahead, 180);
    });

    // Hide on blur with a short delay so click-on-item lands first.
    // (mousedown handler above pre-empts the blur in modern browsers,
    // but the timeout is a defensive belt-and-suspenders.)
    searchInput.addEventListener('blur', () => {
      setTimeout(hideSuggest, 150);
    });
  }

  container.querySelectorAll('.browse-pack-recent').forEach(btn => {
    btn.addEventListener('click', () => {
      const url = btn.dataset.packUrl;
      if (url) browseFetch(url);
    });
  });

  container.querySelector('[data-pack-home]')?.addEventListener('click', (e) => {
    const url = e.currentTarget.dataset.packHome;
    if (url) browseFetch(url);
  });
}

/** Run a pack-scoped search and render results inline. Calls
 * /api/knowledge/search with pack_ids filter so we don't pull in
 * results from other packs. Click result → standard browseFetch path
 * with the zim: URL scheme that _renderZimArticle handles.
 */
async function _doPackSearch(packId, query) {
  const resultsEl = document.getElementById('browse-pack-results');
  if (!resultsEl) return;
  resultsEl.innerHTML = `
    <div class="browse-loading" style="padding:24px 0">
      <div class="browse-skeleton browse-skeleton-line"></div>
      <div class="browse-skeleton browse-skeleton-line"></div>
      <div class="browse-skeleton browse-skeleton-line" style="width:70%"></div>
    </div>
  `;
  let data;
  try {
    const url = `/api/knowledge/search?q=${encodeURIComponent(query)}&pack_ids=${encodeURIComponent(packId)}&limit=12`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Status ${resp.status}`);
    data = await resp.json();
  } catch (err) {
    resultsEl.innerHTML = `<div class="browse-pack-empty"><p>Search failed: ${escapeHtml(err.message)}</p></div>`;
    return;
  }
  const results = Array.isArray(data.results) ? data.results : [];
  if (!results.length) {
    resultsEl.innerHTML = `<div class="browse-pack-empty"><p>No matches in this pack.</p></div>`;
    return;
  }

  // Detect if any results are non-browseable (augpack chunks). Show a
  // single header note instead of disabled buttons users can't reason
  // about.
  const hasNonBrowseable = results.some(r => !(r.source === 'zim' && r.url));
  resultsEl.innerHTML = `
    <p class="browse-landing-recents-label">${results.length} result${results.length === 1 ? '' : 's'}</p>
    ${hasNonBrowseable ? `
      <div class="browse-pack-note">
        These results power chat retrieval. A standalone article view for this pack format is coming soon.
      </div>
    ` : ''}
    <div class="browse-pack-results-list">
      ${results.map(r => {
        const isZim = r.source === 'zim' && r.url;
        const url = isZim ? `zim:${r.pack_id}/${r.url}` : '';
        const title = r.title || r.url || 'Untitled';
        const section = r.section || '';
        const preview = (r.content || '').slice(0, 220).replace(/\s+/g, ' ').trim();
        return `
          <${url ? 'button' : 'div'} class="browse-pack-result${url ? '' : ' browse-pack-result--readonly'}" ${url ? `data-pack-url="${escapeHtml(url)}"` : ''}>
            <div class="browse-pack-result-head">
              <span class="browse-pack-result-title">${escapeHtml(title)}</span>
              ${section ? `<span class="browse-pack-result-section">${escapeHtml(section)}</span>` : ''}
            </div>
            ${preview ? `<div class="browse-pack-result-preview">${escapeHtml(preview)}…</div>` : ''}
          </${url ? 'button' : 'div'}>
        `;
      }).join('')}
    </div>
  `;

  resultsEl.querySelectorAll('.browse-pack-result[data-pack-url]').forEach(btn => {
    btn.addEventListener('click', () => browseFetch(btn.dataset.packUrl));
  });
}

async function _renderLandingBookmarks() {
  const container = document.getElementById('browse-landing-bookmarks');
  if (!container) return;
  const items = await bookmarks.list();
  if (!items.length) {
    container.innerHTML = '';
    return;
  }
  const recent = items.slice(0, 6);
  const more = items.length > 6 ? ` <button class="browse-landing-bookmarks-seeall">See all (${items.length})</button>` : '';
  container.innerHTML = `
    <p class="browse-landing-recents-label">Saved${more}</p>
    <div class="browse-landing-bookmarks-grid">
      ${recent.map(b => `
        <button class="browse-landing-bookmark" data-bm-url="${escapeHtml(b.url)}" title="${escapeHtml(b.title)}">
          ${b.favicon
            ? `<img class="browse-landing-bookmark-favicon" src="${escapeHtml(b.favicon)}" alt="" loading="lazy" onerror="this.style.display='none'">`
            : '<span class="browse-landing-bookmark-favicon-placeholder">\u25CF</span>'}
          <span class="browse-landing-bookmark-title">${escapeHtml(b.title)}</span>
        </button>
      `).join('')}
    </div>
  `;
  container.querySelectorAll('.browse-landing-bookmark').forEach(btn => {
    btn.addEventListener('click', () => browseFetch(btn.dataset.bmUrl));
  });
  container.querySelector('.browse-landing-bookmarks-seeall')?.addEventListener('click', () => _openBookmarksView());
}

/**
 * Replaces the landing content with the full bookmarks grid + search
 * filter. Click a bookmark to open it; click the close button to go
 * back to the standard landing. Lightweight: re-renders on every
 * bookmarks-changed event so add/remove elsewhere stays in sync.
 */
async function _openBookmarksView() {
  const container = dom.searchView;
  if (!container) return;
  const items = await bookmarks.list();

  const renderGrid = (filter = '') => {
    const f = filter.trim().toLowerCase();
    const filtered = f
      ? items.filter(b =>
          (b.title || '').toLowerCase().includes(f)
          || (b.url || '').toLowerCase().includes(f)
          || (b.snippet || '').toLowerCase().includes(f)
        )
      : items;
    if (!filtered.length) {
      return `<div class="browse-bookmarks-empty">${f ? 'No bookmarks match.' : 'No bookmarks yet. Click the bookmark button on any article to save it.'}</div>`;
    }
    return `<div class="browse-bookmarks-grid">
      ${filtered.map(b => {
        const date = new Date(b.savedAt || 0);
        const ageDisplay = _relativeDate(date);
        const host = (() => { try { return new URL(b.url).hostname; } catch { return ''; } })();
        return `<div class="browse-bookmark-card" data-bm-url="${escapeHtml(b.url)}">
          <div class="browse-bookmark-head">
            ${b.favicon ? `<img class="browse-bookmark-favicon" src="${escapeHtml(b.favicon)}" alt="" loading="lazy" onerror="this.style.display='none'">` : ''}
            <span class="browse-bookmark-host">${escapeHtml(host)}</span>
            <span class="browse-bookmark-age">${escapeHtml(ageDisplay)}</span>
            <button class="browse-bookmark-remove" data-bm-remove="${escapeHtml(b.url)}" title="Remove bookmark" aria-label="Remove bookmark">&times;</button>
          </div>
          <h3 class="browse-bookmark-title">${escapeHtml(b.title || b.url)}</h3>
          ${b.snippet ? `<p class="browse-bookmark-snippet">${escapeHtml(b.snippet)}</p>` : ''}
        </div>`;
      }).join('')}
    </div>`;
  };

  container.innerHTML = `
    <div class="browse-bookmarks-header">
      <h2 class="browse-bookmarks-title">Saved <span class="browse-bookmarks-count">${items.length}</span></h2>
      <div class="browse-bookmarks-controls">
        <input type="search" class="browse-bookmarks-search" placeholder="Filter bookmarks…" />
        <button class="browse-bookmarks-export" title="Export bookmarks as JSON">Export</button>
        <button class="browse-bookmarks-close" title="Back to search">&times;</button>
      </div>
    </div>
    <div id="browse-bookmarks-body">${renderGrid()}</div>
  `;

  const body = container.querySelector('#browse-bookmarks-body');
  const search = container.querySelector('.browse-bookmarks-search');
  let filterDebounce;

  search?.addEventListener('input', () => {
    clearTimeout(filterDebounce);
    filterDebounce = setTimeout(() => {
      body.innerHTML = renderGrid(search.value);
      wireCards();
    }, 120);
  });

  const wireCards = () => {
    body.querySelectorAll('.browse-bookmark-card').forEach(card => {
      card.addEventListener('click', (e) => {
        if (e.target.closest('[data-bm-remove]')) return;
        browseFetch(card.dataset.bmUrl);
      });
    });
    body.querySelectorAll('[data-bm-remove]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const url = btn.dataset.bmRemove;
        await bookmarks.remove(url);
        _openBookmarksView();  // re-render with fresh list
      });
    });
  };
  wireCards();

  container.querySelector('.browse-bookmarks-close')?.addEventListener('click', () => {
    // Same fix the pack-close uses — _restoreLanding rebuilds the
    // landing markup that this view overwrote.
    if (state.searchResults) renderSearchResults();
    else _restoreLanding();
    showSearchView();
  });
  container.querySelector('.browse-bookmarks-export')?.addEventListener('click', async () => {
    const data = await bookmarks.list();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `augmentum-bookmarks-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
}

function _relativeDate(date) {
  if (!date || !date.getTime()) return '';
  const now = Date.now();
  const diff = now - date.getTime();
  const day = 86400000;
  if (diff < day) return 'today';
  if (diff < 2 * day) return 'yesterday';
  if (diff < 7 * day) return `${Math.floor(diff / day)} days ago`;
  if (diff < 30 * day) return `${Math.floor(diff / (7 * day))} weeks ago`;
  if (diff < 365 * day) return `${Math.floor(diff / (30 * day))} months ago`;
  return `${Math.floor(diff / (365 * day))} years ago`;
}

// Re-render the landing bookmark strip when the set changes from
// anywhere (toolbar toggle, bulk action, etc.)
document.addEventListener('augmentum:bookmarks-changed', () => {
  _renderLandingBookmarks();
});

function _renderLandingRecents() {
  const container = document.getElementById('browse-landing-recents');
  if (!container) return;
  const recents = _getSearchHistory().slice(0, 5);
  if (!recents.length) {
    container.innerHTML = '';
    return;
  }
  const removeIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  container.innerHTML = `
    <p class="browse-landing-recents-label">Recent</p>
    <div class="browse-landing-recents-list">
      ${recents.map(q => `
        <div class="browse-landing-recent" data-query="${escapeHtml(q)}" role="button" tabindex="0">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
          <span class="browse-landing-recent-query">${escapeHtml(q)}</span>
          <button class="browse-landing-recent-remove" title="Remove" aria-label="Remove from recent searches" data-remove="${escapeHtml(q)}">${removeIcon}</button>
        </div>
      `).join('')}
    </div>
    <div class="browse-landing-recents-footer">
      <button class="browse-landing-recents-clear">Clear history</button>
    </div>
  `;
  container.onclick = (e) => {
    const removeBtn = e.target.closest('.browse-landing-recent-remove');
    if (removeBtn) {
      e.stopPropagation();
      const items = _getSearchHistory().filter(q => q !== removeBtn.dataset.remove);
      _setSearchHistory(items);
      _renderLandingRecents();
      return;
    }
    if (e.target.closest('.browse-landing-recents-clear')) {
      _setSearchHistory([]);
      _renderLandingRecents();
      return;
    }
    const item = e.target.closest('.browse-landing-recent');
    if (item) {
      const query = item.dataset.query;
      if (dom.searchInput) dom.searchInput.value = query;
      browseSearch(query);
    }
  };
  container.onkeydown = (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const item = e.target.closest('.browse-landing-recent');
    if (!item || e.target.closest('.browse-landing-recent-remove')) return;
    e.preventDefault();
    const query = item.dataset.query;
    if (dom.searchInput) dom.searchInput.value = query;
    browseSearch(query);
  };
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
function switchTab(tab) {
  if (state.splitMode && (tab === 'browse' || tab === 'notes')) {
    // In split mode both browse and notes are visible; tab click just
    // reassigns which pane the shared search bar targets.
    state.activeTab = tab;
    try { localStorage.setItem('augmentum_browse_tab', tab); } catch {}
    dom.tabBrowse?.classList.toggle('active', tab === 'browse');
    dom.tabNotes?.classList.toggle('active', tab === 'notes');
    dom.tabDiscovery?.classList.toggle('active', false);
    _updateSearchPlaceholder(tab);
    if (tab === 'notes' && state.notes.length === 0) loadNotes();
    return;
  }

  // Discovery is not splittable; clicking it while in split mode exits
  // split for the duration of this view. The preference stays saved so
  // re-entering browse/notes restores the split layout.
  if (tab === 'discovery' && dom.panel?.classList.contains('browse-split-mode')) {
    dom.panel.classList.remove('browse-split-mode');
  }

  // Find the currently visible panel for fade-out
  const prevTab = state.activeTab;
  state.activeTab = tab;
  try { localStorage.setItem('augmentum_browse_tab', tab); } catch {}

  // Update tab buttons
  dom.tabBrowse?.classList.toggle('active', tab === 'browse');
  dom.tabNotes?.classList.toggle('active', tab === 'notes');
  dom.tabDiscovery?.classList.toggle('active', tab === 'discovery');

  // Fade transition between browse/notes panels
  const prevPanel = _getTabPanel(prevTab);
  const nextPanel = _getTabPanel(tab);

  if (prevPanel && prevPanel !== nextPanel) {
    prevPanel.style.opacity = '0';
    prevPanel.style.transition = 'opacity 0.15s ease';
    setTimeout(() => {
      _hideTabPanel(prevTab);
      prevPanel.style.opacity = '';
      prevPanel.style.transition = '';
      _showTabPanel(tab);
    }, 150);
  } else {
    _hideTabPanel(prevTab);
    _showTabPanel(tab);
  }

  // Discovery tab
  if (tab === 'discovery') {
    const container = dom.bodyDiscovery;
    if (container && window._discovery) {
      if (!container.dataset.inited) {
        window._discovery.init(container);
        container.dataset.inited = '1';
      }
      window._discovery.show();
    }
  } else {
    if (window._discovery) window._discovery.hide();
  }

  _updateSearchPlaceholder(tab);

  // Load notes on first switch
  if (tab === 'notes' && state.notes.length === 0) {
    loadNotes();
  }

  // Re-apply split layout after the fade settles. If the user had split
  // on, switched to discovery (which exited split), and is now switching
  // back to browse/notes, this restores the side-by-side view.
  if ((tab === 'browse' || tab === 'notes') && state.splitMode) {
    setTimeout(_applySplitMode, 160);   // After the 150ms fade
  }
}

function _getTabPanel(tab) {
  if (tab === 'browse') return dom.bodyBrowse;
  if (tab === 'notes') return dom.bodyNotes;
  return null;
}

function _hideTabPanel(tab) {
  if (tab === 'browse' && dom.bodyBrowse) dom.bodyBrowse.style.display = 'none';
  if (tab === 'notes') dom.bodyNotes?.classList.add('hidden');
}

function _showTabPanel(tab) {
  if (tab === 'browse' && dom.bodyBrowse) {
    dom.bodyBrowse.style.display = '';
    _fadeInPanel(dom.bodyBrowse);
  }
  if (tab === 'notes') {
    dom.bodyNotes?.classList.remove('hidden');
    _fadeInPanel(dom.bodyNotes);
  }
}

function _fadeInPanel(el) {
  if (!el) return;
  el.style.opacity = '0';
  requestAnimationFrame(() => {
    el.style.transition = 'opacity 0.15s ease';
    el.style.opacity = '1';
    setTimeout(() => { el.style.transition = ''; }, 150);
  });
}

function _updateSearchPlaceholder(tab) {
  if (!dom.searchInput) return;
  if (tab === 'browse') {
    dom.searchInput.placeholder = 'Search the web or enter a URL...';
    dom.searchInput.value = state.searchQuery || '';
  } else if (tab === 'notes') {
    dom.searchInput.placeholder = 'Filter notes...';
    dom.searchInput.value = state.notesFilter || '';
  } else {
    dom.searchInput.placeholder = 'Discovery';
    dom.searchInput.value = '';
  }
  dom.searchBar?.classList.toggle('notes-mode', tab === 'notes');
}

function _submitHeaderSearch() {
  const value = dom.searchInput?.value || '';
  if (state.activeTab === 'notes') {
    state.notesFilter = value.trim();
    renderNotesList();
    return;
  }
  if (state.activeTab === 'browse') {
    browseSearch(value);
  }
}

// ---------------------------------------------------------------------------
// Split-screen mode (browse + notes both visible)
// ---------------------------------------------------------------------------
// Apply the current state.splitMode to the DOM. Factored out of the
// toggle handler so the panel-open restore path can reuse the same
// logic. The media query at >=1201px provides the side-by-side layout;
// below that width the same class falls back to the natural column
// flex of .browse-body-container, which stacks browse on top and
// notes on the bottom — still useful, so we don't collapse on resize.
function _applySplitMode() {
  dom.panel?.classList.toggle('browse-split-mode', state.splitMode);

  if (state.splitMode) {
    if (dom.bodyBrowse) dom.bodyBrowse.style.display = '';
    dom.bodyNotes?.classList.remove('hidden');
    if (state.notes.length === 0) loadNotes();
  } else {
    // Tab mode — show only the active tab (discovery handled by switchTab)
    const tab = state.activeTab || 'browse';
    if (tab === 'browse') {
      if (dom.bodyBrowse) dom.bodyBrowse.style.display = '';
      dom.bodyNotes?.classList.add('hidden');
    } else if (tab === 'notes') {
      if (dom.bodyBrowse) dom.bodyBrowse.style.display = 'none';
      dom.bodyNotes?.classList.remove('hidden');
    } else {
      if (dom.bodyBrowse) dom.bodyBrowse.style.display = 'none';
      dom.bodyNotes?.classList.add('hidden');
    }
  }

  _applyNotesHistoryVisibility();
}

function _toggleSplitMode() {
  state.splitMode = !state.splitMode;
  try { localStorage.setItem('augmentum_browse_split', state.splitMode ? '1' : '0'); } catch {}
  _persistBrowseSettingsPatch({ browseDefaultSplit: state.splitMode });
  _applySplitMode();
}

function _applyNotesHistoryVisibility() {
  const canCollapse = !!(state.splitMode && state.activeNoteId);
  const collapsed = !!(canCollapse && state.notesHistoryCollapsed);

  dom.bodyNotes?.classList.toggle('notes-history-collapsed', collapsed);
  if (!dom.notesHistoryToggle) return;

  dom.notesHistoryToggle.hidden = !canCollapse;
  dom.notesHistoryToggle.setAttribute('aria-pressed', collapsed ? 'true' : 'false');
  dom.notesHistoryToggle.setAttribute(
    'aria-label',
    collapsed ? 'Show notes list' : 'Hide notes list',
  );
  dom.notesHistoryToggle.title = collapsed ? 'Show notes list' : 'Hide notes list';
}

function _toggleNotesHistory() {
  state.notesHistoryCollapsed = !state.notesHistoryCollapsed;
  try {
    localStorage.setItem(
      'augmentum_browse_notes_history_collapsed',
      state.notesHistoryCollapsed ? '1' : '0',
    );
  } catch { /* private browsing / quota — server-side persist below is authoritative */ }
  _persistBrowseSettingsPatch({ browseNotesHistoryCollapsed: state.notesHistoryCollapsed });
  _applyNotesHistoryVisibility();
}

// ---------------------------------------------------------------------------
// Panel open/close
// ---------------------------------------------------------------------------
export function openBrowsePanel(options = {}) {
  // ``skipAutoFocus`` is for callers (like openInBrowse) that are about
  // to load content immediately — without it, focusing the search input
  // fires the focus handler that opens the recent-searches dropdown,
  // which then flashes briefly before the article render covers it.
  const { skipAutoFocus = false } = options;

  if (state.panelOpen) return;
  state.panelOpen = true;
  dom.panel?.classList.remove('hidden');

  _syncBrowsePrefsFromSettings();
  // Restore split/reader preferences from Settings before painting the panel.
  _applySplitMode();

  // Always open to the home/search view across sessions. The previous
  // behavior re-fetched the URL the user had active before app reload,
  // but those pages were usually days old and unrelated to the current
  // task — history is one click away if they want it back. Mid-session
  // reopens (panel just toggled hidden) keep their live content because
  // state.currentUrl is still set in memory.
  const hasLiveContent = state.currentUrl != null;
  // Desktop only — focus on mobile forces the virtual keyboard open.
  if (_canAutoFocusEditable() && window.innerWidth >= 768 && !hasLiveContent && !skipAutoFocus) {
    _autoFocusSearchInput();
  }
  if (!hasLiveContent && state.activeTab === 'browse') {
    showSearchView();
  }
  try { localStorage.setItem('augmentum_browse_open', '1'); } catch {}
  // Track in ViewStack so the close path routes restore-focus through a
  // single coordinator (instead of the ad-hoc augmentum:feature-closed
  // event, which only snapped the orb back but left chat-input unfocused).
  ViewStack.pushOverlay('browse', { onClose: _doCloseBrowsePanel });
  // Re-sync scroll-jump visibility now that the panel has layout.
  // While the panel was display:none, scrollHeight/clientHeight were 0
  // so the initial pass during init() correctly produced "no buttons".
  requestAnimationFrame(updateScrollJumpButtons);
}

export function closeBrowsePanel() {
  if (ViewStack.hasOverlay('browse')) {
    ViewStack.popOverlay('browse');  // onClose → _doCloseBrowsePanel
    return;
  }
  _doCloseBrowsePanel();
}

function _doCloseBrowsePanel() {
  // Companion presence: the page is no longer on screen — clear the
  // "this page" referent so stale deixis doesn't bind to it.
  import('./architect-observer.js')
    .then(m => m.reportAttention('surface.browse.page_closed', {}))
    .catch(() => {});
  import('./companion-context.js')
    .then(m => m.clearCompanionLoadable('page'))
    .catch(() => {});
  // The page-scoped app-menu actions just went dark — re-sync liveness.
  import('./command-palette.js')
    .then(m => m.refreshAgentCatalog())
    .catch(() => {});
  // Abort in-flight work + clean up local state regardless of mode.
  if (state.aiAbort) {
    state.aiAbort.abort();
    state.aiAbort = null;
  }
  try { const ytMod = window._ytPanel; if (ytMod?.close) ytMod.close(); } catch {}
  // Inline video/audio embeds rendered by renderArticle aren't owned by
  // _ytPanel — stop them explicitly so closing the panel doesn't leave
  // audio playing in a hidden DOM subtree.
  _stopReaderMedia();
  flushNoteSave();

  // When browse is mounted as a surface tab (drag-opened), the panel lives
  // inside a LayoutManager container, not in #app. Closing means destroying
  // the tab; otherwise we'd leave an orphaned tab/container behind and the
  // panel's hidden class would have no visible effect. Legacy overlay close
  // (feature-panel mode) still just hides the panel.
  const browseSurface = SurfaceRegistry.ofType('browse')[0];
  if (browseSurface) {
    LayoutManager.closeSurface(browseSurface.id);
    state.panelOpen = false;
    try { localStorage.removeItem('augmentum_browse_open'); localStorage.removeItem('augmentum_browse_tab'); } catch {}
    return;
  }

  if (!state.panelOpen) return;
  state.panelOpen = false;
  dom.panel?.classList.add('hidden');
  try { localStorage.removeItem('augmentum_browse_open'); localStorage.removeItem('augmentum_browse_tab'); } catch {}
  // Notify orb nav that the feature panel closed — snap back to mode
  document.dispatchEvent(new CustomEvent('augmentum:feature-closed', { detail: { feature: 'browse' } }));
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
async function browseSearch(query, category, loadMore = false) {
  query = (query || '').trim();
  if (!query) return;

  // If on notes tab, filter notes from header search bar
  if (state.activeTab === 'notes') {
    state.notesFilter = query;
    renderNotesList();
    return;
  }

  if (category) state.activeCategory = category;

  // Expand provider-filter chip selection (if any) into a site: prefix.
  // _injectProviderFilter also syncs state.activeProvider if the user
  // typed site:X themselves so the chip highlights match their query.
  const effectiveQuery = _injectProviderFilter(query);

  if (!loadMore) {
    state.searchQuery = query;  // store the display form (without our site: prefix)
    state.searchPage = 1;
    state.historyIdx = -1;
    dom.searchInput.value = query;
    _saveSearchQuery(query);
    // Persist the active provider so it survives reload.
    try {
      if (state.activeProvider) {
        localStorage.setItem('augmentum_browse_provider', state.activeProvider);
      } else {
        localStorage.removeItem('augmentum_browse_provider');
      }
    } catch { /* ignore quota */ }
    showSearchView();
    showSearchLoading();
  } else {
    state.searchPage = (state.searchPage || 1) + 1;
  }

  try {
    const params = new URLSearchParams({
      q: effectiveQuery, categories: state.activeCategory, page: state.searchPage,
    });
    // Video-surface filters — only forwarded when set, so /api/browse/search
    // sees no-op defaults for non-video searches and stays unchanged.
    if (state.videoTimeRange) params.set('time_range', state.videoTimeRange);
    if (state.videoDuration)  params.set('duration', state.videoDuration);
    if (state.videoSortBy)    params.set('sort_by', state.videoSortBy);
    const resp = await fetch(`/api/browse/search?${params}`);
    const data = await resp.json();

    if (!resp.ok) {
      showSearchError(data.error || 'Search failed');
      return;
    }

    if (loadMore) {
      state.searchResults = [...(state.searchResults || []), ...(data.results || [])];
    } else {
      state.searchResults = data.results || [];
    }
    state.hasMoreResults = data.has_more || false;
    renderSearchResults();
    _emitSignal('search_query', {
      source_url: '',
      source_title: query,
      content_type: 'search',
      weight: 0.5,
      metadata: { category: category || 'general' },
    });
  } catch (err) {
    showSearchError(`Search failed: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Search History (localStorage)
// ---------------------------------------------------------------------------

// Per-user storage key — `_searchHistoryKey()` returns null when no
// user is known, in which case the typeahead dropdown is empty and
// writes are skipped. The old global key let Profile A's search
// queries surface in Profile B's autocomplete on the same browser.
const _SEARCH_HISTORY_KEY_BASE = 'augmentum:browse_search_history';
const _SEARCH_HISTORY_MAX = 15;

function _searchHistoryKey() {
  const u = getCurrentUser();
  return u && u.id ? `${_SEARCH_HISTORY_KEY_BASE}::u:${u.id}` : null;
}

function _getSearchHistory() {
  const key = _searchHistoryKey();
  if (!key) return [];
  try {
    return JSON.parse(localStorage.getItem(key) || '[]');
  } catch { return []; }
}

function _setSearchHistory(items) {
  const key = _searchHistoryKey();
  if (!key) return;
  try {
    localStorage.setItem(key, JSON.stringify(items));
  } catch (e) {
    // QuotaExceededError typically — trim to the 20 newest entries and
    // retry. If that still fails, give up and warn. Search history is
    // disposable, but silently throwing used to crash the save path.
    console.warn('browse: search history save failed, retrying with trimmed list', e);
    try {
      localStorage.setItem(key, JSON.stringify((items || []).slice(0, 20)));
    } catch { /* best-effort */ }
  }
}

function _saveSearchQuery(query) {
  query = (query || '').trim();
  if (!query) return;
  let items = _getSearchHistory();
  // Remove duplicate (case-insensitive) then prepend
  items = items.filter(q => q.toLowerCase() !== query.toLowerCase());
  items.unshift(query);
  if (items.length > _SEARCH_HISTORY_MAX) items.length = _SEARCH_HISTORY_MAX;
  _setSearchHistory(items);
}

function _removeSearchQuery(query) {
  let items = _getSearchHistory();
  items = items.filter(q => q !== query);
  _setSearchHistory(items);
  _showSearchHistory(); // re-render
}

function _clearSearchHistory() {
  const key = _searchHistoryKey();
  if (key) localStorage.removeItem(key);
  _closeSearchHistory();
}

function _showSearchHistory() {
  const container = dom.searchHistory;
  if (!container) return;

  const items = _getSearchHistory();
  if (!items.length) {
    _closeSearchHistory();
    return;
  }

  const clockIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>';
  const removeIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';

  let html = '';
  for (const q of items) {
    html += `<div class="browse-search-history-item" data-query="${escapeHtml(q)}">
      ${clockIcon}
      <span class="browse-search-history-query">${escapeHtml(q)}</span>
      <button class="browse-search-history-remove" title="Remove" data-remove="${escapeHtml(q)}">${removeIcon}</button>
    </div>`;
  }
  html += `<div class="browse-search-history-footer">
    <button class="browse-search-history-clear">Clear history</button>
  </div>`;

  container.innerHTML = html;
  container.classList.add('open');

  // Click handlers (delegated)
  container.onclick = (e) => {
    const removeBtn = e.target.closest('.browse-search-history-remove');
    if (removeBtn) {
      e.stopPropagation();
      _removeSearchQuery(removeBtn.dataset.remove);
      return;
    }
    const clearBtn = e.target.closest('.browse-search-history-clear');
    if (clearBtn) {
      _clearSearchHistory();
      return;
    }
    const item = e.target.closest('.browse-search-history-item');
    if (item) {
      const query = item.dataset.query;
      dom.searchInput.value = query;
      _closeSearchHistory();
      browseSearch(query);
    }
  };
}

function _closeSearchHistory() {
  dom.searchHistory?.classList.remove('open');
}

/**
 * Stop any media playing inside the reader view. Hiding the reader via
 * CSS leaves <video>/<audio>/<iframe> elements live in the DOM and they
 * keep playing audio — the back button and close button both route
 * through here so the user's "leave the page" intent actually silences
 * whatever they were watching. iframe src is nulled to 'about:blank'
 * because just removing the attribute leaves YouTube's player state
 * intact; blanking it forces an unload.
 */
function _stopReaderMedia() {
  const root = dom.readerView;
  if (!root) return;
  root.querySelectorAll('video, audio').forEach(el => {
    try { el.pause(); } catch {}
  });
  root.querySelectorAll('iframe').forEach(el => {
    try { el.src = 'about:blank'; } catch {}
  });
}

function showSearchView() {
  _stopReaderMedia();
  if (dom.searchView) dom.searchView.style.display = '';
  if (dom.readerView) dom.readerView.style.display = 'none';
  if (dom.readerAskBar) dom.readerAskBar.style.display = 'none';
  if (dom.readerAiBlocks) dom.readerAiBlocks.innerHTML = '';
  // Refresh landing page recents when returning to empty state
  _renderLandingRecents();
  updateNavButtons();
}

function showReaderView() {
  if (dom.searchView) dom.searchView.style.display = 'none';
  if (dom.readerView) dom.readerView.style.display = '';
  if (dom.readerAskBar) dom.readerAskBar.style.display = '';
  updateNavButtons();
}

function showSearchLoading() {
  const container = dom.searchView;
  if (!container) return;
  container.innerHTML = `
    ${renderSearchControls()}
    <div class="browse-loading">
      <div class="browse-skeleton browse-skeleton-title"></div>
      <div class="browse-skeleton browse-skeleton-meta"></div>
      <div class="browse-skeleton browse-skeleton-line"></div>
      <div class="browse-skeleton browse-skeleton-line"></div>
      <div class="browse-skeleton browse-skeleton-title" style="width:70%"></div>
      <div class="browse-skeleton browse-skeleton-meta"></div>
      <div class="browse-skeleton browse-skeleton-line"></div>
      <div class="browse-skeleton browse-skeleton-line"></div>
    </div>
  `;
  wireSearchControls(container);
}

function showSearchError(msg) {
  const container = dom.searchView;
  if (!container) return;
  container.innerHTML = `
    ${renderSearchControls()}
    <div class="browse-error">
      <div class="browse-error-icon browse-error-icon--warning">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="28" height="28">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
      </div>
      <h3 class="browse-error-title">Search unavailable</h3>
      <p class="browse-error-message">${escapeHtml(msg)}</p>
      <button class="browse-error-retry" data-action="retry-search">Try again</button>
    </div>
  `;
  wireSearchControls(container);
  container.querySelector('[data-action="retry-search"]')?.addEventListener('click', () => {
    if (state.searchQuery) browseSearch(state.searchQuery);
  });
}

function renderCategoryTabs() {
  const cats = ['general', 'news', 'science', 'it', 'images', 'videos'];
  return cats.map(c =>
    `<button class="browse-category-btn${c === state.activeCategory ? ' active' : ''}" data-cat="${c}">${escapeHtml(c)}</button>`
  ).join('');
}

function wireCategories(container) {
  container.querySelectorAll('.browse-category-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      browseSearch(state.searchQuery, btn.dataset.cat);
    });
  });
}

/**
 * Render the full search-controls stack — category row + provider chip
 * row + (for videos) the video filter bar. Used at the top of every
 * search-view render so all three stay in sync with state.
 */
function renderSearchControls() {
  return (
    `<div class="browse-categories">${renderCategoryTabs()}</div>`
    + `<div class="browse-providers">${renderProviderChips()}</div>`
    + renderVideoFilters()
  );
}

function wireSearchControls(container) {
  wireCategories(container);
  wireProviderChips(container);
  wireVideoFilters(container);
}

// ---------------------------------------------------------------------------
// Video filter bar — Time / Duration / Sort
//
// Shown only when the active category is videos (or every result is from
// an embeddable video host — flipped by renderSearchResults via the
// `browse-video-filters` element). Mirrors the filter idiom users know
// from YouTube and Google Video search: compact horizontal bar with
// three labeled dropdowns. Native <select> for accessibility + mobile
// affordance — custom popovers would add ~150 LOC for no UX win.
// ---------------------------------------------------------------------------

const _VIDEO_TIME_OPTIONS = [
  { value: '',      label: 'Any time' },
  { value: 'day',   label: 'Past 24 hours' },
  { value: 'week',  label: 'Past week' },
  { value: 'month', label: 'Past month' },
  { value: 'year',  label: 'Past year' },
];

const _VIDEO_DURATION_OPTIONS = [
  { value: '',       label: 'Any duration' },
  { value: 'short',  label: 'Short (< 4 min)' },
  { value: 'medium', label: 'Medium (4–20 min)' },
  { value: 'long',   label: 'Long (> 20 min)' },
];

const _VIDEO_SORT_OPTIONS = [
  { value: '',              label: 'Relevance' },
  { value: 'date',          label: 'Newest first' },
  { value: 'duration_desc', label: 'Longest first' },
  { value: 'duration_asc',  label: 'Shortest first' },
];

function _videoFiltersVisible() {
  if (state.activeCategory === 'videos') return true;
  const results = state.searchResults || [];
  return results.length > 0 && results.every(r => r && r.is_video);
}

function _renderVideoFilterSelect(name, options, current) {
  const opts = options.map(o =>
    `<option value="${escapeHtml(o.value)}"${o.value === current ? ' selected' : ''}>${escapeHtml(o.label)}</option>`
  ).join('');
  return `<select class="browse-vf-select" data-vf="${escapeHtml(name)}">${opts}</select>`;
}

function renderVideoFilters() {
  if (!_videoFiltersVisible()) return '';
  return (
    `<div class="browse-video-filters" role="group" aria-label="Video filters">`
    + `<label class="browse-vf"><span class="browse-vf-icon" aria-hidden="true">🕓</span>`
    + _renderVideoFilterSelect('time_range', _VIDEO_TIME_OPTIONS, state.videoTimeRange)
    + `</label>`
    + `<label class="browse-vf"><span class="browse-vf-icon" aria-hidden="true">⏱</span>`
    + _renderVideoFilterSelect('duration', _VIDEO_DURATION_OPTIONS, state.videoDuration)
    + `</label>`
    + `<label class="browse-vf"><span class="browse-vf-icon" aria-hidden="true">↕</span>`
    + _renderVideoFilterSelect('sort_by', _VIDEO_SORT_OPTIONS, state.videoSortBy)
    + `</label>`
    + `</div>`
  );
}

function wireVideoFilters(container) {
  container.querySelectorAll('.browse-vf-select').forEach(sel => {
    sel.addEventListener('change', () => {
      const name = sel.dataset.vf;
      const value = sel.value;
      const storageKey = {
        time_range: 'augmentum_browse_video_timerange',
        duration:   'augmentum_browse_video_duration',
        sort_by:    'augmentum_browse_video_sort',
      }[name];
      const stateKey = {
        time_range: 'videoTimeRange',
        duration:   'videoDuration',
        sort_by:    'videoSortBy',
      }[name];
      if (!stateKey) return;
      state[stateKey] = value;
      try {
        if (value) localStorage.setItem(storageKey, value);
        else localStorage.removeItem(storageKey);
      } catch { /* quota — ignore */ }
      // Re-issue the search with the new filter applied. No query → no
      // results to filter, just rerender so the dropdown stays in sync.
      if (state.searchQuery) {
        browseSearch(state.searchQuery);
      } else {
        showSearchView();
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Reader typography + a11y preferences
//
// Persisted to localStorage and applied as CSS custom properties on the
// article body. Users open the panel from the article toolbar; settings
// take effect immediately and carry across sessions. Defaults chosen
// for long-form readability, not visual consistency with the rest of
// the app — the article is a distinct mode.
// ---------------------------------------------------------------------------
const _READER_PREFS_KEY = 'augmentum_reader_prefs';
const _READER_DEFAULTS = {
  size: 'm',            // s | m | l | xl
  family: 'serif',      // sans | serif | mono | dyslexic
  height: 'normal',     // tight | normal | airy
  width: 'medium',      // narrow | medium | wide
  justify: false,       // ragged-right default; toggled on for full justify
};

function _pickReaderValue(value, allowed, fallback) {
  return allowed.includes(value) ? value : fallback;
}

function _normalizeReaderPrefs(prefs = {}) {
  return {
    size: _pickReaderValue(prefs.size, ['s', 'm', 'l', 'xl'], _READER_DEFAULTS.size),
    family: _pickReaderValue(prefs.family, ['sans', 'serif', 'mono', 'dyslexic'], _READER_DEFAULTS.family),
    height: _pickReaderValue(prefs.height, ['tight', 'normal', 'airy'], _READER_DEFAULTS.height),
    width: _pickReaderValue(prefs.width, ['narrow', 'medium', 'wide'], _READER_DEFAULTS.width),
    justify: !!prefs.justify,
  };
}

function _readerPrefsFromSettings() {
  const s = getSettings();
  return _normalizeReaderPrefs({
    size: s.browseReaderSize,
    family: s.browseReaderFamily,
    height: s.browseReaderHeight,
    width: s.browseReaderWidth,
    justify: s.browseReaderJustify,
  });
}

function _persistBrowseSettingsPatch(patch) {
  Object.assign(getSettings(), patch);
  try { saveSettings(); } catch {}
}

function _syncBrowsePrefsFromSettings() {
  const s = getSettings();
  state.splitMode = !!s.browseDefaultSplit;
  state.notesHistoryCollapsed = !!s.browseNotesHistoryCollapsed;
  state.readerPrefs = _readerPrefsFromSettings();
  try {
    localStorage.setItem('augmentum_browse_split', state.splitMode ? '1' : '0');
    localStorage.setItem(
      'augmentum_browse_notes_history_collapsed',
      state.notesHistoryCollapsed ? '1' : '0',
    );
    localStorage.setItem(_READER_PREFS_KEY, JSON.stringify(state.readerPrefs));
  } catch { /* private browsing / quota — prefs won't persist across reload */ }
}

document.addEventListener('augmentum:browse-settings-changed', () => {
  _syncBrowsePrefsFromSettings();
  _applySplitMode();
  _applyReaderPrefs();
});

function _loadReaderPrefs() {
  try {
    const raw = localStorage.getItem(_READER_PREFS_KEY);
    if (!raw) return _readerPrefsFromSettings();
    const parsed = JSON.parse(raw);
    return _normalizeReaderPrefs({ ...parsed, ..._readerPrefsFromSettings() });
  } catch { return _readerPrefsFromSettings(); }
}

function _saveReaderPrefs(prefs) {
  const normalized = _normalizeReaderPrefs(prefs);
  state.readerPrefs = normalized;
  try { localStorage.setItem(_READER_PREFS_KEY, JSON.stringify(normalized)); } catch {}
  _persistBrowseSettingsPatch({
    browseReaderSize: normalized.size,
    browseReaderFamily: normalized.family,
    browseReaderHeight: normalized.height,
    browseReaderWidth: normalized.width,
    browseReaderJustify: normalized.justify,
  });
}

/**
 * Apply the reader prefs as data-attributes on the article body plus
 * inline CSS custom properties. Data-attrs gate style rules in
 * browse.css (the `.browse-article-body[data-reader-size="xl"]` form);
 * custom properties avoid a restyling blowout for the rare case where
 * we want ad-hoc per-user overrides in the future.
 */
function _applyReaderPrefs() {
  if (!dom.readerView) return;
  const prefs = state.readerPrefs || _loadReaderPrefs();
  const body = dom.readerView.querySelector('.browse-article-body');
  if (!body) return;
  body.dataset.readerSize = prefs.size;
  body.dataset.readerFamily = prefs.family;
  body.dataset.readerHeight = prefs.height;
  body.dataset.readerWidth = prefs.width;
  body.dataset.readerJustify = prefs.justify ? '1' : '0';
}

function _openTypographyPopover(anchorBtn) {
  // Close any existing popover so repeat clicks toggle
  const existing = document.getElementById('browse-typography-popover');
  if (existing) { existing.remove(); return; }

  const prefs = state.readerPrefs = state.readerPrefs || _loadReaderPrefs();
  const pop = document.createElement('div');
  pop.id = 'browse-typography-popover';
  pop.className = 'browse-typography-popover';
  pop.setAttribute('role', 'dialog');
  pop.setAttribute('aria-label', 'Reading preferences');

  const opt = (value, label, current) =>
    `<button class="browse-typo-opt${value === current ? ' active' : ''}" data-val="${escapeHtml(value)}">${escapeHtml(label)}</button>`;

  pop.innerHTML = `
    <div class="browse-typo-row" data-setting="size">
      <span class="browse-typo-label">Size</span>
      <div class="browse-typo-group">
        ${['s','m','l','xl'].map(v => opt(v, v.toUpperCase(), prefs.size)).join('')}
      </div>
    </div>
    <div class="browse-typo-row" data-setting="family">
      <span class="browse-typo-label">Font</span>
      <div class="browse-typo-group">
        ${opt('sans', 'Sans', prefs.family)}
        ${opt('serif', 'Serif', prefs.family)}
        ${opt('mono', 'Mono', prefs.family)}
        ${opt('dyslexic', 'Dyslexia', prefs.family)}
      </div>
    </div>
    <div class="browse-typo-row" data-setting="height">
      <span class="browse-typo-label">Spacing</span>
      <div class="browse-typo-group">
        ${opt('tight', 'Tight', prefs.height)}
        ${opt('normal', 'Normal', prefs.height)}
        ${opt('airy', 'Airy', prefs.height)}
      </div>
    </div>
    <div class="browse-typo-row" data-setting="width">
      <span class="browse-typo-label">Width</span>
      <div class="browse-typo-group">
        ${opt('narrow', 'Narrow', prefs.width)}
        ${opt('medium', 'Medium', prefs.width)}
        ${opt('wide', 'Wide', prefs.width)}
      </div>
    </div>
    <div class="browse-typo-row" data-setting="justify">
      <span class="browse-typo-label">Justify</span>
      <label class="browse-typo-switch">
        <input type="checkbox" ${prefs.justify ? 'checked' : ''}>
        <span></span>
      </label>
    </div>
    <div class="browse-typo-row browse-typo-reset">
      <button class="browse-typo-reset-btn" data-reset>Reset to defaults</button>
    </div>
  `;

  // Position below the button that opened us
  const r = anchorBtn.getBoundingClientRect();
  pop.style.position = 'fixed';
  pop.style.top = `${r.bottom + 6}px`;
  pop.style.right = `${window.innerWidth - r.right}px`;

  document.body.appendChild(pop);

  // Wire controls
  pop.querySelectorAll('.browse-typo-row').forEach(row => {
    const key = row.dataset.setting;
    if (key === 'justify') {
      const cb = row.querySelector('input[type="checkbox"]');
      cb?.addEventListener('change', () => {
        prefs.justify = cb.checked;
        _saveReaderPrefs(prefs);
        _applyReaderPrefs();
      });
    } else {
      row.querySelectorAll('.browse-typo-opt').forEach(btn => {
        btn.addEventListener('click', () => {
          prefs[key] = btn.dataset.val;
          _saveReaderPrefs(prefs);
          _applyReaderPrefs();
          row.querySelectorAll('.browse-typo-opt').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
        });
      });
    }
  });
  pop.querySelector('[data-reset]')?.addEventListener('click', () => {
    state.readerPrefs = { ..._READER_DEFAULTS };
    _saveReaderPrefs(state.readerPrefs);
    _applyReaderPrefs();
    pop.remove();
  });

  // Close on outside click
  setTimeout(() => {
    const offclick = (e) => {
      if (pop.contains(e.target) || e.target === anchorBtn) return;
      pop.remove();
      document.removeEventListener('click', offclick);
    };
    document.addEventListener('click', offclick);
  }, 0);
}

function _openTocPopover(anchorBtn) {
  const existing = document.getElementById('browse-toc-popover');
  if (existing) { existing.remove(); return; }

  const entries = state.articleToc || [];
  if (!entries.length) return;

  const pop = document.createElement('div');
  pop.id = 'browse-toc-popover';
  pop.className = 'browse-toc-popover';
  pop.setAttribute('role', 'dialog');
  pop.setAttribute('aria-label', 'Table of contents');

  const items = entries.map(({ level, id, text }) => {
    const cls = `browse-toc-item browse-toc-${level.toLowerCase()}`;
    return `<a class="${cls}" href="#${escapeHtml(id)}" data-toc-target="${escapeHtml(id)}">${escapeHtml(text)}</a>`;
  }).join('');
  pop.innerHTML = `
    <div class="browse-toc-header">On this page</div>
    <div class="browse-toc-list">${items}</div>
  `;

  const r = anchorBtn.getBoundingClientRect();
  pop.style.position = 'fixed';
  pop.style.top = `${r.bottom + 6}px`;
  pop.style.right = `${window.innerWidth - r.right}px`;

  document.body.appendChild(pop);

  pop.addEventListener('click', (e) => {
    const link = e.target.closest('[data-toc-target]');
    if (!link) return;
    e.preventDefault();
    const articleBody = dom.readerView?.querySelector('.browse-article-body');
    const target = articleBody?.querySelector(`#${CSS.escape(link.dataset.tocTarget)}`);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      target.classList.add('browse-toc-flash');
      setTimeout(() => target.classList.remove('browse-toc-flash'), 1600);
    }
    pop.remove();
  });

  setTimeout(() => {
    const offclick = (e) => {
      if (pop.contains(e.target) || e.target === anchorBtn) return;
      pop.remove();
      document.removeEventListener('click', offclick);
    };
    document.addEventListener('click', offclick);
  }, 0);
}

// ---------------------------------------------------------------------------
// Read aloud — streams the article body to TTS, sentence by sentence.
// Reuses the chat TTS pipeline lazily so we don't pull it in for users
// who never hit the button. Idempotent play/pause: second click stops.
// ---------------------------------------------------------------------------

/**
 * Selectors whose contents should NEVER be read aloud. Covers:
 *   - Code / pre blocks (unreadable when spoken; would be noise).
 *   - Nav chrome from our intercepts (chip rows, meta strips, tag
 *     clouds, TOC nav, toolbars, badges, file lists).
 *   - Comment/post metadata from Reddit / HN / Stack Exchange (author
 *     + score + age lines repeat hundreds of times in megathreads
 *     and are just noise to a listener).
 *   - The article's own toolbar (Summarize / Key Points / etc. button
 *     labels) if anything ever accidentally pulls it into scope.
 * Listed as CSS selectors so adding a new intercept's metadata row is
 * a one-line change.
 */
const _READ_ALOUD_SKIP_SELECTORS = [
  'pre', 'code', 'kbd', 'samp', 'var',
  'nav', 'aside',
  '.browse-article-toc', '.browse-article-toolbar', '.browse-article-meta-source',
  '.browse-read-aloud-label', '.browse-bookmark-label',
  // Intercept metadata chips + navigation rows
  '.hf-chips', '.hf-tags', '.hf-blob-meta', '.hf-blob-path-sub',
  '.hf-files', '.hf-gated-banner', '.hf-kind-chip',
  '.gh-meta', '.gh-topics', '.gh-blob-meta', '.gh-blob-path-sub',
  '.gh-readme-raw',
  '.reddit-post-header', '.reddit-sub-line', '.reddit-post-meta',
  '.reddit-comment-meta', '.reddit-listing-meta',
  '.hn-post-header', '.hn-sub-line', '.hn-post-meta', '.hn-comment-meta',
  '.hn-more-replies',
  '.se-meta', '.se-tags', '.se-answer-meta',
  '.gist-file-header', '.gist-meta',
  // Image alt that doesn't help when read
  'figure figcaption',
  // Reference numbers at the very end of Wikipedia-style articles
  '#References', '#external-links', '.references',
].join(',');

/**
 * Tag names whose text content is the prose we actually want spoken.
 * Everything else passes through unless explicitly skipped — this
 * handles both our intercept output and trafilatura-extracted pages.
 */
const _READ_ALOUD_PROSE_TAGS = new Set([
  'P', 'LI', 'BLOCKQUOTE', 'DT', 'DD',
  'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
]);

/**
 * Walk the article body and extract just the prose. Strips code blocks,
 * metadata chips, TOC entries, and other chrome. Inserts paragraph
 * breaks between blocks so the TTS engine gets a natural pause.
 *
 * Returns an empty string if nothing readable survives — caller falls
 * back to a "no readable prose" toast instead of feeding silence to
 * the TTS engine.
 */
function _extractReadableText(container) {
  if (!container) return '';
  // Work on a clone so we don't mutate the live DOM.
  const clone = container.cloneNode(true);
  // Prune everything we don't want spoken.
  clone.querySelectorAll(_READ_ALOUD_SKIP_SELECTORS).forEach(el => el.remove());

  const parts = [];
  const seen = new Set();
  const walker = document.createTreeWalker(clone, NodeFilter.SHOW_ELEMENT, {
    acceptNode(node) {
      if (!_READ_ALOUD_PROSE_TAGS.has(node.tagName)) return NodeFilter.FILTER_SKIP;
      // Skip nested prose (a <li> that contains a <p> — we'd read the
      // outer once and then the inner again). Only collect the outer.
      for (const ancestor of _parentChain(node, clone)) {
        if (_READ_ALOUD_PROSE_TAGS.has(ancestor.tagName)) return NodeFilter.FILTER_SKIP;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  let node;
  while ((node = walker.nextNode())) {
    const text = (node.textContent || '').replace(/\s+/g, ' ').trim();
    if (!text) continue;
    if (seen.has(text)) continue;  // dedupe verbatim repeats (menus, chips)
    seen.add(text);
    // Add a sentence-ending period if the block doesn't have one, so
    // heading lines don't run into the first paragraph below them.
    const punct = /[.!?:]$/.test(text) ? '' : '.';
    parts.push(text + punct);
  }
  return parts.join('\n\n');
}

function _parentChain(node, stopAt) {
  const out = [];
  let cur = node.parentElement;
  while (cur && cur !== stopAt) { out.push(cur); cur = cur.parentElement; }
  return out;
}

async function _toggleReadAloud(btn) {
  const { readAloud } = await import('./read-aloud.js');
  const articleBody = dom.readerView?.querySelector('.browse-article-body');
  if (!articleBody) return;
  const text = _extractReadableText(articleBody);
  if (!text) {
    showToast('No readable prose found in this article.', 'info', 2000);
    return;
  }
  const title = dom.readerView?.querySelector('.browse-article-title')?.textContent?.trim() || '';
  const sourceUrl = state.currentUrl || '';
  return readAloud(text, btn, { title, sourceUrl });
}

// ---------------------------------------------------------------------------
// Bookmarks — localStorage-backed for v1 (Promise-returning so we can
// swap to a server-side store later without touching callers).
// ---------------------------------------------------------------------------
const _BOOKMARKS_KEY = 'augmentum_browse_bookmarks';

const bookmarks = {
  async list() {
    try {
      const raw = localStorage.getItem(_BOOKMARKS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  },
  async has(url) {
    const items = await this.list();
    return items.some(b => b.url === url);
  },
  async add(entry) {
    const items = await this.list();
    // Dedup by URL; promote to front if already present.
    const filtered = items.filter(b => b.url !== entry.url);
    filtered.unshift({ ...entry, id: 'bm_' + Date.now().toString(36), savedAt: Date.now() });
    try { localStorage.setItem(_BOOKMARKS_KEY, JSON.stringify(filtered)); } catch {}
    document.dispatchEvent(new CustomEvent('augmentum:bookmarks-changed'));
  },
  async remove(url) {
    const items = await this.list();
    const filtered = items.filter(b => b.url !== url);
    try { localStorage.setItem(_BOOKMARKS_KEY, JSON.stringify(filtered)); } catch {}
    document.dispatchEvent(new CustomEvent('augmentum:bookmarks-changed'));
  },
};

async function _toggleBookmark(btn) {
  const url = state.currentUrl;
  if (!url) return;
  const saved = await bookmarks.has(url);
  if (saved) {
    await bookmarks.remove(url);
    btn.classList.remove('saved');
    const lbl = btn.querySelector('.browse-bookmark-label');
    if (lbl) lbl.textContent = 'Bookmark';
    showToast('Bookmark removed', 'info', 1500);
  } else {
    // Pull a quick snippet from the article body so the bookmark card
    // has a preview without us reparsing the page later.
    const body = dom.readerView?.querySelector('.browse-article-body');
    const snippet = body
      ? (body.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 200)
      : '';
    const titleEl = dom.readerView?.querySelector('.browse-article-title');
    const title = titleEl?.textContent?.trim() || url;
    // Favicon — resolve host and proxy through our image endpoint.
    let favicon = '';
    try {
      const host = new URL(url).hostname;
      favicon = `/api/browse/image?url=${encodeURIComponent(
        `https://www.google.com/s2/favicons?domain=${host}&sz=32`
      )}`;
    } catch { /* ignore */ }
    await bookmarks.add({ url, title, favicon, snippet });
    btn.classList.add('saved');
    const lbl = btn.querySelector('.browse-bookmark-label');
    if (lbl) lbl.textContent = 'Saved';
    showToast('Bookmark saved', 'success', 1500);
  }
}

// App menu: the companion can press these via app.act ("save this
// page", "read it to me"). Outcome actions on the OPEN article only —
// the when guard keys off state.currentUrl so they go dark when no
// page is up. Liveness refresh fires from the page open/close
// presence call sites.
import('./command-palette.js').then(({ registerCommand }) => {
  registerCommand({
    id: 'browse.bookmark-current',
    label: 'Bookmark this page',
    group: 'Browse',
    keywords: 'bookmark save keep this page article for later',
    when: () => !!state.currentUrl,
    agent: {
      description: 'Bookmark the article currently open in Browse',
      speak: 'Bookmarked it for you.',
    },
    run: async () => {
      const url = state.currentUrl;
      if (!url) return;
      // The toolbar button is a TOGGLE; her intent is "save it" —
      // keep the agent path idempotent so a re-ask never unsaves.
      if (await bookmarks.has(url)) {
        showToast('Already bookmarked', 'info', 1500);
        return;
      }
      const btn = dom.readerView?.querySelector('[data-article-action="bookmark"]');
      if (btn) await _toggleBookmark(btn);
    },
  });
  registerCommand({
    id: 'browse.read-aloud-current',
    label: 'Read this article aloud',
    group: 'Browse',
    keywords: 'read aloud listen narrate this page article to me',
    when: () => !!state.currentUrl
      && !!dom.readerView?.querySelector('.browse-article-body'),
    agent: {
      description: 'Read the article currently open in Browse out loud',
      speak: 'Starting the read-aloud.',
    },
    run: () => {
      const btn = dom.readerView?.querySelector('[data-article-action="read-aloud"]');
      if (btn) _toggleReadAloud(btn);
    },
  });
}).catch(() => {});

// Refresh the bookmark button's filled state whenever an article
// finishes rendering. Kept in a dedicated helper so renderArticle
// doesn't need to await the bookmarks store.
function _refreshBookmarkButtonState() {
  const btn = dom.readerView?.querySelector('[data-article-action="bookmark"]');
  if (!btn || !state.currentUrl) return;
  bookmarks.has(state.currentUrl).then(saved => {
    btn.classList.toggle('saved', saved);
    const lbl = btn.querySelector('.browse-bookmark-label');
    if (lbl) lbl.textContent = saved ? 'Saved' : 'Bookmark';
  });
}

// Keyboard shortcut Ctrl+D / Cmd+D → toggle bookmark on current article.
// Prevent the browser's built-in bookmark dialog only when our reader
// is the active surface — respect the browser default elsewhere.
document.addEventListener('keydown', (e) => {
  if (!state.panelOpen || !state.currentUrl) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') {
    e.preventDefault();
    const btn = dom.readerView?.querySelector('[data-article-action="bookmark"]');
    if (btn) _toggleBookmark(btn);
  }
});

// ---------------------------------------------------------------------------
// Provider filter chips — one-click `site:` scoping for the sites we
// render well. Selection is exclusive (one at a time). Re-clicking the
// active chip clears the filter. If the user types `site:X` directly
// into the query we respect that and highlight a matching chip.
//
// Ordering: strong intercepts first (render cleanly in the reader),
// then good-but-trafilatura sites, then partial-support sites. Each
// chip gets its favicon via /api/browse/image so the CSP is happy.
// ---------------------------------------------------------------------------
const _PROVIDER_CHIPS = [
  { id: 'wikipedia',    label: 'Wikipedia', site: 'wikipedia.org' },
  { id: 'youtube',      label: 'YouTube',   site: 'youtube.com' },
  { id: 'github',       label: 'GitHub',    site: 'github.com' },
  { id: 'huggingface',  label: 'HF',        site: 'huggingface.co' },
  { id: 'reddit',       label: 'Reddit',    site: 'reddit.com' },
  { id: 'stackoverflow', label: 'StackOverflow', site: 'stackoverflow.com' },
  // Stack Exchange network (math, physics, biology, ai, unix, apple,
  // worldbuilding, academia, …). Same rich Q&A renderer as StackOverflow
  // — `_try_stackexchange_api` classifies every `*.stackexchange.com`
  // subdomain automatically.
  { id: 'stackexchange', label: 'Stack Exchange', site: 'stackexchange.com' },
  { id: 'hn',           label: 'HN',        site: 'news.ycombinator.com' },
  { id: 'arxiv',        label: 'arXiv',     site: 'arxiv.org' },
  { id: 'mdn',          label: 'MDN',       site: 'developer.mozilla.org' },
  { id: 'tiktok',       label: 'TikTok',    site: 'tiktok.com' },
];

function renderProviderChips() {
  return _PROVIDER_CHIPS.map(p => {
    const active = state.activeProvider === p.id;
    const favicon = `/api/browse/image?url=${encodeURIComponent(
      `https://www.google.com/s2/favicons?domain=${p.site}&sz=32`
    )}`;
    return (
      `<button class="browse-provider-chip${active ? ' active' : ''}" `
      + `data-provider="${escapeHtml(p.id)}" `
      + `data-site="${escapeHtml(p.site)}" `
      + `title="Scope search to ${escapeHtml(p.site)}">`
      + `<img class="browse-provider-favicon" src="${favicon}" alt="" loading="lazy" decoding="async" onerror="this.remove()">`
      + `<span class="browse-provider-label">${escapeHtml(p.label)}</span>`
      + `</button>`
    );
  }).join('');
}

function wireProviderChips(container) {
  container.querySelectorAll('.browse-provider-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.provider;
      // Toggle: re-clicking the active chip clears the filter.
      state.activeProvider = (state.activeProvider === id) ? null : id;
      // Re-run with the stripped-and-refiltered query. _injectProviderFilter
      // pulls any existing site: out first so repeated clicks don't stack.
      const rawQuery = (state.searchQuery || '').replace(/\bsite:\S+\s*/gi, '').trim();
      if (rawQuery) {
        browseSearch(rawQuery);
      } else {
        // No query yet — re-render the chips so the visual state updates
        // and wait for the user to type.
        showSearchView();
      }
    });
  });
}

/**
 * Prefix a `site:DOMAIN` operator to the query based on state.activeProvider
 * (or a site: embedded in the user's own query). Strips any existing
 * site: tokens before prepending so the provider chip always wins.
 * Returns the effective query the backend should see.
 */
function _injectProviderFilter(query) {
  query = (query || '').trim();
  if (!query) return query;

  // If the user typed site: themselves, let that take precedence and
  // auto-match a chip.
  const userSiteMatch = query.match(/\bsite:(\S+)/i);
  if (userSiteMatch) {
    const matchedSite = userSiteMatch[1].toLowerCase();
    const chip = _PROVIDER_CHIPS.find(p => matchedSite.endsWith(p.site));
    if (chip) state.activeProvider = chip.id;
    return query;  // preserve exactly what they typed
  }

  if (!state.activeProvider) return query;
  const chip = _PROVIDER_CHIPS.find(p => p.id === state.activeProvider);
  if (!chip) return query;
  return `site:${chip.site} ${query}`;
}

function highlightQuery(text, query) {
  if (!text || !query) return escapeHtml(text);
  const escaped = escapeHtml(text);
  const words = query.split(/\s+/).filter(w => w.length > 2).map(w =>
    w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  );
  if (!words.length) return escaped;
  const re = new RegExp(`(${words.join('|')})`, 'gi');
  return escaped.replace(re, '<mark>$1</mark>');
}

/**
 * Render a SearXNG date string as a compact relative label. Engines
 * return either ISO 8601 ("2026-05-20T12:00:00Z"), a bare date, or a
 * pre-relativised phrase ("3 days ago"). The first two get parsed and
 * fed to _relativeDate; pre-relativised strings pass through. Empty /
 * unparseable input returns "" so callers can skip the row.
 */
function _humanizeResultDate(raw) {
  if (!raw) return '';
  const text = String(raw).trim();
  if (!text) return '';
  // Already-relative — common from YouTube engine etc.
  if (/\b(ago|today|yesterday|just now)\b/i.test(text)) return text;
  const parsed = new Date(text);
  if (!parsed.getTime()) return text;  // unparseable but non-empty — show as-is
  return _relativeDate(parsed);
}

/**
 * Format a raw view-count int into the K/M/B-suffixed shorthand the
 * YouTube card uses. Returns "" for missing/zero so the UI can skip
 * the row entirely without showing "0 views" noise.
 */
function _formatViews(raw) {
  if (raw === null || raw === undefined || raw === '') return '';
  const n = typeof raw === 'number' ? raw : parseInt(String(raw).replace(/[^\d]/g, ''), 10);
  if (!Number.isFinite(n) || n <= 0) return '';
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1).replace(/\.0$/, '')}B views`;
  if (n >= 1_000_000)     return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, '')}M views`;
  if (n >= 1_000)         return `${(n / 1_000).toFixed(1).replace(/\.0$/, '')}K views`;
  return `${n} views`;
}

function renderSearchResults() {
  const container = dom.searchView;
  if (!container) return;

  if (!state.searchResults || !state.searchResults.length) {
    container.innerHTML = `
      ${renderSearchControls()}
      <div class="browse-no-results">
        <p>No results found for "${escapeHtml(state.searchQuery)}"</p>
        <p class="browse-no-results-hint">Try different search terms or another category</p>
      </div>
    `;
    wireSearchControls(container);
    return;
  }

  const isImageCat = state.activeCategory === 'images';
  const isVideoCat = state.activeCategory === 'videos';
  const q = state.searchQuery;
  // Whole-feed video mode: category == videos, OR every single result is a
  // known video host. We flip the list container to a grid in these cases
  // so cards align as a proper video wall instead of stretching to full
  // width (which would make each thumbnail oversized and the grid sparse).
  const allVideoResults = !isImageCat && !isVideoCat
    && state.searchResults.length > 0
    && state.searchResults.every(r => r.is_video);
  const useVideoGrid = isVideoCat || allVideoResults;

  const cards = state.searchResults.map(r => {
    let hostname = '';
    try { hostname = new URL(r.url).hostname.replace(/^www\./, ''); } catch {}
    const favicon = `/api/browse/image?url=${encodeURIComponent(`https://www.google.com/s2/favicons?domain=${hostname}&sz=16`)}`;
    const hasThumbnail = r.thumbnail && !isImageCat;
    const thumbProxy = r.thumbnail ? `/api/browse/image?url=${encodeURIComponent(r.thumbnail)}` : '';
    const isVideo = !!r.is_video || isVideoCat;

    // Image grid card
    if (isImageCat && r.thumbnail) {
      return `
        <div class="browse-result-image-card" data-url="${escapeHtml(r.url)}">
          <img class="browse-result-image-thumb" src="${escapeHtml(thumbProxy)}" alt="${escapeHtml(r.title)}" loading="lazy">
          <div class="browse-result-image-info">
            <div class="browse-result-image-title">${escapeHtml(r.title)}</div>
            <div class="browse-result-image-domain">${escapeHtml(hostname)}</div>
            ${r.img_format ? `<div class="browse-result-image-size">${escapeHtml(r.img_format)}</div>` : ''}
          </div>
        </div>
      `;
    }

    // Video card — thumbnail-first, 16:9, duration badge, channel.
    // The reader already handles the embed on click, so this is purely a
    // visual affordance: signal "this is a video" at the search stage.
    if (isVideo && r.thumbnail) {
      const duration = (r.duration || '').trim();
      const channel = (r.author || '').trim();
      const relDate = _humanizeResultDate(r.published_date);
      const viewsLabel = _formatViews(r.views);
      return `
        <div class="browse-result-video-card" data-url="${escapeHtml(r.url)}">
          <div class="browse-result-video-thumb-wrap">
            <img class="browse-result-video-thumb" src="${escapeHtml(thumbProxy)}" alt="${escapeHtml(r.title)}" loading="lazy" onerror="this.style.display='none'">
            <span class="browse-result-video-play" aria-hidden="true">▶</span>
            ${duration ? `<span class="browse-result-video-duration">${escapeHtml(duration)}</span>` : ''}
          </div>
          <div class="browse-result-video-info">
            <div class="browse-result-video-title">${highlightQuery(r.title, q)}</div>
            <div class="browse-result-video-meta">
              ${r.reputation ? `<span class="browse-reputation-dot" data-rep="${escapeHtml(r.reputation)}" title="${escapeHtml(r.reputation)}"></span>` : ''}
              <img class="browse-favicon" src="${escapeHtml(favicon)}" alt="" loading="lazy" onerror="this.style.display='none'">
              <span>${escapeHtml(channel || hostname)}</span>
              ${viewsLabel ? `<span class="browse-result-views">${escapeHtml(viewsLabel)}</span>` : ''}
              ${relDate ? `<span class="browse-result-date">${escapeHtml(relDate)}</span>` : ''}
            </div>
          </div>
        </div>
      `;
    }

    // Standard result card
    const stdRelDate = _humanizeResultDate(r.published_date);
    return `
      <div class="browse-result-card${hasThumbnail ? ' has-thumb' : ''}" data-url="${escapeHtml(r.url)}">
        <div class="browse-result-body">
          <div class="browse-result-source">
            ${r.reputation ? `<span class="browse-reputation-dot" data-rep="${escapeHtml(r.reputation)}" title="${escapeHtml(r.reputation)}"></span>` : ''}
            <img class="browse-favicon" src="${escapeHtml(favicon)}" alt="" loading="lazy" onerror="this.style.display='none'">
            <span>${escapeHtml(hostname)}</span>
            ${stdRelDate ? `<span class="browse-result-date">${escapeHtml(stdRelDate)}</span>` : ''}
          </div>
          <div class="browse-result-title">${highlightQuery(r.title, q)}</div>
          ${r.snippet ? `<div class="browse-result-snippet">${highlightQuery(r.snippet, q)}</div>` : ''}
        </div>
        ${hasThumbnail ? `<img class="browse-result-thumb" src="${escapeHtml(thumbProxy)}" alt="" loading="lazy" onerror="this.style.display='none'">` : ''}
      </div>
    `;
  }).join('');

  const loadMoreBtn = state.hasMoreResults
    ? `<button class="browse-load-more-btn" id="browse-load-more">More results</button>`
    : '';

  let listClass = 'browse-results-list';
  if (isImageCat) listClass = 'browse-results-grid';
  else if (useVideoGrid) listClass = 'browse-results-video-grid';

  container.innerHTML = `
    ${renderSearchControls()}
    <div class="browse-results-count">${state.searchResults.length} results for "${escapeHtml(state.searchQuery)}"</div>
    <div class="${listClass}">${cards}</div>
    ${loadMoreBtn}
  `;

  wireSearchControls(container);

  // Wire result clicks
  container.querySelectorAll('.browse-result-card, .browse-result-image-card, .browse-result-video-card').forEach(card => {
    card.addEventListener('click', () => {
      const url = card.dataset.url;
      if (url) browseFetch(url);
    });
  });

  // Per-card Cast overlay — video + image cards only. Lets the user
  // send a result straight to a TV without first navigating into it
  // here in browse. Cast button stops propagation so the surrounding
  // card click (which navigates) doesn't also fire.
  container.querySelectorAll('.browse-result-video-card').forEach((card) => {
    const url = card.dataset.url;
    if (!url) return;
    card.classList.add('cast-btn-host');
    const wrap = card.querySelector('.browse-result-video-thumb-wrap');
    if (!wrap) return;
    const title = card.querySelector('.browse-result-video-title')?.textContent?.trim() || 'Video';
    // Browse-search video results are virtually always YouTube / Vimeo /
    // similar webpages (not direct .mp4 URLs). Native <video src=…> on
    // the TV can't play those — display.web_show@1 loads the URL in an
    // iframe via the cast-receiver's html.generic surface, same path
    // M8 (chat YT cards) uses successfully.
    const castBtn = mountCastButton({
      capability: 'display.web_show@1',
      size: 'sm',
      className: 'cast-btn-on-image cast-btn-hover-reveal browse-result-cast',
      title: 'Cast to TV',
      getContent: () => ({
        contentUrl: url,
        title,
        contentKey: url,
        metadata: { source: 'browse-search-video' },
      }),
    });
    wrap.appendChild(castBtn);

    // Add-to-playlist — YouTube only, since the playlist controller plays
    // items via the YouTube iframe API or HTML5 media element. Other hosts
    // (Vimeo, etc.) would need new dispatchers; deferred.
    const ytMatch = url.match(/(?:v=|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})/);
    if (ytMatch) {
      const videoId = ytMatch[1];
      const channel = card.querySelector('.browse-result-video-meta span:not(.browse-result-date)')?.textContent?.trim() || '';
      const playlistBtn = document.createElement('button');
      playlistBtn.type = 'button';
      playlistBtn.className = 'browse-result-playlist cast-btn-on-image cast-btn-hover-reveal';
      playlistBtn.title = 'Add to Grove playlist';
      playlistBtn.setAttribute('aria-label', 'Add to playlist');
      playlistBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="13" y2="18"/><line x1="18" y1="15" x2="18" y2="21"/><line x1="15" y1="18" x2="21" y2="18"/></svg>`;
      playlistBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        window.dispatchEvent(new CustomEvent('playlist:add-item', {
          detail: {
            type: 'youtube',
            videoId,
            title,
            channel,
            thumbnail: `https://i.ytimg.com/vi/${videoId}/mqdefault.jpg`,
          },
        }));
      });
      wrap.appendChild(playlistBtn);
    }
  });

  container.querySelectorAll('.browse-result-image-card').forEach((card) => {
    const url = card.dataset.url;
    const img = card.querySelector('.browse-result-image-thumb');
    if (!url || !img) return;
    card.classList.add('cast-btn-host');
    const title = card.querySelector('.browse-result-image-title')?.textContent?.trim() || 'Image';
    // For image results the thumbnail URL is the actual asset; the
    // card's data-url is the *source page* about the image. Cast the
    // asset so the TV gets pixels, not a webpage.
    const castBtn = mountCastButton({
      capability: 'display.image_show@1',
      size: 'sm',
      className: 'cast-btn-on-image cast-btn-hover-reveal browse-result-cast',
      title: 'Cast to TV',
      getContent: () => ({
        contentUrl: img.src,
        title,
        contentKey: img.src,
        metadata: { source: 'browse-search-image', sourcePage: url },
      }),
    });
    card.appendChild(castBtn);
  });

  // Wire load more — always restore button state on completion so a
  // network failure doesn't leave it frozen at "Loading...". browseSearch
  // doesn't throw on rejection (it catches internally), but the page may
  // also navigate away mid-fetch, so guard with try/finally regardless.
  const moreBtn = container.querySelector('#browse-load-more');
  if (moreBtn) {
    const originalLabel = moreBtn.textContent;
    moreBtn.addEventListener('click', async () => {
      moreBtn.textContent = 'Loading...';
      moreBtn.disabled = true;
      try {
        await browseSearch(state.searchQuery, null, true);
      } catch (err) {
        console.warn('load_more_failed', err);
      } finally {
        // If the search succeeded the results panel will be re-rendered
        // and this button will be replaced anyway; if it failed, restore
        // the label so the user can retry.
        if (moreBtn.isConnected) {
          moreBtn.textContent = originalLabel || 'Load more';
          moreBtn.disabled = false;
        }
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Fetch / Reader
// ---------------------------------------------------------------------------
async function browseFetch(url) {
  if (!url) return;

  if (state.historyIdx >= 0 && state.history[state.historyIdx]) {
    state.history[state.historyIdx].scrollPos = dom.contentArea?.scrollTop || 0;
  }

  // Knowledge-pack ZIM articles bypass the web extraction pipeline — the
  // content is already clean encyclopedic HTML and lives behind our own
  // /api/zim route. We just iframe it. Empty-state preserved: this branch
  // only fires when something passes a zim: URL, which only happens when
  // a pack is installed AND the user activated a citation/source link.
  if (url.startsWith('zim:')) {
    return _renderZimArticle(url);
  }

  // Clear ZIM-mode layout classes when navigating away from a pack article
  // back to a regular web page. Without this, the reader view stays in
  // full-bleed flex mode and the extracted-article typography (centered
  // 760px column) collapses into the wrong shape.
  dom.contentArea?.classList.remove('is-zim');
  dom.readerView?.classList.remove('is-zim');

  state.currentUrl = url;
  showReaderView();
  showArticleLoading();

  if (dom.searchInput) dom.searchInput.value = url;

  try {
    const resp = await fetch(`/api/browse/fetch?url=${encodeURIComponent(url)}`);
    const data = await resp.json();

    if (!resp.ok) {
      showArticleError(data.error || 'Failed to load page');
      return;
    }

    // Backend short-circuited a known-hostile domain (Amazon, Facebook,
    // LinkedIn, Google Maps, etc.). Render the friendly open-in-browser
    // card instead of blanking out. Still update history + tab so the
    // back/forward stack works and the tab title shows the hostname.
    if (data.unsupported) {
      state.currentUrl = data.url || url;
      const entry = {
        url: data.url || url,
        title: data.title || data.hostname || url,
        scrollPos: 0,
      };
      if (state.historyIdx < state.history.length - 1) {
        state.history = state.history.slice(0, state.historyIdx + 1);
      }
      state.history.push(entry);
      state.historyIdx = state.history.length - 1;
      const activeTab = _activePageTab();
      if (activeTab) {
        activeTab.url = entry.url;
        activeTab.title = entry.title;
        activeTab.favicon = data.favicon_url || _faviconForUrl(entry.url);
        _renderPageTabStrip();
        _savePageTabsToStorage();
      }
      showReaderView();
      showUnsupportedSite(data);
      updateNavButtons();
      return;
    }

    state.currentContent = data.text;
    state.currentHtml = data.html;
    state.currentVideos = data.videos || [];
    state.currentPageType = data.page_type || 'article';

    // Let subscribed chat surfaces (via SurfaceFlows) know a page was
    // extracted so they can offer it as ambient context. Payload shape
    // matches surface-flows.js's augmentum:browse-extracted handler.
    if (data.text) {
      document.dispatchEvent(new CustomEvent('augmentum:browse-extracted', {
        detail: { url: data.url || url, title: data.title || url, text: data.text },
      }));
    }

    _emitSignal('page_visit', {
      source_url: url,
      source_title: data.title || '',
      content_type: 'article',
      raw_content: data.html || '',
      is_html: true,
    });

    const entry = { url: data.url || url, title: data.title || url, scrollPos: 0 };
    if (state.historyIdx < state.history.length - 1) {
      state.history = state.history.slice(0, state.historyIdx + 1);
    }
    state.history.push(entry);
    state.historyIdx = state.history.length - 1;
    _syncBrowseSurface(entry.url, entry.title);

    // Companion presence: this page is now what "this page/article" means.
    // browse_history only records discovery-feed opens, so in-panel
    // navigation (article links, searches) was invisible to her. The
    // excerpt is what lets "tell me about this article" be a real
    // answer instead of a re-open — title alone isn't discussable.
    import('./architect-observer.js')
      .then(m => m.reportAttention('surface.browse.page_opened', {
        url: entry.url,
        title: entry.title,
        excerpt: String(data.text || '').slice(0, 1500),
      }))
      .catch(() => {});
    // "Read this page" handoff: the observer only carries a 1500-char
    // excerpt; this lets the user hand her the FULL article text on
    // demand. The body is captured here (the page just rendered).
    const _fullPageText = String(data.text || '');
    if (_fullPageText.trim()) {
      import('./companion-context.js')
        .then(m => m.setCompanionLoadable('page', entry.title || entry.url, () => ({
          label: entry.title || entry.url, content: _fullPageText, ref: entry.url,
        })))
        .catch(() => {});
    }
    // Page-scoped app-menu actions (bookmark / read-aloud) just went live.
    import('./command-palette.js')
      .then(m => m.refreshAgentCatalog())
      .catch(() => {});

    // Update the active page tab's display metadata so the tab strip
    // shows the real page title + favicon instead of "New Tab".
    const activeTab = _activePageTab();
    if (activeTab) {
      activeTab.url = entry.url;
      activeTab.title = entry.title;
      activeTab.favicon = data.favicon_url || _faviconForUrl(entry.url);
      _renderPageTabStrip();
      _savePageTabsToStorage();
    }

    renderArticle(data);
    if (dom.readerAiBlocks) dom.readerAiBlocks.innerHTML = '';
    updateNavButtons();
  } catch (err) {
    showArticleError(`Failed to load: ${err.message}`);
  }
}

/** Try to upgrade a ZIM iframe to the isolated origin.
 *
 * Fire-and-forget. Calls /api/content/preview-token for the pack; on
 * success rewrites the iframe src to ``<isolated>/api/knowledge/zim/...
 * ?_pvt=<token>`` so the iframe is served from a different origin and
 * cannot reach Augmentum's main /api/* with the user's session cookie.
 *
 * Failure modes (all silently keep the same-origin fallback):
 *   - 501 from mint endpoint = content_iframe_isolation_enabled is off
 *   - 503 = token store not initialised yet
 *   - Network error = transient, try again on next render
 *   - Mint denied = pack ownership / existence check failed
 *
 * Each call is best-effort and idempotent on the iframe element (we
 * only replace src if we actually got an isolated URL back).
 */
async function _tryUpgradeZimIframeToIsolated(iframe, packId, sameOriginUrl) {
  if (!iframe || !packId || !sameOriginUrl) return;
  let data;
  try {
    const resp = await fetch('/api/content/preview-token', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: 'knowledge_pack', id: packId }),
    });
    if (!resp.ok) {
      // 501 (isolation off) is the common case — log at debug-equivalent
      // level so production noise stays low. Other non-OK statuses log
      // their status for triage.
      if (resp.status !== 501) {
        console.log('[browse.zim] isolation upgrade declined', {
          status: resp.status, pack: packId,
        });
      }
      return;
    }
    data = await resp.json();
  } catch (err) {
    console.log('[browse.zim] isolation upgrade network error → same-origin fallback', {
      error: err?.message || String(err),
    });
    return;
  }
  if (!data || !data.token || !data.isolated_origin) return;
  // Compose the isolated URL. sameOriginUrl already carries the path +
  // query (theme=, reader=); append _pvt= preserving them.
  const sep = sameOriginUrl.includes('?') ? '&' : '?';
  const target = `${data.isolated_origin}${sameOriginUrl}${sep}_pvt=${encodeURIComponent(data.token)}`;
  iframe.dataset.zimIsolated = '1';
  iframe.src = target;
}


/** Render a knowledge-pack ZIM article via sandboxed iframe.
 *
 * URL shape: ``zim:<pack_id>/<entry_path>`` (e.g. ``zim:mdwiki_en_all_2025-11/A/Diabetes``).
 * The iframe loads ``/api/zim/<pack_id>/<entry_path>`` which serves the
 * cleaned HTML with internal links rewritten back at the same route, so
 * navigation within the ZIM stays scoped to this iframe.
 *
 * Sandbox: ``allow-same-origin`` only — no JS, no top-level navigation,
 * no popups. The ZIM HTML is upstream content (Wikipedia, MDWiki, ...);
 * we trust the source but not enough to let it run inside the user's
 * authenticated session.
 *
 * History/tab state mirrors the regular web-fetch path so back/forward
 * and the page-tab strip work consistently.
 */
function _renderZimArticle(zimUrl, options = {}) {
  // ``skipHistoryPush`` is set when restoring a back/forward navigation —
  // we're already at the right history index, so pushing a new entry
  // would duplicate the URL and break subsequent navigation.
  const { skipHistoryPush = false } = options;

  // Strip the "zim:" prefix and split into pack_id + entry_path.
  const tail = zimUrl.slice(4);
  const slashIdx = tail.indexOf('/');
  if (slashIdx <= 0) {
    showArticleError(`Malformed pack URL: ${zimUrl}`);
    return;
  }
  const packId = tail.slice(0, slashIdx);
  const entryPath = tail.slice(slashIdx + 1);
  // Pass current theme so the served HTML's reader-mode CSS matches
  // Augmentum's palette. Theme picker writes augmentum-theme; default 'dark'
  // mirrors what grove.js does on first load. Allowed values: dark, light,
  // midnight, sepia (server falls back to dark on unknown).
  let theme = 'dark';
  try { theme = localStorage.getItem('augmentum-theme') || 'dark'; } catch {}
  // Reader-mode toggle persisted per-browser. Default on; user clicks
  // the "Raw" pill in iframe chrome to opt out when our reader CSS
  // fights an SPA's own theme (devdocs/fcc occasionally). The flag is
  // an escape hatch, NOT a pack-by-pack preference — same toggle
  // applies across every ZIM the user opens until they flip it back.
  let readerOn = true;
  try { readerOn = (localStorage.getItem('augmentum-zim-reader') ?? 'on') !== 'off'; } catch {}
  const readerParam = readerOn ? '' : '&reader=off';
  const iframeSrc = `/api/knowledge/zim/${encodeURIComponent(packId)}/${entryPath}?theme=${encodeURIComponent(theme)}${readerParam}`;
  // Track the URL we expect the iframe to load so the load handler can
  // distinguish "this was our navigation" (skip) from "user clicked an
  // internal link" (push to history). Without this every internal link
  // would either get double-pushed (initial load + nav) or invisible
  // (parent never sees iframe-internal nav at all, so back goes to home).
  _zimExpectedIframePath = iframeSrc;

  // Title heuristic: take the last path segment, replace underscores
  // with spaces. Good enough for MediaWiki-style ZIMs ("A/Type_2_diabetes"
  // → "Type 2 diabetes"). The iframe's own <title> would be more accurate
  // but we can't read it cross-frame under sandbox without postMessage,
  // and a slight title lag isn't worth that complexity for v1.
  const lastSeg = entryPath.split('/').pop() || entryPath;
  const title = decodeURIComponent(lastSeg).replace(/_/g, ' ');

  state.currentUrl = zimUrl;
  showReaderView();

  // Companion presence: knowledge-pack articles are pages too —
  // "tell me about this page" while reading a Wikipedia ZIM article
  // should bind to the article, not become a literal search.
  import('./architect-observer.js')
    .then(m => m.reportAttention('surface.browse.page_opened', { url: zimUrl, title }))
    .catch(() => {});

  // Flatten the reader-view layout for ZIM mode — see the .is-zim CSS
  // rules for why. Toggling on contentArea kills its padding so the iframe
  // sits flush; toggling on readerView removes the 760px max-width that
  // would otherwise shrink the iframe to a small centered box.
  dom.contentArea?.classList.add('is-zim');
  dom.readerView?.classList.add('is-zim');

  const container = dom.readerView;
  if (container) {
    // Iframe sized to fill the reader. The wrapper div carries the same
    // class the article view uses so adjacent UI (ask bar, AI tools)
    // continues to anchor correctly.
    //
    // Sandbox tokens:
    //  - allow-same-origin: iframe shares cookies + can fetch
    //    /api/knowledge/zim/... resources with the user's session.
    //  - allow-scripts: required for SPA-style ZIMs (freeCodeCamp,
    //    DevDocs, Stack Exchange mirrors) whose content only renders
    //    after a Vite/React bundle runs. Static ZIMs (Wikipedia, MDWiki)
    //    don't ship scripts so they're unaffected. The CSP served with
    //    each /api/knowledge/zim/* response (script-src 'self'
    //    'unsafe-inline' 'unsafe-eval') is the actual policy boundary.
    //  - allow-forms: some SPAs use forms for in-bundle navigation /
    //    interactive exercises (freeCodeCamp curriculum tasks).
    //  - allow-popups + allow-popups-to-escape-sandbox: external article
    //    links (target="_blank", added by the server-side rewriter) open
    //    in a fresh new tab WITHOUT inheriting our sandbox.
    //
    // Trust note: same-origin + allow-scripts is documented as
    // sandbox-escape-capable. When the operator enables content
    // isolation (settings.content_iframe_isolation_enabled +
    // wired Caddy listener), `_tryUpgradeZimIframeToIsolated` below
    // replaces the iframe src with the isolated-origin URL after the
    // element renders — at that point allow-same-origin is safe
    // because "same-origin" inside the iframe means the isolated
    // origin, not Augmentum's main API. When isolation is off the
    // iframe stays on the main origin; the trust model falls back
    // to "user-trusted ZIM content from kiwix mirrors" + the CSP
    // (script-src 'self' 'unsafe-inline' 'unsafe-eval') bounding
    // what the scripts can do.
    // Reader-mode pill: small absolute-positioned toggle in the
    // top-right of the iframe wrapper. Click flips the persisted
    // setting and reloads the iframe with the new query param. Label
    // mirrors the current state ("Reader" when on, "Raw" when off) so
    // the user always knows what they'll get next.
    const readerPillLabel = readerOn ? 'Reader' : 'Raw';
    const readerPillTitle = readerOn
      ? 'Augmentum reader mode is on. Click for raw site presentation.'
      : 'Showing raw site presentation. Click to re-enable Augmentum reader mode.';
    container.innerHTML = `
      <div class="browse-zim-wrapper">
        <div class="browse-zim-chrome">
          <button type="button"
            class="browse-zim-meta-toggle"
            data-pack-id="${escapeHtml(packId)}"
            data-pack-name="${escapeHtml(title)}"
            title="Show pack info"
            aria-label="Show pack info"
            aria-expanded="false">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" width="14" height="14" aria-hidden="true"><circle cx="8" cy="8" r="6.5"/><path d="M8 7v4M8 4.8v.4"/></svg>
          </button>
          <button type="button"
            class="browse-zim-reader-toggle"
            data-reader-on="${readerOn ? '1' : '0'}"
            title="${escapeHtml(readerPillTitle)}"
            aria-pressed="${readerOn ? 'true' : 'false'}">${escapeHtml(readerPillLabel)}</button>
        </div>
        <iframe
          class="browse-zim-frame"
          src="${escapeHtml(iframeSrc)}"
          sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-popups-to-escape-sandbox"
          referrerpolicy="no-referrer"
          loading="eager"
          title="${escapeHtml(title)}"
        ></iframe>
        <aside class="browse-zim-meta-panel" hidden aria-label="Pack info"></aside>
      </div>
    `;
    const toggle = container.querySelector('.browse-zim-reader-toggle');
    if (toggle) {
      toggle.addEventListener('click', () => {
        const nextOn = toggle.dataset.readerOn !== '1';
        try {
          localStorage.setItem('augmentum-zim-reader', nextOn ? 'on' : 'off');
        } catch { /* private browsing / quota — reader pref won't persist */ }
        // Re-render the article with the new flag. _renderZimArticle
        // reads the localStorage value so a single recursive call is
        // the cleanest way to apply the change everywhere (URL, ETag,
        // history entry, page-tab strip).
        _renderZimArticle(zimUrl, { skipHistoryPush: true });
      });
    }

    // Metadata sidebar: lazy-fetched on first open, toggled visible /
    // hidden thereafter. Esc closes if open. Clicking the iframe area
    // also closes (modal pattern — clearer than expecting users to find
    // the toggle again to dismiss).
    const metaToggle = container.querySelector('.browse-zim-meta-toggle');
    const metaPanel = container.querySelector('.browse-zim-meta-panel');
    if (metaToggle && metaPanel) {
      const closeMeta = () => {
        metaPanel.hidden = true;
        metaPanel.classList.remove('is-open');
        metaToggle.setAttribute('aria-expanded', 'false');
      };
      const openMeta = () => {
        metaPanel.hidden = false;
        // Force layout flush so the .is-open transition runs from hidden→visible.
        // Without this, the panel pops in instantly on first open.
        requestAnimationFrame(() => metaPanel.classList.add('is-open'));
        metaToggle.setAttribute('aria-expanded', 'true');
        _renderZimMetaPanel(metaPanel, packId, title);
      };
      metaToggle.addEventListener('click', () => {
        if (metaPanel.hidden || !metaPanel.classList.contains('is-open')) {
          openMeta();
        } else {
          closeMeta();
        }
      });
      // Esc closes the panel when open.
      const escHandler = (e) => {
        if (e.key === 'Escape' && !metaPanel.hidden) {
          closeMeta();
          e.stopPropagation();
        }
      };
      document.addEventListener('keydown', escHandler);
      // Clean up when the reader view is dismantled. The next render of
      // the reader area will replace these nodes; the keydown listener
      // would otherwise leak across navigations.
      metaPanel.addEventListener('augmentum:cleanup', () => {
        document.removeEventListener('keydown', escHandler);
      }, { once: true });
    }
    // Hook the iframe's load event so internal MediaWiki link clicks
    // become entries in Augmentum's back/forward stack. Without this
    // the parent never sees nav inside the iframe, and back jumps to
    // whatever was open before the article (or to the home screen
    // when the article was the cold-open from a chat citation).
    // The same handler also extracts the article text and stashes it on
    // state.currentContent so the Ask bar / AI tools work against the
    // ZIM article (otherwise they show "No page content to analyze").
    const iframe = container.querySelector('.browse-zim-frame');
    if (iframe) {
      iframe.addEventListener('load', () => _onZimIframeNavigate(iframe));
      // Best-effort upgrade to the isolated origin. Fires fire-and-
      // forget after the iframe is in the DOM so the user sees the
      // same-origin fallback render immediately if isolation is off
      // (no flicker). When the mint succeeds, the iframe src is
      // swapped to the isolated origin and the new load triggers a
      // single re-render with the safer trust boundary. Failures
      // (501 isolation off, network error, mint denial) leave the
      // same-origin src untouched.
      _tryUpgradeZimIframeToIsolated(iframe, packId, iframeSrc);
    }
  }

  // Push to back/forward history so navigation works the same way as
  // web-fetched articles. Same shape as the entry pushed in the regular
  // browseFetch path. Skipped when restoreHistoryEntry is calling us —
  // it's already at the right index.
  if (!skipHistoryPush) {
    const entry = { url: zimUrl, title, scrollPos: 0 };
    if (state.historyIdx < state.history.length - 1) {
      state.history = state.history.slice(0, state.historyIdx + 1);
    }
    state.history.push(entry);
    state.historyIdx = state.history.length - 1;
  }

  // Update the active page tab so the tab strip shows the article title.
  const activeTab = _activePageTab();
  if (activeTab) {
    activeTab.url = zimUrl;
    activeTab.title = title;
    activeTab.favicon = '';  // ZIMs don't carry favicons; leave blank
    _renderPageTabStrip();
    _savePageTabsToStorage();
  }

  if (dom.searchInput) dom.searchInput.value = title;
  if (dom.readerAiBlocks) dom.readerAiBlocks.innerHTML = '';
  updateNavButtons();
}

// Tracks the URL we just told the iframe to load. The first load event
// after a render/restore is our own navigation — we want to skip it. Any
// subsequent load is an internal MediaWiki link the user clicked, and we
// push it as a real Augmentum history entry.
let _zimExpectedIframePath = null;

/** Iframe load handler — converts internal ZIM navigation to Augmentum
 * history entries.
 *
 * The iframe is same-origin, so we can read ``contentWindow.location``
 * and ``contentDocument.title`` directly (no postMessage shim needed).
 * Cross-origin would block this; we don't expect that case because
 * external links carry ``target="_blank"`` and open in a new tab.
 *
 * Edge cases handled:
 *   - Initial load: URL matches what we just set; skip the push.
 *   - Restore from back/forward: same — _renderZimArticle sets the
 *     expected URL before recreating the iframe.
 *   - Iframe failed to load (404, etc.): contentWindow access throws or
 *     URL is about:blank; bail silently rather than push garbage.
 */
function _onZimIframeNavigate(iframe) {
  let absoluteUrl = '';
  try {
    absoluteUrl = iframe.contentWindow?.location?.href || '';
  } catch {
    return;  // cross-origin (shouldn't happen) or detached frame
  }
  if (!absoluteUrl || absoluteUrl === 'about:blank') return;

  // Always capture article text for state.currentContent BEFORE the
  // expected-path early-return. The initial load (which matches the
  // expected path) is exactly when the user wants Ask / AI tools to
  // work — bailing early there left them with the "No page content"
  // toast on every fresh article.
  _captureZimContentToState(iframe);

  // Let cross-cutting features wire into the freshly-loaded article DOM.
  // (Language learning attaches click-to-define here, but only when the
  // Learning toggle is on — otherwise this is a no-op.)
  document.dispatchEvent(new CustomEvent('augmentum:browse-iframe-loaded', {
    detail: { iframe, url: absoluteUrl },
  }));

  // Strip origin so we can compare against the relative path we asked for.
  const path = absoluteUrl.replace(/^https?:\/\/[^\/]+/, '');

  // Skip if this load matches what we just told the iframe to render —
  // that's our own navigation, not a user click. Comparison ignores
  // query string drift (theme param) so a re-render with the same theme
  // counts as the same URL.
  const stripQuery = (s) => s.replace(/\?.*$/, '');
  if (_zimExpectedIframePath && stripQuery(path) === stripQuery(_zimExpectedIframePath)) {
    _zimExpectedIframePath = null;
    return;
  }

  // Parse /api/knowledge/zim/{pack_id}/{path}?...
  const match = path.match(/^\/api\/knowledge\/zim\/([^\/]+)\/(.+?)(?:\?|$)/);
  if (!match) return;

  const newPackId = decodeURIComponent(match[1]);
  const newEntryPath = match[2];
  const newZimUrl = `zim:${newPackId}/${newEntryPath}`;

  // Read the iframe's title for a friendlier history entry. Falls back
  // to the URL's last segment when the article has no title element.
  let title = newEntryPath.split('/').pop() || newEntryPath;
  try {
    const docTitle = (iframe.contentDocument?.title || '').trim();
    if (docTitle) title = docTitle;
  } catch { /* ignore */ }
  title = decodeURIComponent(title.replace(/_/g, ' '));

  // Don't double-push if we somehow re-fired on the same URL.
  const cur = state.history[state.historyIdx];
  if (cur && cur.url === newZimUrl) return;

  state.currentUrl = newZimUrl;
  // Same shape as _renderZimArticle's history push. Truncate any forward
  // history first — the user just made a new branch.
  if (state.historyIdx < state.history.length - 1) {
    state.history = state.history.slice(0, state.historyIdx + 1);
  }
  state.history.push({ url: newZimUrl, title, scrollPos: 0 });
  state.historyIdx = state.history.length - 1;

  // Mirror _renderZimArticle's UI updates so the chrome reflects the
  // new article (search input shows title, page tab strip updates).
  if (dom.searchInput) dom.searchInput.value = title;
  const activeTab = _activePageTab();
  if (activeTab) {
    activeTab.url = newZimUrl;
    activeTab.title = title;
    activeTab.favicon = '';
    _renderPageTabStrip();
    _savePageTabsToStorage();
  }
  // (state.currentContent already updated at top of handler.)
  updateNavButtons();
}

/** Read the current iframe document into state.currentContent so the
 * Ask bar and AI tools work against the ZIM article. The iframe is
 * served from /api/knowledge/zim on the same origin we run from, so
 * cross-origin restrictions don't apply.
 *
 * Failures are swallowed and leave state.currentContent at its
 * previous value — Ask bar will fall back to its existing "No page
 * content to analyze" toast, which is the right behavior for the
 * "iframe not ready" race rather than crashing.
 */
function _captureZimContentToState(iframe) {
  if (!iframe) return;
  try {
    const doc = iframe.contentDocument;
    if (!doc) return;
    // .innerText respects styling (display:none chrome we hid via CSS
    // is excluded), unlike .textContent which would include it. The
    // server-side reader CSS already hid Vector skin, so this is just
    // the article body text.
    const body = doc.body;
    if (!body) return;
    const text = (body.innerText || '').trim();
    if (!text) return;
    // Cap at 100K chars — Wikipedia articles can exceed that and big
    // payloads stall both the LLM call and ambient-context routing.
    state.currentContent = text.slice(0, 100000);
    state.currentHtml = doc.documentElement.outerHTML.slice(0, 200000);
    state.currentPageType = 'article';
    // Notify subscribed chat surfaces that ambient context is available
    // for the just-rendered ZIM article. Mirrors the dispatch the web
    // fetch path does on every render.
    document.dispatchEvent(new CustomEvent('augmentum:browse-extracted', {
      detail: {
        url: state.currentUrl || '',
        title: doc.title || '',
        text: state.currentContent,
      },
    }));
  } catch {
    // Cross-origin (shouldn't happen for our route) or detached frame.
    // Leave state.currentContent untouched.
  }
}

/** Public entry point: open Browse on a specific URL.
 *
 * Used by chat citations and other surfaces that want to open content
 * in the Browse panel. Accepts both regular web URLs and ``zim:`` URLs;
 * the dispatch happens inside ``browseFetch``. Empty-state safe — does
 * nothing if the user closes Browse before the fetch resolves.
 */
/**
 * Architect entry — open the browse panel and run a search.
 *
 * Triggered by intent-action-router on a ``browse.search`` channel
 * event (architect's web.search primitive). Opens the panel if it's
 * hidden, sets the search input, kicks off the search. Optional
 * ``category`` narrows the result set ("news", "academic", etc.).
 */
export function browseSearchByQuery(query, category = '') {
  if (!query || !String(query).trim()) return;
  // Open the panel + give the search bar time to mount.
  openBrowsePanel({ skipAutoFocus: true });
  // Drive the existing search path so site:filters / history /
  // provider chip handling all run unchanged.
  setTimeout(() => {
    try { browseSearch(String(query).trim(), category || ''); }
    catch (err) { console.warn('[browse] architect search failed', err); }
  }, 0);
}

export function openInBrowse(url) {
  if (!url) return;
  // skipAutoFocus prevents the search input from being focused on open —
  // otherwise the recent-searches dropdown briefly flashes before the
  // article render covers it. The user clicked a citation; they want to
  // read the source, not search.
  openBrowsePanel({ skipAutoFocus: true });
  // Fire-and-forget — browseFetch handles its own errors via showArticleError.
  browseFetch(url);
}

function showArticleLoading() {
  const container = dom.readerView;
  if (!container) return;
  container.innerHTML = `
    <div class="browse-loading">
      <div class="browse-skeleton browse-skeleton-title"></div>
      <div class="browse-skeleton browse-skeleton-meta"></div>
      <div class="browse-skeleton browse-skeleton-line"></div>
      <div class="browse-skeleton browse-skeleton-line"></div>
      <div class="browse-skeleton browse-skeleton-line"></div>
      <div class="browse-skeleton browse-skeleton-line" style="width:60%"></div>
      <div class="browse-skeleton browse-skeleton-line"></div>
      <div class="browse-skeleton browse-skeleton-line"></div>
      <div class="browse-skeleton browse-skeleton-line" style="width:80%"></div>
    </div>
  `;
}

/**
 * Render the backend's "unsupported site" response as a friendly card
 * with a favicon, hostname, reason text, and a big Open-in-browser CTA.
 * Used when the backend short-circuits a known-hostile domain (Amazon,
 * Facebook, LinkedIn, Google Maps, etc.) that never returns useful
 * content to a reader.
 */
function showUnsupportedSite(data) {
  const container = dom.readerView;
  if (!container) return;
  const url = data.url || state.currentUrl || '';
  const hostname = data.hostname || '';
  const reason = data.reason || '';
  const favicon = data.favicon_url || _faviconForUrl(url);
  container.innerHTML = `
    <div class="browse-unsupported">
      <div class="browse-unsupported-icon">
        ${favicon
          ? `<img src="${escapeHtml(favicon)}" alt="" width="56" height="56" onerror="this.remove()">`
          : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="40" height="40"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>`
        }
      </div>
      <h3 class="browse-unsupported-title">Can't read this page here</h3>
      <p class="browse-unsupported-host">${escapeHtml(hostname || 'This site')}</p>
      ${reason ? `<p class="browse-unsupported-reason">${escapeHtml(reason)}</p>` : ''}
      <div class="browse-unsupported-actions">
        <a class="browse-unsupported-cta" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">
          <span>Open in your browser</span>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
            <polyline points="15 3 21 3 21 9"/>
            <line x1="10" y1="14" x2="21" y2="3"/>
          </svg>
        </a>
      </div>
      <p class="browse-unsupported-hint">Sites that need a real browser session
         — logins, bot checks, JavaScript apps — can't be displayed in the reader.</p>
    </div>
  `;
}

function showArticleError(msg) {
  const container = dom.readerView;
  if (!container) return;
  const url = state.currentUrl || '';
  container.innerHTML = `
    <div class="browse-error">
      <div class="browse-error-icon browse-error-icon--error">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="28" height="28">
          <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
      </div>
      <h3 class="browse-error-title">Cannot load page</h3>
      <p class="browse-error-message">${escapeHtml(msg)}</p>
      <div class="browse-error-actions">
        <button class="browse-error-retry" data-action="retry-fetch">Retry</button>
        ${url ? `<button class="browse-error-original" data-action="open-original">Open original</button>` : ''}
      </div>
    </div>
  `;
  container.querySelector('[data-action="retry-fetch"]')?.addEventListener('click', () => {
    if (url) browseFetch(url);
  });
  container.querySelector('[data-action="open-original"]')?.addEventListener('click', () => {
    if (url) window.open(url, '_blank', 'noopener,noreferrer');
  });
}

function _addEmbedBadges(container) {
  if (!window.MediaCards) return;
  container.querySelectorAll('iframe').forEach(iframe => {
    const src = iframe.src || iframe.dataset?.src || '';
    if (!src) return;
    const platform = MediaCards.detectPlatform(src);
    if (platform === 'unknown') return;
    const p = MediaCards.PLATFORMS[platform];
    if (!p) return;

    // Wrap iframe in relative container if not already wrapped
    let wrapper = iframe.parentElement;
    if (!wrapper.classList.contains('browse-embed-wrap')) {
      wrapper = document.createElement('div');
      wrapper.className = 'browse-embed-wrap';
      wrapper.style.position = 'relative';
      iframe.parentNode.insertBefore(wrapper, iframe);
      wrapper.appendChild(iframe);
    }

    // Add platform badge
    const badge = document.createElement('span');
    badge.className = 'browse-embed-badge';
    badge.style.color = p.color;
    badge.textContent = `${p.icon} ${p.label}`;
    wrapper.insertBefore(badge, iframe);

    // Add "Open in Panel" button below
    const openBtn = document.createElement('button');
    openBtn.className = 'browse-embed-open';
    openBtn.innerHTML = `Open in Media Panel &rarr;`;
    openBtn.addEventListener('click', () => {
      window.dispatchEvent(new CustomEvent('media:play', {
        detail: { url: src, platform, title: '', channel: '' }
      }));
    });
    wrapper.appendChild(openBtn);

    // Cast button — one-click send to TV without detouring through the
    // floating-video player. Uses display.web_show@1 (iframe load on
    // the TV's html.generic surface) — the only capability that works
    // for YouTube/Vimeo/etc. since their watch URLs aren't valid
    // <video src> targets. media.video_play@1 would silently fall
    // through to native media.video and show a grey screen on the TV.
    const castBtn = mountCastButton({
      capability: 'display.web_show@1',
      className: 'browse-embed-cast',
      title: `Cast to TV`,
      getContent: () => ({
        contentUrl: src,
        title: iframe.title || document.title || `${p.label} video`,
        contentKey: src,
        metadata: { platform, source: 'browse-embed' },
      }),
      onCast: () => {
        // Stop the local iframe so the embed doesn't keep playing
        // here after handoff. Clearing src is the only reliable way
        // to silence a YouTube/Vimeo iframe without their JS API
        // (which requires enablejsapi=1 — not always present in the
        // article-extracted URL). Replace the iframe with a quiet
        // "Casting to TV" placeholder so the layout doesn't jump.
        try {
          const placeholder = document.createElement('div');
          placeholder.className = 'browse-embed-casting';
          placeholder.textContent = 'Casting to TV — use the cast pill to control';
          iframe.replaceWith(placeholder);
        } catch (err) {
          console.warn('[browse] embed iframe teardown failed', err);
          try { iframe.src = ''; } catch {}
        }
        import('./cast-shelf.js')
          .then(m => m.notifyCastStarted?.({ openShelf: true }))
          .catch(() => {});
      },
    });
    wrapper.appendChild(castBtn);
  });
}

// CSP frame-src allowlist mirror — must match the server's CSP header. Any
// iframe whose src host isn't on this list will be blocked by the browser
// anyway; we strip them here so the article view doesn't render a broken
// "Framing '<host>' violates Content Security Policy" placeholder.
// Biggest offender: WordPress's self-referential `/embed/` oEmbed block
// (TechCrunch, every WP-backed publisher) — the extractor keeps it, but
// it's useless inside a view that already IS the article.
const _FRAME_SRC_ALLOWED_HOSTS = [
  'www.youtube.com', 'youtube.com', 'youtube-nocookie.com', 'www.youtube-nocookie.com',
  'player.vimeo.com', 'vimeo.com',
  'www.dailymotion.com', 'dailymotion.com',
  'www.tiktok.com', 'tiktok.com',
  'clips.twitch.tv',
  'platform.twitter.com',
];
function _isFrameHostAllowed(host) {
  if (!host) return false;
  if (_FRAME_SRC_ALLOWED_HOSTS.includes(host)) return true;
  // Wildcards from the server CSP: *.dailymotion.com.
  return /\.dailymotion\.com$/i.test(host);
}
function _stripDisallowedFrames(html) {
  if (!html || !html.includes('<iframe')) return html;
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  tmp.querySelectorAll('iframe').forEach((f) => {
    const src = f.getAttribute('src') || '';
    let host = '';
    try { host = new URL(src, window.location.href).hostname; } catch {}
    if (!_isFrameHostAllowed(host)) f.remove();
  });
  return tmp.innerHTML;
}

function renderArticle(data) {
  const container = dom.readerView;
  if (!container) return;

  const metaParts = [];
  if (data.author) metaParts.push(`<span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>${escapeHtml(data.author)}</span>`);
  if (data.date) metaParts.push(`<span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>${escapeHtml(data.date)}</span>`);
  if (data.word_count) metaParts.push(`<span>${data.word_count.toLocaleString()} words</span>`);
  if (data.reading_time_min) metaParts.push(`<span>${data.reading_time_min} min read</span>`);

  let bodyHtml;
  if (data.error) {
    const errUrl = escapeHtml(data.url || '');
    bodyHtml = `<div class="browse-error">
      <div class="browse-error-icon browse-error-icon--warning">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="28" height="28">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
      </div>
      <h3 class="browse-error-title">Content unavailable</h3>
      <p class="browse-error-message">${escapeHtml(data.error)}</p>
      <div class="browse-error-actions">
        ${errUrl ? `<a class="browse-error-original" href="${errUrl}" target="_blank" rel="noopener noreferrer">Open original</a>` : ''}
      </div>
    </div>`;
  } else if (data.html) {
    bodyHtml = typeof DOMPurify !== 'undefined' ? _stripDisallowedFrames(DOMPurify.sanitize(data.html, {
      ALLOWED_TAGS: ['h1','h2','h3','h4','h5','h6','p','br','hr','ul','ol','li',
        'table','thead','tbody','tr','th','td','caption','colgroup','col',
        'blockquote','pre','code','em','strong','b','i','u','s','del','ins','mark',
        'a','img','figure','figcaption','picture','source',
        'video','audio','iframe','embed',
        'div','span','section','article','aside','header','footer','nav','main',
        'details','summary','dl','dt','dd','abbr','time','sub','sup','small',
        'ruby','rt','rp','bdo','wbr','cite','q','dfn','var','samp','kbd',
        'svg','path','circle','rect','line','polyline','polygon','text','g','defs',
        'clipPath','use','symbol','title','desc','foreignObject','tspan',
        'linearGradient','radialGradient','stop','pattern','mask','filter',
        'feGaussianBlur','feOffset','feBlend','feMerge','feMergeNode'],
      ALLOWED_ATTR: ['href','src','alt','title','class','id','width','height',
        'colspan','rowspan','scope','headers','controls','poster','preload',
        'loading','decoding','target','rel','type','start','role',
        'aria-label','aria-describedby','dir','lang',
        // Native media attributes — required by the direct-media + imgur-gifv
        // paths in browse_routes so the server can emit <video autoplay muted
        // loop> for silent-loop content and inline playback on iOS.
        'playsinline','autoplay','muted','loop','crossorigin',
        'data-src','data-srcset','srcset','sizes','media',
        'viewBox','xmlns','fill','stroke','stroke-width','stroke-linecap',
        'stroke-linejoin','d','cx','cy','r','x','y','x1','y1','x2','y2',
        'points','transform','opacity','style'],
      ALLOW_DATA_ATTR: false,
      ADD_TAGS: ['iframe'],
      ADD_ATTR: ['allowfullscreen','frameborder','allow','sandbox'],
      ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto|tel|data):|[^a-z]|[a-z+.-]+(?:[^a-z+.:-]|$))/i,
      FORBID_TAGS: ['script','style','noscript','form','input','select','textarea','button','object','applet'],
      // Belt-and-suspenders: DOMPurify's allowlist approach already strips
      // anything not in ALLOWED_ATTR, so on* handlers are blocked by
      // omission. Listing them in FORBID_ATTR documents the security
      // posture and guards against future allowlist drift (e.g., if
      // someone adds 'onclick' to support a legitimate use without
      // realising it opens XSS). javascript: URIs in href/src are
      // separately blocked by ALLOWED_URI_REGEXP above.
      FORBID_ATTR: ['onload','onerror','onclick','onmouseover','onmouseout',
        'onfocus','onblur','onsubmit','onchange','onkeydown','onkeyup',
        'onkeypress','ontouchstart','ontouchend','onpointerdown','onpointerup',
        'formaction'],
    })) : data.html;
  } else if (data.text) {
    bodyHtml = data.text.split(/\n\n+/).map(p => `<p>${escapeHtml(p)}</p>`).join('');
  } else {
    bodyHtml = '<p style="color:var(--text-muted);font-style:italic">No content could be extracted from this page.</p>';
  }

  container.innerHTML = `
    <div class="browse-article-header">
      <div class="browse-article-site">
        <img src="${escapeHtml(data.favicon_url || '')}" alt="" width="16" height="16" onerror="this.style.display='none'">
        <span>${escapeHtml(data.sitename || '')}</span>
        ${data.source && data.source !== 'direct' ? `<span class="browse-article-source-badge" data-source="${escapeHtml(data.source)}">${escapeHtml({
          'wayback': 'Archived version',
          'json-ld': 'Structured data',
          'amp': 'AMP version',
          'rss': 'RSS feed',
        }[data.source] || data.source)}</span>` : ''}
      </div>
      <h1 class="browse-article-title" dir="auto">${escapeHtml(data.title || 'Untitled')}</h1>
      ${metaParts.length ? `<div class="browse-article-meta">${metaParts.join('')}</div>` : ''}
    </div>
    <div class="browse-article-toolbar">
      <div class="browse-article-toolbar-group">
        <button class="browse-action-pill" data-article-ai="summarize"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="21" y1="10" x2="3" y2="10"/><line x1="21" y1="6" x2="3" y2="6"/><line x1="21" y1="14" x2="3" y2="14"/><line x1="15" y1="18" x2="3" y2="18"/></svg> Summarize</button>
        <button class="browse-action-pill" data-article-ai="keypoints"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg> Key Points</button>
        <button class="browse-action-pill" data-article-ai="explain"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> Explain</button>
        <button class="browse-action-pill" data-article-ai="extract"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg> Extract</button>
        <button class="browse-action-pill" data-article-ai="translate"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 8l6 6"/><path d="M4 14l6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="M22 22l-5-10-5 10"/><path d="M14 18h6"/></svg> Translate</button>
      </div>
      <div class="browse-article-toolbar-group">
        <button class="browse-action-pill" data-article-action="toc" title="On this page" aria-label="On this page" style="display:none"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg> Contents</button>
        <button class="browse-action-pill" data-article-action="typography" title="Reading preferences" aria-label="Reading preferences"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7V4h16v3"/><path d="M9 20h6"/><path d="M12 4v16"/></svg> Aa</button>
        <button class="browse-action-pill" data-article-action="read-aloud" title="Read aloud" aria-label="Read aloud"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 14h4l5-4v8l-5-4H3z"/><path d="M15 8a5 5 0 0 1 0 8"/><path d="M18 5a9 9 0 0 1 0 14"/></svg><span class="browse-read-aloud-label">Listen</span></button>
        <button class="browse-action-pill" data-article-action="bookmark" title="Bookmark (Ctrl+D)" aria-label="Bookmark this page"><svg class="browse-bookmark-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg> <span class="browse-bookmark-label">Bookmark</span></button>
        <button class="browse-action-pill" data-article-action="discuss"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg> Discuss</button>
        <button class="browse-action-pill" data-article-action="save-note"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg> Save as Note</button>
        <button class="browse-action-pill" data-article-action="save"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/></svg> Save to Context</button>
        <a class="browse-action-pill" href="${escapeHtml(data.url || '')}" target="_blank" rel="noopener noreferrer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg> Original</a>
      </div>
    </div>
    <div class="browse-article-body" dir="auto">${bodyHtml}</div>
  `;

  // Clear previous AI blocks
  if (dom.readerAiBlocks) dom.readerAiBlocks.innerHTML = '';
  if (dom.contentArea) dom.contentArea.scrollTop = 0;
  // Article body just landed — let layout settle, then refresh the
  // scroll-jump button state (scrollTop didn't change if we were
  // already at 0, so the scroll listener won't fire).
  requestAnimationFrame(updateScrollJumpButtons);

  // Post-render image quality check — hide tiny/broken images after they load.
  // The server filters most junk, but images without explicit dimensions in HTML
  // can only be checked after the browser loads them.
  const articleBody = dom.readerView?.querySelector('.browse-article-body');
  if (articleBody) {
    // Apply the user's persisted reader prefs (font, width, spacing)
    // before the rest of the post-render work, so subsequent image +
    // TOC measurements see the final typography and don't jitter.
    _applyReaderPrefs();

    // Sync the bookmark button's filled-state to the current URL.
    _refreshBookmarkButtonState();

    // Render any TeX/MathML in the article. No-op when KaTeX isn't
    // vendored; fire-and-forget so image checks below aren't blocked by
    // KaTeX's lazy script load on the first math-bearing page.
    renderMathIn(articleBody);

    // Syntax-highlight code blocks in the article. Covers GitHub READMEs,
    // Stack Exchange answers, arXiv abstracts, MDN, Read the Docs,
    // random dev blogs — any fetched or intercept-rendered HTML with
    // <pre><code>. Deferred to idle so initial paint stays snappy; the
    // hljs global is populated by lib/highlight.js/highlight.min.js.
    _highlightArticleCodeDeferred(articleBody);

    // Decorate code blocks with copy buttons once highlighted.
    _addCopyButtonsToCodeBlocks(articleBody);

    // Build a table-of-contents side panel for long articles (3+ h1/h2
    // headings). Covers GitHub READMEs, MDN, Read the Docs, most blog
    // posts with headings. Hidden when there aren't enough headings to
    // be worth it.
    _buildArticleToc(articleBody);
    for (const img of articleBody.querySelectorAll('img')) {
      img.addEventListener('load', function _imgQualityCheck() {
        // Hide images that rendered as tiny (icons, spacers that bypassed server filter)
        if (this.naturalWidth < 40 || this.naturalHeight < 40) {
          this.style.display = 'none';
          return;
        }
        // Hide images that are extremely narrow or tall (decorative lines, separators)
        const ratio = this.naturalWidth / this.naturalHeight;
        if (ratio > 12 || ratio < 0.08) {
          this.style.display = 'none';
        }
      }, { once: true });
      img.addEventListener('error', function() {
        this.style.display = 'none';
      }, { once: true });
      // Check images already loaded (cached from prior visit)
      if (img.complete && img.naturalWidth > 0) {
        if (img.naturalWidth < 40 || img.naturalHeight < 40) {
          img.style.display = 'none';
        } else {
          const ratio = img.naturalWidth / img.naturalHeight;
          if (ratio > 12 || ratio < 0.08) img.style.display = 'none';
        }
      } else if (img.complete && img.naturalWidth === 0) {
        img.style.display = 'none'; // Broken image
      }
    }

    // Wire "Ask about this video" buttons for embedded videos with transcripts
    for (const askBtn of articleBody.querySelectorAll('.browse-video-ask-btn')) {
      askBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const wrapper = e.target.closest('.browse-video-ask');
        if (!wrapper) return;
        const videoId = wrapper.dataset.videoId;
        const videoTitle = wrapper.dataset.videoTitle;
        const video = state.currentVideos?.find(v => v.video_id === videoId);
        if (!video?.transcript) return;
        // Focus the ask bar and pre-fill with video context hint
        if (dom.readerAskInput) {
          dom.readerAskInput.placeholder = `Ask about "${videoTitle}"...`;
          dom.readerAskInput.focus();
          // Store video-specific context for the next ask action
          state._videoAskContext = video;
        }
      });
    }

    // Overlay platform badges on embedded video iframes
    _addEmbedBadges(articleBody);
  }
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function browseBack() {
  if (state.historyIdx <= 0) {
    if (state.historyIdx === 0) {
      state.history[0].scrollPos = dom.contentArea?.scrollTop || 0;
    }
    state.historyIdx = -1;
    showSearchView();
    if (state.searchResults) renderSearchResults();
    if (dom.searchInput) dom.searchInput.value = state.searchQuery;
    updateNavButtons();
    return;
  }

  state.history[state.historyIdx].scrollPos = dom.contentArea?.scrollTop || 0;
  state.historyIdx--;
  restoreHistoryEntry();
}

function browseForward() {
  if (state.historyIdx >= state.history.length - 1) return;

  if (state.historyIdx >= 0) {
    state.history[state.historyIdx].scrollPos = dom.contentArea?.scrollTop || 0;
  }
  state.historyIdx++;
  restoreHistoryEntry();
}

// Jump straight to the landing surface from anywhere in history. Mirrors
// the "back-out-to-empty" terminal state of ``browseBack`` so users don't
// have to step back through every visited entry. History stays intact —
// pressing Forward after Home returns to the last viewed page.
//
// ``_restoreLanding()`` is called unconditionally because focused views
// (per-pack home, bookmarks gallery) overwrite ``searchView.innerHTML``;
// without the rebuild, Home from inside one of those leaves the user
// staring at the prior view's stale markup. The helper is a no-op when
// no overwrite has happened, so the call is cheap on the common path.
function browseHome() {
  if (state.historyIdx >= 0) {
    state.history[state.historyIdx].scrollPos = dom.contentArea?.scrollTop || 0;
  }
  state.historyIdx = -1;
  // Clear ZIM-mode layout classes that the iframe path may have set
  // (mirror restoreHistoryEntry's reset-on-exit).
  dom.contentArea?.classList.remove('is-zim');
  dom.readerView?.classList.remove('is-zim');
  // Drop any in-flight search results so the user actually lands on the
  // landing surface — chips, recents, bookmarks, packs — rather than
  // re-rendering whatever results page they were on. ``browseBack``
  // intentionally keeps results on the way down; Home is the explicit
  // "take me to the start" affordance, so it clears them.
  state.searchResults = null;
  state.searchQuery = '';
  _restoreLanding();
  showSearchView();
  if (dom.searchInput) dom.searchInput.value = '';
  if (dom.contentArea) dom.contentArea.scrollTop = 0;
  updateNavButtons();
}

async function restoreHistoryEntry() {
  const entry = state.history[state.historyIdx];
  if (!entry) return;

  // Knowledge-pack URLs go through the iframe path, not the web fetch.
  // Without this branch, back/forward to a previously-visited ZIM article
  // would call /api/browse/fetch?url=zim:... which the backend rejects
  // ("URL must start with http:// or https://"). browseFetch has the same
  // dispatch — restoring history needs to mirror it.
  if (entry.url && entry.url.startsWith('zim:')) {
    state.currentUrl = entry.url;
    _syncBrowseSurface(entry.url, entry.title);
    _renderZimArticle(entry.url, { skipHistoryPush: true });
    if (dom.contentArea && entry.scrollPos) {
      requestAnimationFrame(() => { dom.contentArea.scrollTop = entry.scrollPos; });
    }
    updateNavButtons();
    return;
  }

  // Coming back from a ZIM entry to a regular web page — clear the
  // ZIM-mode layout classes so the extracted-article view renders in
  // its centered 760px column instead of full-bleed flex.
  dom.contentArea?.classList.remove('is-zim');
  dom.readerView?.classList.remove('is-zim');

  state.currentUrl = entry.url;
  _syncBrowseSurface(entry.url, entry.title);
  if (dom.searchInput) dom.searchInput.value = entry.url;
  showReaderView();

  showArticleLoading();
  try {
    const resp = await fetch(`/api/browse/fetch?url=${encodeURIComponent(entry.url)}`);
    const data = await resp.json();
    if (resp.ok) {
      state.currentContent = data.text;
      state.currentHtml = data.html;
      state.currentVideos = data.videos || [];
      state.currentPageType = data.page_type || 'article';
      renderArticle(data);
      if (dom.readerAiBlocks) dom.readerAiBlocks.innerHTML = '';
      if (dom.contentArea && entry.scrollPos) {
        requestAnimationFrame(() => { dom.contentArea.scrollTop = entry.scrollPos; });
      }
    } else {
      showArticleError(data.error || 'Failed to load page');
    }
  } catch (err) {
    showArticleError(`Failed to load: ${err.message}`);
  }
  updateNavButtons();
}

function updateNavButtons() {
  if (dom.backBtn) dom.backBtn.disabled = state.historyIdx < 0 && !state.searchResults;
  if (dom.fwdBtn) dom.fwdBtn.disabled = state.historyIdx >= state.history.length - 1;
  // Home stays enabled whenever the panel is open. Two reasons:
  //   (a) Focused views (per-pack home, bookmarks gallery) overwrite
  //       searchView.innerHTML without pushing history, so a strict
  //       ``historyIdx < 0 && !searchResults`` disable would leave the
  //       button greyed out exactly when the user most needs it.
  //   (b) Clicking from the bare landing is a no-op visually but still
  //       clears the search input — a useful "reset" affordance.
  if (dom.homeBtn) dom.homeBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// AI Sidebar
// ---------------------------------------------------------------------------
const _BROWSE_ACTION_LABELS = {
  summarize: 'AI Summary', keypoints: 'AI Key Points', explain: 'AI Explanation',
  extract: 'AI Extracted Data', translate: 'AI Translation', ask: 'AI Answer',
};

async function browseAiAction(action, question) {
  if (!state.currentContent) {
    showToast('No page content to analyze', 'warning');
    return;
  }

  _emitSignal('ai_action', {
    source_url: state.currentUrl || '',
    source_title: document.querySelector('.browse-title')?.textContent || '',
    content_type: 'article',
    weight: 2.0,
    metadata: { action },
  });

  if (state.aiAbort) state.aiAbort.abort();
  state.aiAbort = new AbortController();

  const container = dom.readerAiBlocks;
  if (!container) return;

  // Create inline block
  const label = _BROWSE_ACTION_LABELS[action] || 'AI Result';
  const block = document.createElement('div');
  block.className = 'browse-ai-block browse-ai-block-streaming';
  block.innerHTML = `
    <div class="browse-ai-block-header">
      <span class="browse-ai-block-label">${escapeHtml(label)}</span>
      <div class="browse-ai-block-actions">
        <button class="browse-ai-block-btn" data-action="copy">Copy</button>
        <button class="browse-ai-block-btn remove" data-action="remove">&times;</button>
      </div>
    </div>
    <div class="browse-ai-block-content"></div>
  `;
  container.appendChild(block);

  // Scroll to block
  if (dom.contentArea) dom.contentArea.scrollTop = dom.contentArea.scrollHeight;

  const contentEl = block.querySelector('.browse-ai-block-content');
  const body = { action, content: state.currentContent, model: app.state.currentModel || '', page_type: state.currentPageType || 'article' };
  if (question) body.question = question;

  // If user clicked "Ask about this video", focus context on that video's transcript
  if (state._videoAskContext && action === 'ask') {
    const v = state._videoAskContext;
    body.video_context = `[Video: ${v.title || 'Untitled'}]\n${v.transcript}`;
    state._videoAskContext = null;  // one-shot
    // Reset placeholder
    if (dom.readerAskInput) dom.readerAskInput.placeholder = 'Ask about this page...';
  } else if (state.currentVideos?.length) {
    // Include all video transcripts for unified AI context
    const vCtx = state.currentVideos
      .filter(v => v.transcript)
      .map(v => `[Video: ${v.title || 'Untitled'}]\n${v.transcript}`)
      .join('\n\n');
    if (vCtx) body.video_context = vCtx;
  }

  let fullText = '';

  try {
    const resp = await fetch('/api/browse/ai', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: state.aiAbort.signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: 'AI analysis failed' }));
      block.classList.remove('browse-ai-block-streaming');
      contentEl.innerHTML = `<p style="color:var(--text-muted)">${escapeHtml(err.error)}</p>`;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Shared incremental renderer (chat/stream-render.js): coalesced to one
    // rAF per frame + stable/active split, so a fast stream no longer
    // re-parses the whole answer on every delta. ``compact`` uses the full
    // chat markdown (highlighting/tables/images) minus the chat-only code
    // toolbar — same look as the chat surface.
    const aiRender = makeStreamRenderer(contentEl, {
      compact: true,
      onFlush: () => { if (dom.contentArea) dom.contentArea.scrollTop = dom.contentArea.scrollHeight; },
    });

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') break;

        try {
          const data = JSON.parse(payload);
          if (data.error) {
            block.classList.remove('browse-ai-block-streaming');
            contentEl.innerHTML = `<p style="color:var(--text-muted)">${escapeHtml(data.error)}</p>`;
            return;
          }
          if (data.delta) {
            fullText += data.delta;
            aiRender.render(fullText);
          }
        } catch {
          // skip
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') { block.remove(); return; }
    contentEl.innerHTML = `<p style="color:var(--text-muted)">Error: ${escapeHtml(err.message)}</p>`;
  }

  block.classList.remove('browse-ai-block-streaming');
  if (fullText) {
    // Flatten the streaming split into one render (matches saved/restored
    // blocks) and highlight off the critical path.
    contentEl.innerHTML = renderMarkdown(fullText, { compact: true });
    highlightCodeDeferred(contentEl);
    block.dataset.markdown = fullText;
  }
}

// ---------------------------------------------------------------------------
// Save to Context (existing RAG)
// ---------------------------------------------------------------------------
async function browseSave() {
  if (!state.currentContent) {
    showToast('No content to save', 'warning');
    return;
  }

  const entry = state.history[state.historyIdx];

  try {
    const resp = await fetch('/api/browse/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: state.currentUrl || '',
        title: entry?.title || '',
        content: state.currentContent,
      }),
    });

    if (resp.ok) {
      showToast('Page saved to document context', 'success');
    } else {
      const err = await resp.json().catch(() => ({}));
      showToast(err.error || 'Failed to save', 'error');
    }
  } catch (err) {
    showToast(`Save failed: ${err.message}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// Discuss in Chat — attach page content and switch to chat
// ---------------------------------------------------------------------------

async function discussInChat() {
  if (!state.currentContent) {
    showToast('No page content to discuss', 'warning');
    return;
  }

  const entry = state.history[state.historyIdx];
  const title = entry?.title || 'Web Page';
  const url = state.currentUrl || '';

  // Save to document store (reuse the save-to-context logic)
  try {
    const resp = await fetch('/api/browse/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, title, content: state.currentContent }),
    });

    if (!resp.ok) {
      showToast('Failed to attach page', 'error');
      return;
    }

    const data = await resp.json();

    // Dispatch event for app.js to attach to chat input
    document.dispatchEvent(new CustomEvent('augmentum:web-to-chat', {
      detail: {
        docId: data.id,
        filename: data.filename || `${title}.txt`,
        chunkCount: data.chunk_count || 0,
        title,
        url,
      }
    }));

    _emitSignal('discuss', {
      source_url: state.currentUrl || '',
      source_title: document.querySelector('.browse-title')?.textContent || '',
      content_type: 'article',
      weight: 2.5,
    });

    // Close browse panel
    closeBrowsePanel();
    showToast('Page ready — ask your question', 'success');
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// HTML → Markdown for Notes (preserves images from articles)
// ---------------------------------------------------------------------------

/**
 * Convert article HTML to markdown-friendly text for the notes editor.
 * Extracts images as ![alt](url) and merges them into the plain text
 * at roughly the right positions.
 */
function _htmlToNoteMarkdown(html, plainText) {
  if (!html) return plainText;

  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');
  const parts = [];

  function walk(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent.trim();
      if (text) parts.push(text);
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;

    const tag = node.tagName.toLowerCase();

    // Images → markdown
    if (tag === 'img') {
      const src = node.getAttribute('src') || '';
      if (src && !src.startsWith('data:image/svg')) {
        const alt = node.getAttribute('alt') || '';
        // Convert proxied browse URLs to direct URLs for note portability
        const directSrc = src.replace(/^\/api\/browse\/image\?url=/, '');
        const decodedSrc = directSrc.startsWith('http') ? directSrc : decodeURIComponent(directSrc);
        parts.push(`\n![${alt.replace(/[[\]]/g, '')}](${decodedSrc})\n`);
      }
      return;
    }

    // Block elements → newlines
    if (['p', 'div', 'section', 'article'].includes(tag)) {
      parts.push('\n\n');
      for (const child of node.childNodes) walk(child);
      parts.push('\n');
      return;
    }

    // Headings → markdown headings
    const hMatch = tag.match(/^h([1-6])$/);
    if (hMatch) {
      const level = '#'.repeat(parseInt(hMatch[1]));
      parts.push(`\n\n${level} `);
      for (const child of node.childNodes) walk(child);
      parts.push('\n');
      return;
    }

    // Lists
    if (tag === 'li') {
      const parent = node.parentElement?.tagName?.toLowerCase();
      const prefix = parent === 'ol'
        ? `${Array.from(node.parentElement.children).indexOf(node) + 1}. `
        : '- ';
      parts.push(`\n${prefix}`);
      for (const child of node.childNodes) walk(child);
      return;
    }

    // Blockquote
    if (tag === 'blockquote') {
      parts.push('\n\n> ');
      for (const child of node.childNodes) walk(child);
      parts.push('\n');
      return;
    }

    // Code blocks
    if (tag === 'pre') {
      const code = node.textContent || '';
      parts.push(`\n\n\`\`\`\n${code.trim()}\n\`\`\`\n`);
      return;
    }

    // Inline formatting
    if (tag === 'strong' || tag === 'b') {
      parts.push('**');
      for (const child of node.childNodes) walk(child);
      parts.push('**');
      return;
    }
    if (tag === 'em' || tag === 'i') {
      parts.push('*');
      for (const child of node.childNodes) walk(child);
      parts.push('*');
      return;
    }

    // Links
    if (tag === 'a') {
      const href = node.getAttribute('href') || '';
      if (href && !href.startsWith('#') && !href.startsWith('javascript:')) {
        const linkUrl = href.replace(/^\/api\/browse\/fetch\?url=/, '');
        parts.push('[');
        for (const child of node.childNodes) walk(child);
        parts.push(`](${decodeURIComponent(linkUrl)})`);
        return;
      }
    }

    // Figure with figcaption
    if (tag === 'figure') {
      for (const child of node.childNodes) walk(child);
      return;
    }
    if (tag === 'figcaption') {
      parts.push('\n*');
      for (const child of node.childNodes) walk(child);
      parts.push('*\n');
      return;
    }

    // Skip style/script
    if (tag === 'style' || tag === 'script') return;

    // Default: recurse
    for (const child of node.childNodes) walk(child);
  }

  walk(doc.body);

  let md = parts.join('');
  // Clean up excessive newlines
  md = md.replace(/\n{3,}/g, '\n\n').trim();
  return md;
}

// ---------------------------------------------------------------------------
// Save article as Note
// ---------------------------------------------------------------------------
async function saveArticleAsNote() {
  if (!state.currentContent && !state.currentHtml) {
    showToast('No content to save', 'warning');
    return;
  }

  const entry = state.history[state.historyIdx];
  const title = entry?.title || 'Untitled';

  // Convert HTML to markdown-friendly content, preserving images
  let content = state.currentContent || '';
  if (state.currentHtml) {
    content = _htmlToNoteMarkdown(state.currentHtml, content);
  }

  try {
    const resp = await fetch('/api/browse/notes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        content,
        source_url: state.currentUrl || '',
        source_title: title,
        format: 'article',
        tags: [],
      }),
    });

    if (resp.ok) {
      const note = await resp.json();
      showToast('Saved as note', 'success');
      _emitSignal('note_save', {
        source_url: state.currentUrl || '',
        source_title: document.querySelector('.browse-title')?.textContent || '',
        content_type: 'article',
        weight: 3.0,
      });
      await loadNotes();
      switchTab('notes');
      openNote(note.id);
    } else {
      console.error('Save as note failed:', resp.status, resp.statusText);
      const err = await resp.json().catch(() => ({}));
      showToast(err.error || 'Failed to save note', 'error');
    }
  } catch (err) {
    showToast(`Save failed: ${err.message}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// Notes — CRUD
// ---------------------------------------------------------------------------
async function loadNotes() {
  // Show skeleton loaders while fetching
  const container = dom.notesItems;
  if (container) {
    container.innerHTML = Array.from({ length: 3 }, () =>
      `<div class="browse-note-skeleton">
        <div class="browse-note-skeleton-line"></div>
        <div class="browse-note-skeleton-line"></div>
        <div class="browse-note-skeleton-line"></div>
      </div>`
    ).join('');
  }

  try {
    const resp = await fetch('/api/browse/notes');
    if (resp.ok) {
      const data = await resp.json();
      state.notes = data.notes || [];
    } else {
      console.error('loadNotes failed:', resp.status, resp.statusText);
    }
  } catch (err) {
    console.error('loadNotes error:', err);
  }
  renderNotesList();

  // Warm the CM6 editor bundle while the user reads the list, so the first
  // note-open paints text immediately instead of paying the ~8-module ESM
  // import waterfall on the open path (the LCP cost on cold open). Idle +
  // fire-and-forget: harmless if it never gets opened, deduped by the
  // loadCM6 cache inside the editor module.
  if (!state._notesEditorWarmed) {
    state._notesEditorWarmed = true;
    const warm = () => { prefetchNotesEditor().catch(() => {}); };
    if (typeof window.requestIdleCallback === 'function') {
      window.requestIdleCallback(warm, { timeout: 2000 });
    } else {
      setTimeout(warm, 0);
    }
  }
}

function _bucketForDate(d) {
  if (!d) return 'Earlier';
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const t = new Date(d).getTime();
  if (Number.isNaN(t)) return 'Earlier';
  const diffDays = Math.floor((today - new Date(t).setHours(0,0,0,0)) / 86400000);
  if (diffDays <= 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return 'Previous 7 days';
  if (diffDays < 30) return 'Previous 30 days';
  return 'Earlier';
}

function _formatRelativeTime(d) {
  if (!d) return '';
  const t = new Date(d).getTime();
  if (Number.isNaN(t)) return '';
  const diffMs = Date.now() - t;
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function renderNotesList() {
  const container = dom.notesItems;
  if (!container) return;

  const filter = (state.notesFilter || '').toLowerCase();
  let filtered = filter
    ? state.notes.filter(n =>
        (n.title || '').toLowerCase().includes(filter) ||
        (n.preview || '').toLowerCase().includes(filter) ||
        (n.tags || []).some(t => t.toLowerCase().includes(filter))
      )
    : state.notes;
  // Provenance chip: narrow to companion-created notes. Same list,
  // same store — origin is just a column (migration 259).
  if (state.notesOriginCompanion) {
    filtered = filtered.filter(n => n.origin === 'companion');
  }

  if (!filtered.length) {
    if (filter) {
      container.innerHTML = `<div class="browse-notes-empty-state"><span class="browse-notes-empty-text">No matching notes</span></div>`;
    } else {
      container.innerHTML = `
        <div class="browse-notes-empty-state">
          <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round" class="browse-notes-empty-icon"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
          <div class="browse-notes-empty-title">A blank page</div>
          <div class="browse-notes-empty-sub">Notes you save while browsing land here. Or start a fresh one.</div>
          <button class="browse-notes-empty-cta" onclick="document.getElementById('browse-new-note-btn')?.click()">New note</button>
        </div>
      `;
    }
    return;
  }

  const renderItem = (n) => {
    const isActive = n.id === state.activeNoteId;
    const tags = (n.tags || []).slice(0, 3).map(t => `<span class="browse-note-item-tag">#${escapeHtml(t)}</span>`).join('');
    const time = _formatRelativeTime(n.updated_at);
    let source = '';
    if (n.source_url) {
      try {
        const hostname = new URL(n.source_url).hostname;
        source = `<span class="browse-note-item-source"><img src="/api/browse/image?url=${encodeURIComponent(`https://www.google.com/s2/favicons?domain=${hostname}&sz=12`)}" alt="" loading="lazy" onerror="this.style.display='none'">${escapeHtml(hostname)}</span>`;
      } catch { /* ignore */ }
    }
    // Provenance badge — visible even when the chip filter is off,
    // so companion-created notes read as such at a glance. Label is
    // persona-agnostic on purpose (OSS — companions vary).
    const originBadge = n.origin === 'companion'
      ? '<span class="browse-note-item-tag browse-note-origin-badge" title="Created by your companion">companion</span>'
      : '';
    const pinned = !!n.pinned;
    return `
      <div class="browse-note-item${isActive ? ' active' : ''}${pinned ? ' pinned' : ''}" data-id="${escapeHtml(n.id)}">
        <button class="browse-note-item-pin${pinned ? ' active' : ''}" data-pin-id="${escapeHtml(n.id)}" title="${pinned ? 'Unpin' : 'Pin to top'}" aria-label="${pinned ? 'Unpin note' : 'Pin note to top'}" aria-pressed="${pinned ? 'true' : 'false'}">
          <svg viewBox="0 0 24 24" width="13" height="13" fill="${pinned ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z"/></svg>
        </button>
        <button class="browse-note-item-delete" data-delete-id="${escapeHtml(n.id)}" title="Delete note">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
        <div class="browse-note-item-title">${escapeHtml(n.title || 'Untitled')}</div>
        <div class="browse-note-item-preview">${escapeHtml(n.preview || '')}</div>
        <div class="browse-note-item-meta">
          ${time ? `<span class="browse-note-item-time">${time}</span>` : ''}
          ${originBadge}
          ${source}
          ${tags}
        </div>
      </div>
    `;
  };

  // Pinned notes float to the top in their own group; the rest bucket by date.
  const pinnedNotes = filtered.filter(n => n.pinned);
  const rest = filtered.filter(n => !n.pinned);

  let html = '';
  if (pinnedNotes.length) {
    html += `<div class="browse-notes-group-label">Pinned</div>`;
    html += pinnedNotes.map(renderItem).join('');
  }

  // Group the rest by date bucket (preserving the existing updated_at DESC order)
  const buckets = new Map();
  for (const n of rest) {
    const b = _bucketForDate(n.updated_at);
    if (!buckets.has(b)) buckets.set(b, []);
    buckets.get(b).push(n);
  }

  // Render groups in fixed order
  const ORDER = ['Today', 'Yesterday', 'Previous 7 days', 'Previous 30 days', 'Earlier'];
  for (const groupName of ORDER) {
    const items = buckets.get(groupName);
    if (!items || !items.length) continue;
    html += `<div class="browse-notes-group-label">${groupName}</div>`;
    html += items.map(renderItem).join('');
  }
  container.innerHTML = html;

  // Wire note item clicks
  container.querySelectorAll('.browse-note-item').forEach(el => {
    el.addEventListener('click', (e) => {
      // Don't open note if clicking an item-action button (delete / pin)
      if (e.target.closest('.browse-note-item-delete')) return;
      if (e.target.closest('.browse-note-item-pin')) return;
      openNote(el.dataset.id);
    });
  });

  // Wire pin toggles
  container.querySelectorAll('.browse-note-item-pin').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      _toggleNotePin(btn.dataset.pinId);
    });
  });

  // Wire inline delete buttons
  container.querySelectorAll('.browse-note-item-delete').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = btn.dataset.deleteId;
      if (!id || !confirm('Delete this note?')) return;
      try {
        const resp = await fetch(`/api/browse/notes/${id}`, { method: 'DELETE' });
        if (resp.ok) {
          state.notes = state.notes.filter(n => n.id !== id);
          if (state.activeNoteId === id) closeNoteEditor();
          renderNotesList();
          showToast('Note deleted', 'success');
        }
      } catch {
        showToast('Failed to delete', 'error');
      }
    });
  });
}

// Toggle a note's pinned state. Optimistic: flip the stub + re-render
// immediately, persist via PUT, revert on failure. A pin-only PUT doesn't
// bump updated_at server-side, so the recents order is preserved.
async function _toggleNotePin(id) {
  if (!id) return;
  const note = state.notes.find(n => n.id === id);
  if (!note) return;
  const next = !note.pinned;
  note.pinned = next;
  renderNotesList();
  try {
    const resp = await fetch(`/api/browse/notes/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ pinned: next }),
    });
    if (!resp.ok) throw new Error(`pin failed (${resp.status})`);
  } catch (e) {
    note.pinned = !next;  // revert
    renderNotesList();
    showToast('Couldn’t update pin', 'error');
  }
}

async function openNote(noteId) {
  // Flush any pending save for previous note
  NoteEditor.flushSave();

  state.activeNoteId = noteId;

  // Show editor area
  dom.notesEmpty?.classList.add('hidden');
  dom.noteScroll?.classList.remove('hidden');
  // On mobile: hide list, show editor full-screen
  dom.bodyNotes?.classList.add('note-open');
  _applyNotesHistoryVisibility();
  // Clear previous AI blocks
  if (dom.noteAiBlocks) dom.noteAiBlocks.innerHTML = '';

  // Fetch full note
  try {
    const resp = await fetch(`/api/browse/notes/${noteId}`);
    if (!resp.ok) {
      showToast('Failed to load note', 'error');
      return;
    }
    state.activeNote = await resp.json();
  } catch {
    showToast('Failed to load note', 'error');
    return;
  }

  // Delegate to NoteEditor orchestrator (title, metadata, tags, editor loading)
  await NoteEditor.openNote(state.activeNote);

  // Restore persisted AI blocks
  _restoreAiBlocks(state.activeNote.ai_blocks);

  // Highlight in list
  renderNotesList();
}

async function createNewNote() {
  flushNoteSave();
  const defaultFormat = ['note', 'article', 'journal'].includes(getSettings().notesDefaultFormat)
    ? getSettings().notesDefaultFormat
    : 'note';

  try {
    const resp = await fetch('/api/browse/notes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'Untitled', content: '', tags: [], format: defaultFormat }),
    });

    if (resp.ok) {
      const note = await resp.json();
      state.notes.unshift({
        id: note.id,
        title: note.title,
        tags: note.tags,
        format: note.format || defaultFormat,
        source_url: '',
        source_title: '',
        created_at: note.created_at,
        updated_at: note.updated_at,
        preview: '',
      });
      renderNotesList();
      openNote(note.id);
      // Focus inline title for immediate editing
      if (dom.noteInlineTitle) {
        dom.noteInlineTitle.focus();
        // Select all text in contenteditable
        const range = document.createRange();
        range.selectNodeContents(dom.noteInlineTitle);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
    }
  } catch (err) {
    console.error('Failed to create note:', err);
    showToast('Failed to create note', 'error');
  }
}

async function deleteActiveNote() {
  if (!state.activeNoteId) return;
  if (!confirm('Delete this note? This cannot be undone.')) return;

  try {
    const resp = await fetch(`/api/browse/notes/${state.activeNoteId}`, { method: 'DELETE' });
    if (resp.ok) {
      state.notes = state.notes.filter(n => n.id !== state.activeNoteId);

      closeNoteEditor();
      showToast('Note deleted', 'success');
    }
  } catch {
    showToast('Failed to delete note', 'error');
  }
}

function closeNoteEditor() {
  // Delegate teardown to NoteEditor (flushes save, clears editor)
  NoteEditor.closeNote();

  state.activeNoteId = null;
  state.activeNote = null;
  state.milkdownEditor = null;

  // Reset editor area
  dom.notesEmpty?.classList.remove('hidden');
  dom.noteScroll?.classList.add('hidden');
  dom.noteMoreMenu?.classList.add('hidden');

  // On mobile: show list again
  dom.bodyNotes?.classList.remove('note-open');
  _applyNotesHistoryVisibility();

  renderNotesList();
}

function renderNoteTags() {
  if (!dom.noteTags || !state.activeNote) return;

  // Clear existing tags (keep input)
  dom.noteTags.querySelectorAll('.browse-note-tag').forEach(el => el.remove());

  // Insert tags before input
  const input = dom.noteTagInput;
  (state.activeNote.tags || []).forEach(tag => {
    const el = document.createElement('span');
    el.className = 'browse-note-tag';
    el.innerHTML = `${escapeHtml(tag)}<span class="browse-note-tag-remove" data-tag="${escapeHtml(tag)}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></span>`;
    el.querySelector('.browse-note-tag-remove')?.addEventListener('click', () => {
      state.activeNote.tags = state.activeNote.tags.filter(t => t !== tag);
      renderNoteTags();
      debounceNoteSave();
    });
    dom.noteTags.insertBefore(el, input);
  });
}

// ---------------------------------------------------------------------------
// Notes — Slash command: image generation
// ---------------------------------------------------------------------------
async function _slashGenerateImage(query) {
  let imgPrompt = (query || '').trim();
  if (!imgPrompt) {
    imgPrompt = (window.prompt('Image prompt:') || '').trim();
    if (!imgPrompt) return;
  }

  showToast(`Generating: "${imgPrompt.slice(0, 40)}${imgPrompt.length > 40 ? '\u2026' : ''}"`, 'info');
  try {
    const resp = await fetch('/api/image/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: imgPrompt }),
    });
    if (!resp.ok) {
      const txt = await resp.text().catch(() => '');
      showToast(`Image generation failed${txt ? ': ' + txt.slice(0, 80) : ''}`, 'error');
      return;
    }
    const data = await resp.json();
    if (!data.url) {
      showToast('Image generation returned no URL', 'error');
      return;
    }
    // Refocus the editor before inserting (await may have lost focus)
    const pm = dom.noteEditorBody?.querySelector('.ProseMirror');
    if (pm) pm.focus();
    SlashMenu.insertImageAtCaret({ url: data.url, alt: imgPrompt, prompt: imgPrompt });
    debounceNoteSave();
    showToast('Image added to note', 'success');
  } catch (err) {
    showToast('Image generation failed', 'error');
  }
}

// ---------------------------------------------------------------------------
// Notes — Save (debounced)
// ---------------------------------------------------------------------------
function debounceNoteSave() {
  // Delegate to NoteEditor's debounce (handles save status + timer)
  NoteEditor.debounceSave();
}

async function flushNoteSave() {
  // Delegate to NoteEditor
  await NoteEditor.flushSave();
}

// Callback for NoteEditor save — performs the actual API call + local stub update
async function _handleNoteEditorSave(noteId, payload) {
  // Collect AI blocks from DOM so they persist across sessions
  const aiBlocks = _collectAiBlocks();

  try {
    const r = await fetch(`/api/browse/notes/${noteId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, ai_blocks: aiBlocks }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
  } catch (err) {
    showToast('Note save failed: ' + (err?.message || err), 'error');
    console.warn('[browse] note save failed', err);
    return;
  }

  const { title, content, tags, format } = payload;

  // Update local stub
  const stub = state.notes.find(n => n.id === noteId);
  if (stub) {
    stub.title = title;
    stub.tags = tags || [];
    stub.format = format || stub.format || 'note';
    stub.preview = content.slice(0, 120);
  }
}

// More menu for whisper bar (delete, save to RAG, etc.)
function _showNoteMoreMenu(e) {
  e.stopPropagation();
  dom.noteMoreMenu?.classList.toggle('hidden');
}

function _hideNoteMoreMenu() {
  dom.noteMoreMenu?.classList.add('hidden');
}

function _wireNoteMoreMenu() {
  if (!dom.noteMoreMenu || dom.noteMoreMenu.dataset.wired) return;
  dom.noteMoreMenu.dataset.wired = '1';

  dom.noteMoreMenu.addEventListener('click', (e) => {
    const item = e.target.closest('.note-whisper-menu-item');
    if (!item) return;
    const action = item.dataset.action;
    _hideNoteMoreMenu();
    if (action === 'delete') deleteActiveNote();
    else if (action === 'rag') saveNoteToRag();
  });

  document.addEventListener('click', (e) => {
    if (dom.noteMoreMenu?.classList.contains('hidden')) return;
    if (e.target.closest('#note-whisper-menu') || e.target.closest('#note-whisper-more')) return;
    _hideNoteMoreMenu();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') _hideNoteMoreMenu();
  });
}

// ---------------------------------------------------------------------------
// Notes — Export / RAG
// ---------------------------------------------------------------------------
async function saveNoteToRag() {
  if (!state.activeNote) return;

  let content = state.activeNote.content || '';
  if (state.milkdownEditor) {
    try { content = getEditorMarkdown(); } catch {}
  }

  try {
    const resp = await fetch('/api/browse/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: state.activeNote.source_url || '',
        title: NoteEditor.getTitle() || state.activeNote.title || 'Note',
        content,
      }),
    });

    if (resp.ok) {
      showToast('Note saved to AI context', 'success');
    } else {
      showToast('Failed to save to context', 'error');
    }
  } catch {
    showToast('Failed to save to context', 'error');
  }
}

function downloadNote() {
  if (!state.activeNote) return;

  let content = state.activeNote.content || '';
  if (state.milkdownEditor) {
    try { content = getEditorMarkdown(); } catch {}
  }

  const title = NoteEditor.getTitle() || state.activeNote.title || 'note';
  const filename = `${title.replace(/[^a-zA-Z0-9-_ ]/g, '').trim() || 'note'}.md`;

  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
  showToast(`Downloaded ${filename}`, 'success');
}

// ---------------------------------------------------------------------------
// Notes AI — Block Persistence
// ---------------------------------------------------------------------------

/** Collect AI blocks from the DOM into a serializable array. */
function _collectAiBlocks() {
  const container = dom.noteAiBlocks;
  if (!container) return [];
  const blocks = [];
  for (const el of container.querySelectorAll('.browse-ai-block')) {
    const md = el.dataset.markdown;
    if (!md) continue; // skip still-streaming or error blocks
    blocks.push({
      label: el.querySelector('.browse-ai-block-label')?.textContent || 'AI Result',
      action: el.dataset.action || '',
      markdown: md,
    });
  }
  return blocks;
}

/** Restore persisted AI blocks into the DOM. */
function _restoreAiBlocks(aiBlocks) {
  const container = dom.noteAiBlocks;
  if (!container || !Array.isArray(aiBlocks)) return;
  container.innerHTML = '';

  for (const b of aiBlocks) {
    if (!b.markdown) continue;
    const isInsertable = ['expand', 'formalize', 'fix', 'extract_tasks', 'outline', 'translate'].includes(b.action);
    const block = document.createElement('div');
    block.className = 'browse-ai-block';
    block.dataset.markdown = b.markdown;
    block.dataset.action = b.action || '';
    block.innerHTML = `
      <div class="browse-ai-block-header">
        <span class="browse-ai-block-label">${escapeHtml(b.label)}</span>
        <div class="browse-ai-block-actions">
          ${isInsertable ? '<button class="browse-ai-block-btn primary" data-action="insert">Insert into note</button>' : ''}
          <button class="browse-ai-block-btn" data-action="copy">Copy</button>
          <button class="browse-ai-block-btn remove" data-action="remove">&times;</button>
        </div>
      </div>
      <div class="browse-ai-block-content">${renderMarkdown(b.markdown, { compact: true })}</div>
    `;
    container.appendChild(block);
    highlightCodeDeferred(block);
  }
}

// ---------------------------------------------------------------------------
// (Floating AI FAB removed on 2026-04-24 — replaced by the ask-bar
// tools button `#note-ai-tools-btn`. The old robot-head icon was
// pinned to #note-scroll and opened the non-existent #browse-ai-popover,
// so it was fully dormant.)

// ---------------------------------------------------------------------------
// Notes AI — Inline Blocks
// ---------------------------------------------------------------------------
let _noteAiAbort = null;
let _noteAiToolsPopover = null;

const _ACTION_LABELS = {
  summarize: 'AI Summary', keypoints: 'AI Key Points', expand: 'AI Expanded',
  formalize: 'AI Formalized', fix: 'AI Polished', explain: 'AI Explanation',
  extract_tasks: 'AI Tasks', outline: 'AI Outline', translate: 'AI Translation',
  ask: 'AI Answer',
};

// Groups for the ask-bar AI popover. Kept alongside _ACTION_LABELS so
// adding a new backend action is a one-line addition here + the label
// map above.
const _NOTE_AI_TOOL_GROUPS = [
  {
    label: 'Analyze',
    hint: 'Inline result, non-destructive',
    items: [
      { id: 'summarize', label: 'Summarize',   icon: '≡', hint: 'Condense to the essential' },
      { id: 'keypoints', label: 'Key points',  icon: '•', hint: 'Bullet-list the ideas' },
      { id: 'explain',   label: 'Explain',     icon: '?', hint: 'Break it down' },
    ],
  },
  {
    label: 'Rewrite',
    hint: 'Insertable result',
    items: [
      { id: 'expand',    label: 'Expand',    icon: '↗', hint: 'Flesh out the detail' },
      { id: 'fix',       label: 'Polish',    icon: '✶', hint: 'Grammar and clarity' },
      { id: 'formalize', label: 'Formalize', icon: 'Aa', hint: 'Shift to formal tone' },
    ],
  },
  {
    label: 'Extract',
    hint: 'Insertable result',
    items: [
      { id: 'outline',       label: 'Outline',   icon: '≣', hint: 'Structural skeleton' },
      { id: 'extract_tasks', label: 'Tasks',     icon: '☑', hint: 'Pull actionable items' },
      { id: 'translate',     label: 'Translate…', icon: '🌐', hint: 'Ask which language' },
    ],
  },
];

function _wireNoteAiToolsPopover() {
  const btn = dom.noteAiToolsBtn;
  if (!btn) return;

  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (_noteAiToolsPopover && !_noteAiToolsPopover.classList.contains('hidden')) {
      _closeNoteAiToolsPopover();
      return;
    }
    _openNoteAiToolsPopover();
  });

  // Close on outside click or Escape.
  document.addEventListener('mousedown', (e) => {
    if (!_noteAiToolsPopover || _noteAiToolsPopover.classList.contains('hidden')) return;
    if (e.target === btn || btn.contains(e.target)) return;
    if (_noteAiToolsPopover.contains(e.target)) return;
    _closeNoteAiToolsPopover();
  }, true);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _noteAiToolsPopover && !_noteAiToolsPopover.classList.contains('hidden')) {
      _closeNoteAiToolsPopover();
      btn.focus();
    }
  });
}

function _openNoteAiToolsPopover() {
  if (!_noteAiToolsPopover) {
    _noteAiToolsPopover = document.createElement('div');
    _noteAiToolsPopover.className = 'note-ai-tools-popover';
    _noteAiToolsPopover.setAttribute('role', 'menu');
    _noteAiToolsPopover.setAttribute('aria-label', 'AI tools');
    let html = '';
    for (const group of _NOTE_AI_TOOL_GROUPS) {
      html += `<div class="note-ai-tools-group">
        <span class="note-ai-tools-group-label">${escapeHtml(group.label)}</span>
        <span class="note-ai-tools-group-hint">${escapeHtml(group.hint)}</span>
      </div>`;
      for (const item of group.items) {
        html += `<button type="button" class="note-ai-tools-item" role="menuitem" data-action="${escapeHtml(item.id)}">
          <span class="note-ai-tools-icon">${escapeHtml(item.icon)}</span>
          <span class="note-ai-tools-label">${escapeHtml(item.label)}</span>
          <span class="note-ai-tools-hint">${escapeHtml(item.hint)}</span>
        </button>`;
      }
    }
    _noteAiToolsPopover.innerHTML = html;
    document.body.appendChild(_noteAiToolsPopover);
    _noteAiToolsPopover.addEventListener('click', (e) => {
      const item = e.target.closest('.note-ai-tools-item');
      if (!item) return;
      const action = item.dataset.action;
      _closeNoteAiToolsPopover();
      if (!action) return;
      if (action === 'translate') {
        const lang = prompt('Translate to which language?', 'Spanish');
        if (lang) noteAiAction('translate', lang);
        return;
      }
      noteAiAction(action);
    });
  }
  const btn = dom.noteAiToolsBtn;
  if (!btn) return;
  btn.setAttribute('aria-expanded', 'true');
  _noteAiToolsPopover.classList.remove('hidden');
  _positionNoteAiToolsPopover();
}

function _closeNoteAiToolsPopover() {
  if (!_noteAiToolsPopover) return;
  _noteAiToolsPopover.classList.add('hidden');
  dom.noteAiToolsBtn?.setAttribute('aria-expanded', 'false');
}

// Scroll-to-top / scroll-to-bottom jumper. Sits absolute in the
// notes-editor-area so it stays visually fixed as #note-scroll
// scrolls. Appears only when the container is actually scrollable
// (content taller than viewport + a 40px hysteresis). Each end
// dims when the scroll position is already there.
function _wireNoteScrollJumper() {
  const scroll = dom.noteScroll;
  if (!scroll) return;
  const host = scroll.parentElement;
  if (!host) return;
  if (host.querySelector('.note-scroll-jumper')) return;   // already mounted

  const pill = document.createElement('div');
  pill.className = 'note-scroll-jumper hidden';
  pill.setAttribute('role', 'group');
  pill.setAttribute('aria-label', 'Jump to top or bottom of note');
  pill.innerHTML = `
    <button type="button" class="note-scroll-jumper-btn" data-dir="top" title="Jump to top" aria-label="Jump to top">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <polyline points="18 15 12 9 6 15"/>
      </svg>
    </button>
    <button type="button" class="note-scroll-jumper-btn" data-dir="bottom" title="Jump to bottom (ask bar)" aria-label="Jump to bottom">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <polyline points="6 9 12 15 18 9"/>
      </svg>
    </button>
  `;
  host.appendChild(pill);

  const topBtn = pill.querySelector('[data-dir="top"]');
  const botBtn = pill.querySelector('[data-dir="bottom"]');

  pill.addEventListener('click', (e) => {
    const btn = e.target.closest('.note-scroll-jumper-btn');
    if (!btn) return;
    const dir = btn.dataset.dir;
    if (dir === 'top') {
      scroll.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      scroll.scrollTo({ top: scroll.scrollHeight, behavior: 'smooth' });
      // After the smooth scroll lands, give the ask input focus so
      // users can immediately type.
      setTimeout(() => dom.noteAskInput?.focus(), 320);
    }
  });

  let rafPending = false;
  const update = () => {
    rafPending = false;
    const scrollable = scroll.scrollHeight > scroll.clientHeight + 40;
    pill.classList.toggle('hidden', !scrollable);
    if (!scrollable) return;
    const atTop = scroll.scrollTop < 24;
    const atBottom = scroll.scrollTop >= scroll.scrollHeight - scroll.clientHeight - 24;
    topBtn.classList.toggle('at-end', atTop);
    botBtn.classList.toggle('at-end', atBottom);
  };
  const schedule = () => {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(update);
  };

  scroll.addEventListener('scroll', schedule, { passive: true });
  // Content changes (CM6 typing, AI blocks appended) change scrollHeight.
  try {
    const ro = new ResizeObserver(schedule);
    ro.observe(scroll);
    const inner = scroll.firstElementChild;
    if (inner) ro.observe(inner);
  } catch { /* ResizeObserver missing — no-op */ }
  // Initial measurement after layout settles.
  setTimeout(update, 80);
}

function _positionNoteAiToolsPopover() {
  if (!_noteAiToolsPopover) return;
  const btn = dom.noteAiToolsBtn;
  if (!btn) return;
  // Measure after unhiding so offsetWidth is accurate.
  const r = btn.getBoundingClientRect();
  const popW = _noteAiToolsPopover.offsetWidth || 280;
  const popH = _noteAiToolsPopover.offsetHeight || 320;
  // Prefer above the button; fall back to below on very tall popovers.
  const topAbove = r.top - popH - 8;
  const flipBelow = topAbove < 8;
  const top = flipBelow ? r.bottom + 8 : topAbove;
  const left = Math.max(8, Math.min(r.left, window.innerWidth - popW - 8));
  _noteAiToolsPopover.style.position = 'fixed';
  _noteAiToolsPopover.style.top = `${top}px`;
  _noteAiToolsPopover.style.left = `${left}px`;
}

async function noteAiAction(action, question) {
  const content = getEditorMarkdown();
  if (!content) {
    showToast('Note is empty — write something first', 'warning');
    return;
  }

  if (_noteAiAbort) _noteAiAbort.abort();
  _noteAiAbort = new AbortController();

  const container = dom.noteAiBlocks;
  if (!container) return;

  // Create the inline block element
  const blockId = 'aiblock-' + Date.now();
  const label = _ACTION_LABELS[action] || 'AI Result';
  const isInsertable = ['expand', 'formalize', 'fix', 'extract_tasks', 'outline', 'translate'].includes(action);

  const block = document.createElement('div');
  block.className = 'browse-ai-block browse-ai-block-streaming';
  block.id = blockId;
  block.dataset.action = action;
  block.innerHTML = `
    <div class="browse-ai-block-header">
      <span class="browse-ai-block-label">${escapeHtml(label)}</span>
      <div class="browse-ai-block-actions">
        ${isInsertable ? '<button class="browse-ai-block-btn primary" data-action="insert">Insert into note</button>' : ''}
        <button class="browse-ai-block-btn" data-action="copy">Copy</button>
        <button class="browse-ai-block-btn remove" data-action="remove">&times;</button>
      </div>
    </div>
    <div class="browse-ai-block-content"></div>
  `;
  container.appendChild(block);

  // Gently scroll the new block into view (not to the absolute bottom)
  block.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  const contentEl = block.querySelector('.browse-ai-block-content');

  const body = { action, content, model: app.state.currentModel || '' };
  if (question) body.question = question;

  let fullText = '';

  try {
    const resp = await fetch('/api/browse/ai', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: _noteAiAbort.signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: 'AI action failed' }));
      block.classList.remove('browse-ai-block-streaming');
      contentEl.innerHTML = `<p style="color:var(--text-muted)">${escapeHtml(err.error)}</p>`;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Shared incremental renderer — see browseAi() above. onFlush keeps the
    // block bottom in view (not the whole page) as it grows.
    const aiRender = makeStreamRenderer(contentEl, {
      compact: true,
      onFlush: () => {
        const blockBottom = block.getBoundingClientRect().bottom;
        const scrollRect = scrollEl?.getBoundingClientRect();
        if (scrollEl && scrollRect && blockBottom > scrollRect.bottom) {
          scrollEl.scrollTop += blockBottom - scrollRect.bottom + 16;
        }
      },
    });

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') break;

        try {
          const data = JSON.parse(payload);
          if (data.error) {
            block.classList.remove('browse-ai-block-streaming');
            contentEl.innerHTML = `<p style="color:var(--text-muted)">${escapeHtml(data.error)}</p>`;
            return;
          }
          if (data.delta) {
            fullText += data.delta;
            aiRender.render(fullText);
          }
        } catch {
          // skip malformed chunks
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') { block.remove(); return; }
    contentEl.innerHTML = `<p style="color:var(--text-muted)">Error: ${escapeHtml(err.message)}</p>`;
  }

  block.classList.remove('browse-ai-block-streaming');
  if (fullText) {
    contentEl.innerHTML = renderMarkdown(fullText, { compact: true });
    highlightCodeDeferred(contentEl);
    block.dataset.markdown = fullText;
    // Persist the AI block with the note
    debounceNoteSave();
  }
}

function insertAiBlockIntoNote(markdown) {
  if (!markdown || !state.activeNote) return;

  // Append to existing note content
  const current = getEditorMarkdown();
  const newContent = current ? current.trim() + '\n\n' + markdown : markdown;
  state.activeNote.content = newContent;

  // EasyMDE lets us set the value in place — no destroy/recreate needed.
  const editor = state.milkdownEditor;
  if (editor && typeof editor.value === 'function') {
    editor.value(newContent);
  } else if (editor?._textarea) {
    editor._textarea.value = newContent;
  } else if (typeof window.__loadNoteEditor === 'function') {
    window.__loadNoteEditor(dom.noteEditorBody, newContent, () => NoteEditor.debounceSave());
  }
  NoteEditor.debounceSave();
  showToast('Inserted into note', 'success');
}

// ---------------------------------------------------------------------------
// Notes TTS — Read Aloud
// ---------------------------------------------------------------------------
let _noteTtsAudio = null;

async function noteReadAloud(btn) {
  // If already playing, stop
  if (_noteTtsAudio) {
    _noteTtsAudio.pause();
    _noteTtsAudio = null;
    btn?.classList.remove('playing');
    return;
  }

  const content = getEditorMarkdown();
  if (!content) {
    showToast('Note is empty', 'warning');
    return;
  }

  // Clean text for TTS (strip markdown syntax)
  const cleanText = content
    .replace(/```[\s\S]*?```/g, '')       // code blocks
    .replace(/`[^`]+`/g, '')              // inline code
    .replace(/!\[.*?\]\(.*?\)/g, '')      // images
    .replace(/\[([^\]]+)\]\(.*?\)/g, '$1') // links → text
    .replace(/^#{1,6}\s+/gm, '')          // headings → text
    .replace(/[*_~]+/g, '')               // bold/italic/strike
    .replace(/^[-*+]\s+/gm, '')           // list markers
    .replace(/^\d+\.\s+/gm, '')           // numbered lists
    .replace(/^>\s+/gm, '')              // blockquotes
    .replace(/\|[^\n]+\|/g, '')           // tables
    .replace(/---+/g, '')                 // hr
    .replace(/\n{3,}/g, '\n\n')           // collapse whitespace
    .trim();

  if (!cleanText) {
    showToast('No readable text in note', 'warning');
    return;
  }

  btn?.classList.add('playing');

  try {
    const s = getSettings();
    const resp = await fetch('/v1/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input: cleanText.slice(0, 4000), // TTS length cap
        voice: s.voiceDefaultVoice || '',
        response_format: 'mp3',
        speed: s.voiceSpeed || 1.0,
      }),
    });

    if (!resp.ok) {
      showToast('TTS failed — check audio settings', 'error');
      btn?.classList.remove('playing');
      return;
    }

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);

    _noteTtsAudio = audio;

    audio.addEventListener('ended', () => {
      URL.revokeObjectURL(url);
      _noteTtsAudio = null;
      btn?.classList.remove('playing');
    });
    audio.addEventListener('error', () => {
      URL.revokeObjectURL(url);
      _noteTtsAudio = null;
      btn?.classList.remove('playing');
    });

    await audio.play();
  } catch (err) {
    showToast(`TTS error: ${err.message}`, 'error');
    btn?.classList.remove('playing');
    _noteTtsAudio = null;
  }
}

async function noteSaveAudio(btn) {
  const content = getEditorMarkdown();
  if (!content) {
    showToast('Note is empty', 'warning');
    return;
  }

  // Clean text for TTS
  const cleanText = content
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`[^`]+`/g, '')
    .replace(/!\[.*?\]\(.*?\)/g, '')
    .replace(/\[([^\]]+)\]\(.*?\)/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/[*_~]+/g, '')
    .replace(/^[-*+]\s+/gm, '')
    .replace(/^\d+\.\s+/gm, '')
    .replace(/^>\s+/gm, '')
    .replace(/\|[^\n]+\|/g, '')
    .replace(/---+/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  if (!cleanText) {
    showToast('No readable text in note', 'warning');
    return;
  }

  btn?.classList.add('playing');
  showToast('Generating audio... this may take a moment', 'loading', 0);

  // Split into chunks for long notes (TTS has limits)
  const chunks = [];
  const maxChunkLen = 3500;
  let remaining = cleanText;
  while (remaining.length > 0) {
    if (remaining.length <= maxChunkLen) {
      chunks.push(remaining);
      break;
    }
    // Try to split at paragraph, then sentence, then hard cut
    let cut = remaining.lastIndexOf('\n\n', maxChunkLen);
    if (cut < maxChunkLen / 2) cut = remaining.lastIndexOf('. ', maxChunkLen);
    if (cut < maxChunkLen / 2) cut = maxChunkLen;
    chunks.push(remaining.slice(0, cut + 1));
    remaining = remaining.slice(cut + 1).trim();
  }

  const s = getSettings();
  const audioBlobs = [];
  try {
    for (const chunk of chunks) {
      const resp = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: chunk,
          voice: s.voiceDefaultVoice || '',
          response_format: 'mp3',
          speed: s.voiceSpeed || 1.0,
        }),
      });

      if (!resp.ok) {
        showToast('TTS generation failed — check audio settings', 'error');
        btn?.classList.remove('playing');
        return;
      }

      audioBlobs.push(await resp.blob());
    }
  } catch (err) {
    showToast(`TTS error: ${err.message}`, 'error');
    btn?.classList.remove('playing');
    return;
  }

  // Combine blobs and download
  const combined = new Blob(audioBlobs, { type: 'audio/mpeg' });
  const title = NoteEditor.getTitle() || state.activeNote?.title || 'note';
  const filename = `${title.replace(/[^a-zA-Z0-9-_ ]/g, '').trim() || 'note'}.mp3`;

  const url = URL.createObjectURL(combined);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);

  btn?.classList.remove('playing');

  // Dismiss loading toast and show success
  document.querySelectorAll('.toast.loading').forEach(t => {
    t.querySelector('.toast-close')?.click();
  });
  showToast(`Saved ${filename} (${(combined.size / 1024).toFixed(0)} KB)`, 'success');
}

// ---------------------------------------------------------------------------
// Notes editor — CodeMirror 6 + Live Preview (ui/scripts/notes-editor.js).
// Swapped in 2026-04-23 to replace Milkdown Crepe → EasyMDE. The new
// editor treats the page as the writing surface: markdown renders
// inline (headings as headings, bold as bold, quotes as quotes) with
// formatting marks fading on non-caret lines. No visible toolbar.
// Function names keep the legacy `milkdown` prefix so state fields
// (state.milkdownEditor / milkdownLoaded / milkdownLoading) and the
// rest of the call graph don't need a sweeping rename — treat those
// names as "notes editor". Full details in notes-editor.js.
// ---------------------------------------------------------------------------

async function ensureMilkdownLoaded() {
  if (state.milkdownLoaded) return true;
  if (state.milkdownLoading) {
    while (state.milkdownLoading) await new Promise(r => setTimeout(r, 50));
    return state.milkdownLoaded;
  }
  state.milkdownLoading = true;
  try {
    await prefetchNotesEditor();
    state.milkdownLoaded = true;
  } catch (err) {
    console.error('Failed to load notes editor (CM6):', err);
    state.milkdownLoaded = false;
  }
  state.milkdownLoading = false;
  return state.milkdownLoaded;
}

async function loadMilkdownEditor(container, markdown, onChange) {
  // Legacy signature: loadMilkdownEditor(markdown)
  if (typeof container === 'string') {
    markdown = container;
    container = dom.noteEditorBody;
    onChange = null;
  }
  if (!container) return null;

  container.innerHTML = `
    <div class="browse-note-editor-loading" role="status" aria-live="polite">
      <div class="browse-note-editor-loading-art" aria-hidden="true">
        <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M10 38 L26 22 L38 10 C39.5 8.5 41.5 8.5 43 10 C44.5 11.5 44.5 13.5 43 15 L31 27 L15 43 L7 45 Z"/>
          <path d="M26 22 L31 27" opacity="0.5"/>
          <path d="M7 45 L15 43" opacity="0.35"/>
        </svg>
        <span class="browse-note-editor-loading-ink"></span>
      </div>
      <div class="browse-note-editor-loading-title">Preparing the page</div>
      <div class="browse-note-editor-loading-sub">Threading ink, smoothing paper&hellip;</div>
    </div>
  `;

  const loaded = await ensureMilkdownLoaded();
  if (!loaded) {
    return createFallbackEditor(container, markdown, onChange);
  }

  // Tear down any previous editor on this page.
  const prev = window.__activeCrepeInstance || state.milkdownEditor;
  if (prev) {
    try {
      if (typeof prev.destroy === 'function') prev.destroy();
      else if (typeof prev.toTextArea === 'function') prev.toTextArea();
    } catch { /* ignore */ }
    window.__activeCrepeInstance = null;
    state.milkdownEditor = null;
    await new Promise(r => setTimeout(r, 0));
  }

  try {
    const editor = await createNotesEditor({
      element: container,
      value: markdown || '',
      onChange: onChange || debounceNoteSave,
      handlers: {
        openSlashMenu: (view) => {
          // Future wiring: bridge to SlashMenu.open(view) when ported.
        },
        askAi: (selectedText) => {
          // Dispatch the same event the mobile toolbar fires so both
          // surfaces share a single AI flow: selection becomes a
          // context chip above the ask bar; user types their actual
          // question into the bar. Avoids the scope split between
          // loadMilkdownEditor (here) and initializeBrowsePanel
          // (where the chip helpers live).
          document.dispatchEvent(new CustomEvent('note-ai-action', {
            detail: { action: 'ai-menu', selectedText: selectedText || '' },
          }));
        },
      },
    });

    state.milkdownEditor = editor;
    return editor;
  } catch (err) {
    console.error('Notes editor init failed:', err);
    return createFallbackEditor(container, markdown, onChange);
  }
}

function createFallbackEditor(container, markdown, onChange) {
  container.innerHTML = '';
  const textarea = document.createElement('textarea');
  textarea.style.cssText = `
    width: 100%; min-height: calc(100vh - 280px); flex: 1;
    border: none; background: transparent;
    color: var(--text-primary);
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 1.0625rem; line-height: 1.85; letter-spacing: 0.006em;
    resize: none; outline: none; padding: 0;
    caret-color: var(--accent);
  `;
  textarea.placeholder = 'Start writing...';
  textarea.value = markdown || '';
  textarea.addEventListener('input', onChange || debounceNoteSave);
  container.appendChild(textarea);
  const editor = { _textarea: textarea };
  state.milkdownEditor = editor;
  return editor;
}

function getEditorMarkdown() {
  if (!state.milkdownEditor) return state.activeNote?.content || '';

  // Textarea fallback (when EasyMDE failed to load)
  if (state.milkdownEditor._textarea) {
    return state.milkdownEditor._textarea.value;
  }

  // CM6 notes editor exposes .value() / .getMarkdown() (legacy alias).
  try {
    if (typeof state.milkdownEditor.value === 'function') {
      return state.milkdownEditor.value();
    }
    if (typeof state.milkdownEditor.getMarkdown === 'function') {
      return state.milkdownEditor.getMarkdown();
    }
    // Raw EditorView escape hatch
    if (state.milkdownEditor.codemirror?.state?.doc?.toString) {
      return state.milkdownEditor.codemirror.state.doc.toString();
    }
  } catch { /* ignore */ }

  // DOM fallback — CM6 content container, then legacy editors.
  const cmContent = dom.noteEditorBody?.querySelector('.cm-content');
  if (cmContent) return cmContent.textContent || '';
  const ta = dom.noteEditorBody?.querySelector('textarea');
  if (ta && typeof ta.value === 'string') return ta.value;
  const editorEl = dom.noteEditorBody?.querySelector('.editor, .ProseMirror, [contenteditable]');
  return editorEl?.textContent || state.activeNote?.content || '';
}
