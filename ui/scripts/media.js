/**
 * media.js — in-app Media console.
 *
 * Browse + detail + dispatch shell over the existing files surface. The
 * heavy lifting (floating video player with transcode-aware seek +
 * multi-audio + multi-sub + speed + A/V sync, music_video → Grove
 * handoff, comic series-first reader, gallery with kbd nav, read-along
 * Gutenberg sync, armed-device cast intercept) lives in
 * ui/scripts/files/ and is reached via `activateFile(file)` — Media
 * provides the comfort-canon browse/present layer and delegates every
 * leaf playback to the canonical surface.
 *
 * Views (stack-based; Esc/Back walk up):
 *   home    — rails over /api/cast/library/home (shared renderer)
 *   detail  — shared consumption/detail.js over /api/media/details
 *   section — "See all" paginated grid over /api/cast/library/section
 *   search  — cross-library results over /api/cast/library/search
 *   series  — comic chapter drill-in (/api/cast/library/chapters)
 *
 * Dispatch:
 *   browse_series + comic  → in-Media chapters drill-in
 *   browse_series + video  → detail view (episodes live inside it)
 *   Continue-rail tile     → direct play (resume is the point)
 *   image / comic leaf /
 *   music_video            → direct play (gallery / reader / Grove)
 *   movie / episode / audio→ detail view, play from there
 *   secondary (right-click)→ openCastPicker  [trusted-receiver cast]
 *
 * Comfort canon: uniform tiles, under-tile titles, 2px progress, no
 * autoplay hero, quiet at idle. Inherits global theme tokens.
 *
 * Hidden client-side (Media-only, doesn't touch cast-control):
 *   music_videos (Song↔Video toggle in files preview), gallery (image
 *   library), games (Library workspace).
 */

import { renderRails, wireChevrons } from './consumption/rails.js';
import { renderDetail } from './consumption/detail.js';
import { renderSectionGrid } from './consumption/grid.js';
import { fetchHome, clearHomeCache } from './consumption/library-client.js';
import { escapeHtml } from './app.js';
import { registerExtraFileSource } from './files/preview.js';
import { MEDIA_SOURCES } from './files/state.js';

const HIDDEN_RAILS = ['music_videos', 'gallery', 'games'];

// Media's content scope (Matt, 2026-07-25): Media is about WATCH / LISTEN
// (from a media server) — NOT a file browser. Images, loose documents
// (pdf/epub/doc), and LOCAL audio all belong to Files + Library, not here.
// This is the single choke-point that enforces it, so the policy holds
// even if the backend library feed changes what it emits.
//
//   keep  video           — movies / episodes / series / music_video
//   keep  comic           — comic series + chapters
//   keep  audio  IFF media-server-sourced — audiobooks / podcasts; a
//                           bare {kind:'audio'} rail query returns local
//                           uploads too, so gate on the same MEDIA_SOURCES
//                           slug set Files uses (isMediaServerFile).
//   drop  image, pdf/doc/other, and local audio.
function _tileInMediaScope(item) {
  const kind = (item?.kind || '').toLowerCase();
  if (kind === 'video' || kind === 'comic') return true;
  if (kind === 'audio') return MEDIA_SOURCES.has(item?.source || '');
  return false;
}

// Apply _tileInMediaScope to every section. A section emptied by the
// scrub drops out entirely; sections without an items array pass through.
function _scrubToMediaScope(sections) {
  if (!Array.isArray(sections)) return [];
  return sections.reduce((out, section) => {
    const items = section?.items;
    if (!Array.isArray(items)) { out.push(section); return out; }
    const kept = items.filter(_tileInMediaScope);
    if (kept.length === items.length) out.push(section);
    else if (kept.length) out.push({ ...section, items: kept });
    // else: section had nothing in scope → drop it
    return out;
  }, []);
}

// Media-only rails — not in the shared cast RAIL_CATALOG (which drives the
// cast RECEIVER prefs too; a slug there with no cast-home section recreates
// the "dead toggle with no backing" ghost the catalog docstring warns about).
// Live TV is client-injected in Media only: its channels are a different
// domain object (no file_id — server_id/channel_id, logo, EPG) that plays via
// files/live-tv-rails.js::playLiveTvChannel. It still participates fully in the
// user's show/hide + order prefs because those key off bare slugs.
const MEDIA_EXTRA_RAILS = [
  { slug: 'live_tv', title: 'Live TV', hint: 'Live channels from your media server' },
];

// User rail prefs (per-user, server-persisted in the ui config): which rails
// are hidden (ui.mediaRailsHidden) and their display order (ui.mediaRailsOrder).
// Both stack on top of the Media-policy HIDDEN_RAILS above. One cached GET
// backs both; savers patch the cache + PUT the single changed key.
let _railPrefs = null;   // null = not loaded yet; else { hidden: [], order: [] }

function _parseSlugArray(raw) {
  try {
    const arr = JSON.parse(raw || '[]');
    return Array.isArray(arr) ? arr.filter((s) => typeof s === 'string') : [];
  } catch { return []; }
}

async function _loadRailPrefs() {
  if (_railPrefs !== null) return _railPrefs;
  let hidden = [];
  let order = [];
  try {
    const resp = await fetch('/api/config/ui', { credentials: 'same-origin' });
    if (resp.ok) {
      const body = await resp.json();
      hidden = _parseSlugArray(body.mediaRailsHidden);
      order = _parseSlugArray(body.mediaRailsOrder);
    }
  } catch (err) {
    console.warn('[media] rail prefs load failed:', err);
  }
  _railPrefs = { hidden, order };
  return _railPrefs;
}

async function _putUiKey(key, value) {
  try {
    const resp = await fetch('/api/config/ui', {
      method: 'PUT',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: JSON.stringify(value) }),
    });
    if (!resp.ok) console.warn(`[media] ${key} save failed: HTTP`, resp.status);
  } catch (err) {
    console.warn(`[media] ${key} save failed:`, err);
  }
}

async function _saveRailPrefs(hidden) {
  _railPrefs = { ...(_railPrefs || { order: [] }), hidden };
  await _putUiKey('mediaRailsHidden', hidden);
}

async function _saveRailOrder(order) {
  _railPrefs = { ...(_railPrefs || { hidden: [] }), order };
  await _putUiKey('mediaRailsOrder', order);
}

// Sort a slug-keyed list by the user's saved order. Slugs present in `order`
// come first in that order; anything absent (a newly-added rail the user
// hasn't positioned — e.g. live_tv on first run, or a future catalog rail)
// keeps its original relative order and lands AFTER the ordered ones, so it's
// visible and reorderable rather than silently dropped. Stale slugs in `order`
// (retired rails) simply match nothing. `keyOf` maps a list entry → slug.
function _sortByRailOrder(list, order, keyOf) {
  const rank = new Map(order.map((slug, i) => [slug, i]));
  return list
    .map((entry, i) => ({ entry, i, r: rank.has(keyOf(entry)) ? rank.get(keyOf(entry)) : Infinity }))
    .sort((a, b) => (a.r - b.r) || (a.i - b.i))
    .map((x) => x.entry);
}

// Cache of entries fetched here so files/preview.js's _resolvePreviewFile
// can find them when activateFile() hands off by id. Without this,
// openMediaPreview re-resolves through state.files (the Files grid,
// empty when Files isn't open) and silently no-ops. Same pattern as
// files/continue-rail.js — Media is the third surface that fetches
// rail entries outside the Files grid lifecycle.
const _recentFiles = new Map();   // file_id -> file_entry
const _RECENT_FILES_CAP = 64;
registerExtraFileSource(() => Array.from(_recentFiles.values()));
function _rememberFile(file) {
  if (!file?.id) return;
  // Re-insert to bump LRU order (Map iteration is insertion-order).
  _recentFiles.delete(file.id);
  _recentFiles.set(file.id, file);
  while (_recentFiles.size > _RECENT_FILES_CAP) {
    const oldest = _recentFiles.keys().next().value;
    _recentFiles.delete(oldest);
  }
}

async function _fetchEntry(fileId) {
  const api = await import('./files/api.js');
  const file = await api.fetchFileEntry(fileId);
  if (file) _rememberFile(file);
  return file;
}

let _overlay = null;
let _body = null;
let _searchInput = null;
let _initialized = false;
let _opened = false;
let _searchTimer = 0;

// View stack for in-Media navigation. Top of stack is what's rendered.
// Frames: { kind: 'home' }
//         { kind: 'series',  item }          — comic chapter drill-in
//         { kind: 'detail',  item }          — tile-shaped seed
//         { kind: 'section', section }       — see-all grid
//         { kind: 'search',  query }
// All frames carry scrollTop for restore-on-back.
const _viewStack = [];


/* ── DOM scaffolding ───────────────────────────────────────────── */

function _ensureDom() {
  if (_overlay) return;
  _overlay = document.createElement('div');
  _overlay.id = 'media-overlay';
  _overlay.className = 'media-overlay hidden';
  _overlay.setAttribute('role', 'dialog');
  _overlay.setAttribute('aria-modal', 'true');
  _overlay.setAttribute('aria-label', 'Media');
  // tabindex=-1 makes openMedia()'s _overlay.focus() actually move focus.
  _overlay.setAttribute('tabindex', '-1');
  _overlay.innerHTML = `
    <header class="media-header">
      <button class="media-back hidden" id="media-back-btn" type="button" aria-label="Back" title="Back">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="15 18 9 12 15 6"/>
        </svg>
        <span>Back</span>
      </button>
      <h1 class="media-title" id="media-title">Media</h1>
      <div class="media-search" role="search">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        <input type="search" id="media-search-input" placeholder="Search your library…"
               aria-label="Search your library" autocomplete="off" spellcheck="false">
      </div>
      <button class="media-refresh" id="media-rails-btn" type="button" aria-label="Customize rails" title="Customize rails">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <line x1="4" y1="7" x2="20" y2="7"/><circle cx="9" cy="7" r="2.2"/>
          <line x1="4" y1="17" x2="20" y2="17"/><circle cx="15" cy="17" r="2.2"/>
        </svg>
      </button>
      <button class="media-refresh" id="media-refresh-btn" type="button" aria-label="Refresh" title="Refresh">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="23 4 23 10 17 10"/>
          <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
        </svg>
      </button>
      <button class="media-devices" id="media-devices-btn" type="button" aria-label="Connected devices" title="Connected devices">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <rect x="3" y="4" width="13" height="16" rx="2"/>
          <path d="M8 18h3"/>
          <path d="M19 8v8"/>
          <path d="M22 10v4"/>
        </svg>
        <span>Devices</span>
      </button>
      <button class="media-close" id="media-close-btn" type="button" aria-label="Close media (Esc)" title="Close (Esc)">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
        <span>Close</span>
      </button>
    </header>
    <main class="media-body" id="media-body"></main>
  `;
  document.body.appendChild(_overlay);
  _body = _overlay.querySelector('#media-body');
  _searchInput = _overlay.querySelector('#media-search-input');

  _overlay.querySelector('#media-close-btn').addEventListener('click', closeMedia);
  _overlay.querySelector('#media-back-btn').addEventListener('click', _popView);
  _overlay.querySelector('#media-refresh-btn').addEventListener('click', () => {
    clearHomeCache();
    _clearLiveTvCache();
    _renderTop();
  });
  _overlay.querySelector('#media-rails-btn').addEventListener('click', (ev) => {
    _openRailPrefs(ev.currentTarget);
  });
  _overlay.querySelector('#media-devices-btn').addEventListener('click', async () => {
    const mod = await import('./media-servers.js').catch((err) => {
      console.warn('[media] connected devices failed:', err);
      return null;
    });
    await mod?.openMediaServers?.();
  });

  // Search-as-you-type. First keystroke pushes a search frame; further
  // keystrokes update it in place (no stack spam); clearing the box
  // pops back to wherever the user was.
  _searchInput.addEventListener('input', () => {
    clearTimeout(_searchTimer);
    _searchTimer = setTimeout(() => {
      const q = _searchInput.value.trim();
      const top = _viewStack[_viewStack.length - 1];
      if (!q) {
        if (top?.kind === 'search') _popView();
        return;
      }
      if (top?.kind === 'search') {
        top.query = q;
        top.scrollTop = 0;
        _renderTop();
      } else {
        _pushView({ kind: 'search', query: q, scrollTop: 0 });
      }
    }, 300);
  });
  _searchInput.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape') return;
    ev.stopPropagation();
    ev.preventDefault();
    _searchInput.value = '';
    const top = _viewStack[_viewStack.length - 1];
    if (top?.kind === 'search') _popView();
    _searchInput.blur();
  });

  // Document-level Esc handler walks the in-Media stack only —
  // delegated players (files preview / comic reader / Grove) own
  // their own overlays and Esc handling. When one of those is on top
  // it handles Esc first and we never fire here.
  document.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape' || !_opened) return;
    if (_viewStack.length > 1) {
      ev.preventDefault();
      _popView();
    } else {
      ev.preventDefault();
      closeMedia();
    }
  });
}


/* ── View navigation ───────────────────────────────────────────── */

function _pushView(view) {
  if (_viewStack.length) {
    _viewStack[_viewStack.length - 1].scrollTop = _body.scrollTop;
  }
  _viewStack.push(view);
  _renderTop();
}

function _popView() {
  if (_viewStack.length <= 1) return;
  const popped = _viewStack.pop();
  // Leaving search by Back keeps the input in sync with the stack.
  if (popped.kind === 'search' && _searchInput) _searchInput.value = '';
  _renderTop();
}

function _setView(view) {
  _viewStack.length = 0;
  _viewStack.push(view);
  _renderTop();
}

function _renderTop() {
  const top = _viewStack[_viewStack.length - 1];
  const backBtn = _overlay.querySelector('#media-back-btn');
  const titleEl = _overlay.querySelector('#media-title');
  backBtn.classList.toggle('hidden', _viewStack.length <= 1);
  if (top.kind === 'home') {
    titleEl.textContent = 'Media';
    _renderHome();
  } else if (top.kind === 'series') {
    titleEl.textContent = top.item.title || 'Series';
    _renderSeries(top);
  } else if (top.kind === 'detail') {
    titleEl.textContent = top.item.title || 'Details';
    _renderDetailView(top);
  } else if (top.kind === 'section') {
    titleEl.textContent = top.section.title || 'Browse';
    _renderSectionView(top);
  } else if (top.kind === 'search') {
    titleEl.textContent = 'Search';
    _renderSearch(top);
  } else if (top.kind === 'livetv') {
    titleEl.textContent = 'Live TV';
    _renderLiveTvView();
  }
  // Restore scroll on next frame (after innerHTML lands).
  requestAnimationFrame(() => {
    _body.scrollTop = top.scrollTop || 0;
  });
}


/* ── Home view ─────────────────────────────────────────────────── */

async function _renderHome() {
  _body.innerHTML = `
    <div class="media-rails" id="media-rails"></div>
    <div class="media-libraries hidden" id="media-libraries"></div>
    <div class="media-empty hidden" id="media-empty">Nothing here yet.</div>
    <div class="media-error hidden" id="media-error"></div>
  `;
  const rails = _body.querySelector('#media-rails');
  const libsEl = _body.querySelector('#media-libraries');
  const empty = _body.querySelector('#media-empty');
  const error = _body.querySelector('#media-error');
  try {
    const [payload, prefs] = await Promise.all([
      fetchHome({}),
      _loadRailPrefs(),
    ]);
    const userHidden = prefs.hidden;
    const order = prefs.order;
    const hiddenSet = new Set([...HIDDEN_RAILS, ...userHidden]);
    // Standard rails, scrubbed to Media scope and sorted by the user's
    // chosen order (unordered rails keep catalog order at the tail).
    const sections = _sortByRailOrder(
      _scrubToMediaScope(payload?.sections), order, (s) => s.id,
    );
    renderRails(rails, sections, {
      onTileActivate: _onTileActivate,
      onTileSecondary: _onTileSecondary,
      onSeeAll: (section) => _pushView({
        kind: 'section',
        section: { id: section.id, title: section.title },
        scrollTop: 0,
      }),
      hiddenSlugs: [...hiddenSet],
    });
    // Live TV is a Media-only, client-injected rail. Slot it into the
    // rendered rail order (or drop it silently when hidden / no channels).
    if (!hiddenSet.has('live_tv')) {
      await _injectLiveTvRail(rails, order);
    }
    _renderLibrariesRow(libsEl, payload?.libraries);
    if (!rails.children.length) {
      empty.textContent = userHidden.length
        ? 'Nothing here — every rail is hidden. Use the rails button up top to bring some back.'
        : 'Nothing here yet.';
      empty.classList.remove('hidden');
    }
  } catch (e) {
    error.textContent = `Couldn't load media: ${e.message}`;
    error.classList.remove('hidden');
  }
}

/* ── Live TV (Media-only client-injected rail) ─────────────────── */

// Fetch the live-TV rails payload. Same endpoint the Files Live TV chip
// uses; hitting Emby/JF, so cache within the session (home re-renders on
// every reopen) and clear it on the Refresh button. Returns [rails...] or
// null (Live TV is optional — no server means the rail just doesn't show).
let _liveTvRailsCache;   // undefined = unfetched; null = fetched-but-none; []|[...] = value
async function _fetchLiveTvRails() {
  if (_liveTvRailsCache !== undefined) return _liveTvRailsCache;
  try {
    const r = await fetch('/api/livetv/rails', { credentials: 'same-origin' });
    if (!r.ok) { _liveTvRailsCache = null; return null; }
    const body = await r.json();
    const railsArr = Array.isArray(body?.rails) ? body.rails : [];
    _liveTvRailsCache = railsArr.length ? railsArr : null;
  } catch (err) {
    console.warn('[media] live TV rails fetch failed:', err);
    _liveTvRailsCache = null;
  }
  return _liveTvRailsCache;
}

function _clearLiveTvCache() { _liveTvRailsCache = undefined; }

// Flatten every channel across the backend's rails, de-duped by
// server+channel, so the single Media "Live TV" rail is a browsable strip.
function _flattenLiveChannels(railsArr) {
  const seen = new Set();
  const out = [];
  for (const rail of railsArr) {
    for (const ch of (rail?.channels || [])) {
      const key = `${ch.server_id || ''}:${ch.external_id || ''}`;
      if (!ch.external_id || !ch.server_id || seen.has(key)) continue;
      seen.add(key);
      out.push(ch);
    }
  }
  return out;
}

async function _injectLiveTvRail(container, order) {
  const railsArr = await _fetchLiveTvRails();
  if (!railsArr) return;
  const channels = _flattenLiveChannels(railsArr);
  if (!channels.length) return;

  const railEl = _buildLiveTvRailEl(channels, {
    onSeeAll: () => _pushView({ kind: 'livetv', scrollTop: 0 }),
  });

  // Slot into the ordered rails already in the DOM: insert before the
  // first rendered rail whose order-rank exceeds live_tv's. Unordered
  // (rank Infinity, first-run) → appended last. Mirrors _sortByRailOrder.
  const rank = new Map(order.map((slug, i) => [slug, i]));
  const liveRank = rank.has('live_tv') ? rank.get('live_tv') : Infinity;
  let before = null;
  for (const child of container.querySelectorAll(':scope > .media-rail')) {
    const slug = child.dataset.slug || '';
    const childRank = rank.has(slug) ? rank.get(slug) : Infinity;
    if (childRank > liveRank) { before = child; break; }
  }
  container.insertBefore(railEl, before);
}

// Build a Media-canon rail element for Live TV. Reuses the shared rail
// chrome classes + chevron paging so it's visually identical to sibling
// rails; only the tiles differ (channels, not file tiles).
function _buildLiveTvRailEl(channels, { onSeeAll } = {}, { cap = 24 } = {}) {
  const rail = document.createElement('section');
  rail.className = 'media-rail';
  rail.dataset.slug = 'live_tv';

  const header = document.createElement('header');
  header.className = 'media-rail-header';
  header.innerHTML = `
    <div class="media-rail-heading">
      <h3 class="media-rail-title">Live TV</h3>
    </div>
    ${onSeeAll && channels.length > cap
      ? `<button class="media-rail-seeall" type="button" aria-label="See all Live TV channels">
           <span>See all</span>
           <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="9 18 15 12 9 6"/></svg>
         </button>`
      : ''}
  `;
  const seeAllBtn = header.querySelector('.media-rail-seeall');
  if (seeAllBtn && onSeeAll) seeAllBtn.addEventListener('click', onSeeAll);
  rail.appendChild(header);

  const scroller = document.createElement('div');
  scroller.className = 'media-rail-scroller';
  const strip = document.createElement('div');
  strip.className = 'media-rail-strip';
  for (const ch of channels.slice(0, cap)) {
    strip.appendChild(_renderLiveChannelTile(ch));
  }
  scroller.appendChild(strip);
  wireChevrons(scroller, strip);
  rail.appendChild(scroller);
  return rail;
}

function _liveLogoUrl(ch) {
  const theme = (document.documentElement.getAttribute('data-theme') || 'light').toLowerCase();
  const preferred = theme === 'dark' ? 'light' : 'dark';
  const has = { primary: ch.has_logo_primary, light: ch.has_logo_light, dark: ch.has_logo_dark };
  const variant = has[preferred] ? preferred : (has.primary ? 'primary' : '');
  if (!variant || !ch.server_id || !ch.external_id) return '';
  return `/api/livetv/logo/${encodeURIComponent(ch.server_id)}/`
    + `${encodeURIComponent(ch.external_id)}?variant=${encodeURIComponent(variant)}`;
}

function _liveInitials(name) {
  const cleaned = String(name || '').replace(/[^a-zA-Z0-9 ]/g, ' ').trim();
  if (!cleaned) return '#';
  const parts = cleaned.split(/\s+/);
  return (parts.length === 1
    ? parts[0].slice(0, 2)
    : parts[0][0] + parts[1][0]).toUpperCase();
}

function _renderLiveChannelTile(ch) {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'media-tile media-live-tile';
  el.dataset.channelId = ch.external_id || '';
  el.dataset.serverId = ch.server_id || '';
  const name = ch.name || 'Channel';
  const num = (ch.channel_number || '').trim();
  const now = ch.current_program?.name ? String(ch.current_program.name) : '';
  const logo = _liveLogoUrl(ch);
  el.title = now ? `${name} — ${now}` : name;
  el.innerHTML = `
    <div class="media-live-logo-frame">
      ${logo
        ? `<img class="media-live-logo" src="${escapeHtml(logo)}" alt="${escapeHtml(name)}" loading="lazy" onerror="this.classList.add('media-live-logo-missing')">`
        : `<div class="media-live-logo media-live-logo-fallback">${escapeHtml(_liveInitials(name))}</div>`}
      ${num ? `<span class="media-live-number">${escapeHtml(num)}</span>` : ''}
    </div>
    <div class="media-tile-title">${escapeHtml(name)}</div>
    <div class="media-live-now${now ? '' : ' media-live-now-empty'}">
      <span class="media-live-now-dot" aria-hidden="true"></span>
      <span class="media-live-now-text">${now ? escapeHtml(now) : 'Live'}</span>
    </div>
  `;
  el.addEventListener('click', () => _playLiveChannel(ch));
  return el;
}

async function _playLiveChannel(ch) {
  try {
    const mod = await import('./files/live-tv-rails.js');
    await mod.playLiveTvChannel({
      serverId: ch.server_id,
      channelId: ch.external_id,
      name: ch.name || '',
    });
  } catch (err) {
    console.warn('[media] live channel play failed:', err);
  }
}

// Full "See all" Live TV view — the backend's rails rendered as Media
// rails (one strip per category), pushed onto the view stack.
async function _renderLiveTvView() {
  _body.innerHTML = `<div class="media-drill-loading">Loading channels…</div>`;
  const railsArr = await _fetchLiveTvRails();
  if (!railsArr) {
    _body.innerHTML = `<div class="media-drill-empty">No Live TV channels available. Connect an Emby or Jellyfin server with Live TV configured.</div>`;
    return;
  }
  const host = document.createElement('div');
  host.className = 'media-rails';
  for (const rail of railsArr) {
    const channels = (rail.channels || []).filter((c) => c.external_id && c.server_id);
    if (!channels.length) continue;
    const railEl = _buildLiveTvRailEl(channels, {}, { cap: channels.length });
    railEl.dataset.slug = '';   // sub-rails aren't the pref-controlled slot
    const title = railEl.querySelector('.media-rail-title');
    if (title) title.textContent = rail.title || 'Live TV';
    host.appendChild(railEl);
  }
  _body.innerHTML = '';
  _body.appendChild(host.children.length
    ? host
    : Object.assign(document.createElement('div'), {
      className: 'media-drill-empty', textContent: 'No channels available.',
    }));
}

// Per-server library chips — parity with cast-control's "Your
// libraries" strip. Chips open the media-servers panel (manage / sync);
// per-library browse needs a server-side filter and stays future work.
function _renderLibrariesRow(el, libraries) {
  if (!el || !Array.isArray(libraries) || !libraries.length) return;
  el.innerHTML = `
    <h3 class="media-rail-title">Your libraries</h3>
    <div class="media-libraries-chips">
      ${libraries.map((lib) => `
        <button class="media-library-chip" type="button" title="Open server management">
          <span>${escapeHtml(lib.name || lib.id || 'Library')}</span>
          ${lib.provider ? `<span class="media-library-provider">${escapeHtml(lib.provider)}</span>` : ''}
        </button>`).join('')}
    </div>
  `;
  el.classList.remove('hidden');
  el.querySelectorAll('.media-library-chip').forEach((chip) => {
    chip.addEventListener('click', async () => {
      const mod = await import('./media-servers.js').catch(() => null);
      await mod?.openMediaServers?.();
    });
  });
}

/* Rail customization popover — the single place the user organizes Media.
 * Rails come from the shared cast catalog (/api/cast/rails/catalog) plus the
 * Media-only extras (Live TV), minus the rails Media hides by policy (those
 * have their own surfaces). Each row can be shown/hidden (checkbox) AND
 * reordered (drag handle or keyboard ↑/↓); order + visibility persist per-user.
 * We decide neither placement nor visibility for the user — this is the
 * mechanism that lets them decide. */
async function _openRailPrefs(anchor) {
  document.querySelector('.media-rails-prefs')?.remove();
  let catalog = [];
  try {
    const resp = await fetch('/api/cast/rails/catalog', { credentials: 'same-origin' });
    if (resp.ok) {
      const body = await resp.json();
      catalog = Array.isArray(body?.rails) ? body.rails
        : Array.isArray(body) ? body : [];
    }
  } catch (err) {
    console.warn('[media] rails catalog fetch failed:', err);
  }
  catalog = catalog.filter((r) => r?.slug && !HIDDEN_RAILS.includes(r.slug));

  const prefs = await _loadRailPrefs();
  const hidden = new Set(prefs.hidden);
  // Merge catalog + Media-only rails, de-dup by slug, order by saved prefs.
  const bySlug = new Map();
  for (const r of [...catalog, ...MEDIA_EXTRA_RAILS]) {
    if (r?.slug && !bySlug.has(r.slug)) bySlug.set(r.slug, r);
  }
  const rows = _sortByRailOrder([...bySlug.values()], prefs.order, (r) => r.slug);
  if (!rows.length) return;

  const pop = document.createElement('div');
  pop.className = 'media-rails-prefs';
  pop.setAttribute('role', 'dialog');
  pop.setAttribute('aria-label', 'Customize rails');
  pop.innerHTML = `
    <div class="media-rails-prefs-title">Home rails</div>
    <p class="media-rails-prefs-hint">Toggle to show or hide. Drag the handle (or focus it and press ↑/↓) to reorder.</p>
    <div class="media-rails-prefs-list" role="list">
      ${rows.map((r) => `
        <div class="media-rails-pref" role="listitem" data-rail-slug="${escapeHtml(r.slug)}">
          <button type="button" class="media-rails-pref-grip" aria-label="Reorder ${escapeHtml(r.title || r.slug)} (use arrow keys)" title="Drag to reorder">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><line x1="4" y1="8" x2="20" y2="8"/><line x1="4" y1="16" x2="20" y2="16"/></svg>
          </button>
          <label class="media-rails-pref-label">
            <input type="checkbox" data-rail-slug="${escapeHtml(r.slug)}" ${hidden.has(r.slug) ? '' : 'checked'}>
            <span>${escapeHtml(r.title || r.slug)}</span>
          </label>
        </div>`).join('')}
    </div>
    <button type="button" class="media-rails-prefs-reset">Reset order</button>
  `;
  document.body.appendChild(pop);
  const rect = anchor.getBoundingClientRect();
  pop.style.top = `${rect.bottom + 8}px`;
  pop.style.right = `${Math.max(8, window.innerWidth - rect.right)}px`;

  const listEl = pop.querySelector('.media-rails-prefs-list');

  const close = () => {
    pop.remove();
    document.removeEventListener('pointerdown', onAway, true);
  };
  const onAway = (e) => { if (!pop.contains(e.target) && e.target !== anchor) close(); };
  document.addEventListener('pointerdown', onAway, true);

  const _rerenderHomeIfVisible = () => {
    const top = _viewStack[_viewStack.length - 1];
    if (top?.kind === 'home') _renderTop();
  };

  // Persist the current DOM row order.
  const _persistOrder = async () => {
    const order = [...listEl.querySelectorAll('.media-rails-pref')]
      .map((el) => el.dataset.railSlug).filter(Boolean);
    await _saveRailOrder(order);
    _rerenderHomeIfVisible();
  };

  // Visibility toggles.
  pop.querySelectorAll('input[data-rail-slug]').forEach((box) => {
    box.addEventListener('change', async () => {
      const slug = box.dataset.railSlug;
      if (box.checked) hidden.delete(slug);
      else hidden.add(slug);
      await _saveRailPrefs([...hidden]);
      _rerenderHomeIfVisible();
    });
  });

  // Reset order → clear the pref (back to catalog default).
  pop.querySelector('.media-rails-prefs-reset').addEventListener('click', async () => {
    await _saveRailOrder([]);
    close();
    _rerenderHomeIfVisible();
  });

  _wireRailReorder(listEl, _persistOrder);
}

// Pointer-drag + keyboard reorder for the rail-prefs list. Pointer Events
// cover mouse + touch uniformly. Move/up are bound to the DOCUMENT for the
// duration of a drag (not the grip via setPointerCapture, which was flaky on
// desktop — the pointer leaving the 28px button dropped the gesture), so the
// row follows the cursor anywhere. The grip is also a focusable button so
// keyboard users move a row with ↑/↓. No HTML5 DnD (poor on touch).
function _wireRailReorder(listEl, onCommit) {
  let dragRow = null;
  let moveHandler = null;
  let upHandler = null;

  const rowsExcept = (row) =>
    [...listEl.querySelectorAll('.media-rails-pref')].filter((r) => r !== row);

  // Insert `row` at the position implied by pointer Y (before the first
  // sibling whose vertical midpoint is below the pointer).
  const _placeByY = (row, y) => {
    let before = null;
    for (const sib of rowsExcept(row)) {
      const box = sib.getBoundingClientRect();
      if (y < box.top + box.height / 2) { before = sib; break; }
    }
    listEl.insertBefore(row, before);
  };

  const endDrag = async () => {
    if (!dragRow) return;
    const row = dragRow;
    dragRow = null;
    row.classList.remove('is-dragging');
    if (moveHandler) document.removeEventListener('pointermove', moveHandler);
    if (upHandler) {
      document.removeEventListener('pointerup', upHandler);
      document.removeEventListener('pointercancel', upHandler);
    }
    moveHandler = upHandler = null;
    await onCommit();
  };

  listEl.querySelectorAll('.media-rails-pref-grip').forEach((grip) => {
    const row = grip.closest('.media-rails-pref');

    grip.addEventListener('pointerdown', (e) => {
      if (e.button != null && e.button !== 0) return;   // primary button only
      e.preventDefault();
      dragRow = row;
      row.classList.add('is-dragging');
      moveHandler = (ev) => { if (dragRow) _placeByY(dragRow, ev.clientY); };
      upHandler = () => { endDrag(); };
      document.addEventListener('pointermove', moveHandler);
      document.addEventListener('pointerup', upHandler);
      document.addEventListener('pointercancel', upHandler);
    });

    // Keyboard reorder: ↑/↓ move the row one slot, keeping focus on the grip.
    grip.addEventListener('keydown', async (e) => {
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
      e.preventDefault();
      const prev = row.previousElementSibling;
      const next = row.nextElementSibling;
      if (e.key === 'ArrowUp' && prev) listEl.insertBefore(row, prev);
      else if (e.key === 'ArrowDown' && next) listEl.insertBefore(next, row);
      else return;
      grip.focus();
      await onCommit();
    });
  });
}


/* ── Detail view (shared consumption/detail.js) ────────────────── */

function _renderDetailView(view) {
  _body.innerHTML = '<div class="media-detail-host"></div>';
  renderDetail(_body.firstElementChild, view.item, {
    fetchFileEntry: _fetchEntry,
    onPlayFile: _playFile,
    onOpenItem: (tile) => _pushView({ kind: 'detail', item: tile, scrollTop: 0 }),
    onCast: (tile, anchor) => _openCastPickerForItem(tile, anchor),
  });
}


/* ── Section view ("See all" grid) ─────────────────────────────── */

function _renderSectionView(view) {
  _body.innerHTML = '<div class="media-grid-host"></div>';
  renderSectionGrid(_body.firstElementChild, view.section, {
    onTileActivate: _onTileActivate,
    onTileSecondary: _onTileSecondary,
    keep: _tileInMediaScope,   // images / local audio / docs stay in Files + Library
  });
}


/* ── Search view ───────────────────────────────────────────────── */

async function _renderSearch(view) {
  const query = view.query || '';
  _body.innerHTML = `
    <div class="media-rails" id="media-search-rails"></div>
    <div class="media-empty hidden" id="media-search-empty"></div>
    <div class="media-error hidden" id="media-search-error"></div>
  `;
  const rails = _body.querySelector('#media-search-rails');
  const empty = _body.querySelector('#media-search-empty');
  const error = _body.querySelector('#media-search-error');
  if (!query) {
    empty.textContent = 'Type to search your library.';
    empty.classList.remove('hidden');
    return;
  }
  try {
    const resp = await fetch(
      `/api/cast/library/search?q=${encodeURIComponent(query)}`,
      { credentials: 'same-origin' },
    );
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const payload = await resp.json();
    // The user kept typing while we fetched — a newer render owns the body.
    const top = _viewStack[_viewStack.length - 1];
    if (top !== view || top.query !== query || !rails.isConnected) return;
    const sections = _scrubToMediaScope(payload?.sections);
    renderRails(rails, sections, {
      onTileActivate: _onTileActivate,
      onTileSecondary: _onTileSecondary,
      hiddenSlugs: ['gallery'],
      wideSlugs: ['episodes'],
    });
    if (!rails.children.length) {
      empty.textContent = `Nothing matched “${query}”.`;
      empty.classList.remove('hidden');
    }
  } catch (e) {
    error.textContent = `Search failed: ${e.message}`;
    error.classList.remove('hidden');
  }
}


/* ── Series / chapter drill-in view (comics) ───────────────────── */

async function _renderSeries(view) {
  const isComic = view.item.kind === 'comic' || view.item.entity_kind === 'comic_series';
  const endpoint = isComic
    ? `/api/cast/library/chapters/${encodeURIComponent(view.item.file_id)}`
    : `/api/cast/library/episodes/${encodeURIComponent(view.item.file_id)}`;
  _body.innerHTML = `<div class="media-drill-loading">${isComic ? 'Loading chapters' : 'Loading episodes'}…</div>`;
  try {
    const r = await fetch(endpoint, { credentials: 'same-origin', cache: 'no-store' });
    if (!r.ok) {
      _body.innerHTML = `<div class="media-drill-error">Couldn't load ${isComic ? 'chapters' : 'episodes'} (HTTP ${r.status}).</div>`;
      return;
    }
    const body = await r.json();
    if (isComic) {
      const chapters = body.chapters || [];
      if (!chapters.length) {
        _body.innerHTML = `<div class="media-drill-empty">No chapters indexed yet.</div>`;
        return;
      }
      // Series-first drill: hero + read-state actions live in the
      // shared consumption component (parity with files/comics.js).
      const mod = await import('./consumption/comic-series.js');
      _body.innerHTML = '<div class="media-comic-host"></div>';
      await mod.renderComicSeries(_body.firstElementChild, body, {
        onOpenChapter: (ch) => _playItem(ch),
        onSecondary: _onTileSecondary,
      });
    } else {
      const seasons = body.seasons || [];
      if (!seasons.length) {
        _body.innerHTML = `<div class="media-drill-empty">No episodes available yet.</div>`;
        return;
      }
      // Pre-built fragments — every field is escaped inside _seasonGroup.
      const rows = seasons.map(_seasonGroup).join('');
      _body.innerHTML = `<div class="media-drill">${rows}</div>`;
      const flat = seasons.flatMap((s) => s.episodes || []);
      _wireDrillRows(_body.querySelectorAll('[data-media-drill-id]'), flat);
    }
  } catch (err) {
    _body.innerHTML = `<div class="media-drill-error">Network error.</div>`;
  }
}

function _seasonGroup(season) {
  return `
    <section class="media-drill-season">
      <header class="media-drill-season-head">${escapeHtml(season.label || `Season ${season.season_number}`)}</header>
      ${(season.episodes || []).map(_episodeRow).join('')}
    </section>
  `;
}

function _episodeRow(ep) {
  const cover = ep.cover_url || '';
  const pct = Math.max(0, Math.min(100, ep.progress_pct || 0));
  const sn = ep.season_number || 0;
  const en = ep.episode_number || 0;
  const label = sn && en ? `S${sn}E${String(en).padStart(2, '0')}` : '';
  return `
    <div class="media-drill-row" data-media-drill-id="${escapeHtml(ep.file_id)}">
      <div class="media-drill-art">
        ${cover ? `<img src="${escapeHtml(cover)}" alt="" loading="lazy" onerror="this.style.display='none'">` : ''}
      </div>
      <div class="media-drill-meta">
        ${label ? `<div class="media-drill-num">${escapeHtml(label)}</div>` : ''}
        <div class="media-drill-title">${escapeHtml(ep.title || 'Untitled')}</div>
        ${pct > 0 ? `<div class="media-drill-prog"><div class="media-drill-prog-fill" style="width:${pct}%"></div></div>` : ''}
      </div>
    </div>
  `;
}

function _wireDrillRows(rows, items) {
  rows.forEach((el) => {
    const fid = el.dataset.mediaDrillId;
    const item = items.find((x) => x.file_id === fid);
    if (!item) return;
    el.addEventListener('click', () => _playItem(item));
    el.addEventListener('contextmenu', (ev) => {
      ev.preventDefault();
      _onTileSecondary(item, ev);
    });
  });
}


/* ── Tile dispatch ─────────────────────────────────────────────── */

function _onTileActivate(item, ev) {
  const action = item.play?.action || '';
  if (action === 'browse_series') {
    if (item.kind === 'comic' || item.entity_kind === 'comic_series') {
      _pushView({ kind: 'series', item, scrollTop: 0 });
    } else {
      // Video series: the detail view carries next-up + the episode
      // list (richer than the bare drill-in, same data resolved).
      _pushView({ kind: 'detail', item, scrollTop: 0 });
    }
    return;
  }

  // Direct-play paths: the Continue rail (resume IS the intent),
  // comic chapters (reader), and music videos (Grove handoff).
  // Everything else opens the detail view first.
  const railSlug = ev?.currentTarget?.closest?.('.media-rail')?.dataset?.slug || '';
  const ek = (item.entity_kind || '').toLowerCase();
  const directPlay = railSlug === 'resume'
    || item.kind === 'comic'
    || ek === 'music_video';
  if (directPlay) {
    _playItem(item);
  } else {
    _pushView({ kind: 'detail', item, scrollTop: 0 });
  }
}

// Hand off to the canonical opener (files/open-content.js), which
// wraps the local-player cascade in files/actions.js.
// activateFile inspects mime_type + extension + kind helpers to pick
// the right player — floating video (transcode-aware seek, multi-audio,
// multi-sub, speed, A/V sync, next-episode chain), music_video → Grove,
// global audio singleton + mini-player + read-along trigger, comic
// reader (series-first, progress, kbd nav), image gallery (prev/next,
// download, reference-in-chat), pdf/html/archive previews. Also runs
// the armed-device cast intercept — if a TV is armed, the player
// forwards there instead of opening locally.
//
// Cast tiles only carry the lean { file_id, kind, title, ... } shape;
// activateFile's kind-helpers need mime_type / name / source_metadata
// off the full file_index row, so fetch it first.
async function _playItem(item) {
  try {
    const file = await _fetchEntry(item.file_id);
    if (!file) {
      console.warn('[media] no file_entry for', item.file_id);
      return;
    }
    _playFile(file);
  } catch (err) {
    console.warn('[media] activation failed:', err);
  }
}

async function _playFile(file) {
  try {
    _rememberFile(file);
    // Canonical opener (2026-07-18 class fix): raw activateFile here
    // silently ate media-server rows without stream keys (item-menu
    // "Play" on a movie) and sent resume-rail audiobooks to the
    // fullscreen overlay that restarts at 0 instead of the progress-
    // safe mini-player. openContent applies the per-kind rules.
    const oc = await import('./files/open-content.js');
    await oc.openContent(file);
  } catch (err) {
    console.warn('[media] playback handoff failed:', err);
  }
}


/* ── Cast to TV (secondary) ────────────────────────────────────── */

function _capabilityForKind(kind) {
  switch (kind) {
    case 'audio': return 'media.audio_play@1';
    case 'video': return 'media.video_play@1';
    // Comics ride the video-cast trusted path (cast-picker routes
    // trusted devices through /api/cast/send with cast-comic surface).
    case 'comic': return 'media.video_play@1';
    default:      return 'media.video_play@1';
  }
}

async function _openCastPickerForItem(item, anchor) {
  const mod = await import('./cast-picker.js').catch((err) => {
    console.warn('[media] cast-picker import failed:', err);
    return null;
  });
  if (!mod?.openCastPicker) return;
  const kind = (item.kind || '').toLowerCase();
  const capability = _capabilityForKind(kind);
  const content = {
    fileId: item.file_id,
    title: item.title || '',
    posterUrl: item.cover_url || '',
  };
  // DLNA fallback in cast-picker._dispatchCast wants a content_url for
  // legacy renderers. Trusted Augmentum receivers ignore this and use
  // the cast-{kind} surface_url instead (built from fileId).
  if (kind === 'audio' || kind === 'video') {
    content.contentUrl = `/api/media/stream/${encodeURIComponent(item.file_id)}`;
  }
  mod.openCastPicker({ anchor, capability, content });
}

function _onTileSecondary(item, ev) {
  // Series tiles don't have a single playable item — let the user
  // open the drill-in instead of trying to cast the wrapper.
  if ((item.play || {}).action === 'browse_series') {
    _onTileActivate(item, ev);
    return;
  }
  const tileEl = ev?.currentTarget instanceof Element
    && ev.currentTarget.classList?.contains('media-tile')
    ? ev.currentTarget : null;
  import('./consumption/item-menu.js').then((m) => {
    m.openItemMenu(item, ev, {
      onPlay: (it) => _playItem(it),
      onOpenDetail: (it) => _pushView({ kind: 'detail', item: it, scrollTop: 0 }),
      onCast: (it, anchor) => _openCastPickerForItem(it, anchor),
      onChanged: (it) => _refreshTileInPlace(tileEl, it),
    });
  }).catch((err) => {
    console.warn('[media] item menu failed, falling back to cast picker:', err);
    const anchor = ev?.currentTarget || _overlay.querySelector('#media-devices-btn');
    _openCastPickerForItem(item, anchor);
  });
}

// After a context-menu progress write, swap the tile element for a
// fresh render so the watched badge / progress bar reflect the new
// state without re-fetching the whole rail.
async function _refreshTileInPlace(tileEl, item) {
  if (!tileEl || !tileEl.isConnected) return;
  try {
    const { renderTile } = await import('./consumption/tile.js');
    const variant = tileEl.classList.contains('media-tile-wide') ? 'wide' : 'portrait';
    tileEl.replaceWith(renderTile(item, {
      onActivate: _onTileActivate,
      onSecondary: _onTileSecondary,
      variant,
    }));
  } catch (err) {
    console.warn('[media] tile refresh failed:', err);
  }
}


/* ── Public API ────────────────────────────────────────────────── */

export async function openMedia() {
  _ensureDom();
  _overlay.classList.remove('hidden');
  _overlay.focus();
  _opened = true;
  if (!_initialized) {
    _initialized = true;
    _setView({ kind: 'home', scrollTop: 0 });
  } else {
    // Re-render the current top view. Home view re-fetches in
    // background; drill-in views are cheap to re-paint from cache-busted
    // endpoint (the user expects "go back to where I was").
    _renderTop();
  }
}

export function closeMedia() {
  if (_overlay) _overlay.classList.add('hidden');
  _opened = false;
}
