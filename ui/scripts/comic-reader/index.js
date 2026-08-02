/**
 * Comic / manga reader — Phase 1 MVP.
 *
 * Full-screen overlay that displays one page at a time, fetching pages
 * from ``/api/media/comic/page/{file_id}?page=N`` (1-indexed). The route
 * attaches provider auth server-side, so this module just does vanilla
 * ``<img>`` loads.
 *
 * What's in scope here:
 *   - Single-page viewer with object-fit: contain
 *   - Keyboard nav: ← →, Space, Esc, Home, End, R (reading direction)
 *   - Touch swipe (horizontal)
 *   - Chrome auto-hide on idle (2s) — revealed on pointer movement / tap
 *   - Preload ±2 pages via <link rel=preload> (warm browser cache)
 *   - Page counter + progress ring in the chrome
 *   - Progress push to /api/media/progress on close + page change
 *   - prefers-reduced-motion honored (snaps instead of transitioning)
 *
 * Deliberately out of scope (deferred to reader-plan Phase 2):
 *   - Dual-page view, long-strip mode, crop-borders, double-page split
 *   - Bookmarks within a book
 *   - Translation overlay, Guided View
 *   - Thumbnail scrubber
 *   - Background color picker, sepia/night modes
 *
 * The goal is a polished single-page reader good enough for live testing
 * against a real Suwayomi / Komga server. Fancier viewer modes land later.
 */

import { escapeHtml, showChoiceToast, showToast } from '../app.js';
import {
  bluetoothHandoffAvailable,
  copySurfaceReceiverUrl,
  createComicSurfaceHandoff,
  isBluetoothHandoffBlockedError,
  patchSurfaceState,
  rememberBluetoothHandoffBlocked,
  requestSurfaceBluetoothTarget,
  sendHandoffOverBluetooth,
} from '../surface-handoff.js?v=surface-handoff-20260512a';
import {
  sendHandoffOverCast,
  surfaceCastConfigured,
} from '../surface-cast.js?v=surface-handoff-20260512a';
import { openCastPicker } from '../cast-picker.js';
import { getSettings } from '../settings.js';
import { mountNarrationBar } from './narration-bar.js';

// Reading direction per file_id — users who open a Japanese manga once want
// it to stay RTL on reopen. Server-synced (see the pref store below);
// localStorage is the offline cache. Read/written through the in-memory cache.
const _DIR_STORAGE_KEY = 'comic-reader-direction-v1';

// Returns '' when this file has no stored direction — NOT 'ltr'. Those are
// different facts, and collapsing them is what made opening a new manga
// silently reset to left-to-right: with no "unset" state, every unseen file
// looked like a deliberate LTR preference. Direction isn't cosmetic — it drives
// panel order, swipe direction AND narration order, so a wrong one scrambles a
// chapter and reads as a model failure rather than as a setting.
function _loadDir(fileId) {
  return _prefs.dir[fileId] || '';
}

// The last direction the user EXPLICITLY chose, anywhere. This is the fallback
// for a file and series we have never seen, because someone who reads manga has
// already answered this question — silently answering it again with 'ltr'
// throws that away. Not an auto-pick: it is the user's own most recent choice,
// which is the only defensible thing to carry forward.
const _LAST_DIR_STORAGE_KEY = 'comic-reader-last-direction-v1';

function _loadLastDir() {
  const v = _prefs.global.lastDirection || _lsGet(_LAST_DIR_STORAGE_KEY) || '';
  return (v === 'ltr' || v === 'rtl') ? v : '';
}

function _saveLastDir(dir) {
  _prefs.global.lastDirection = dir;
  _lsSet(_LAST_DIR_STORAGE_KEY, dir);
  // Rides in the existing global blob rather than a key of its own — that's the
  // one _ensurePrefsLoaded already hydrates from the server, so a separate key
  // would persist but never come back on another device.
  _putUiKey('comicReaderPrefs', _prefs.global);
}

function _saveDir(fileId, dir) {
  _prefs.dir[fileId] = dir;
  _lsSet(_DIR_STORAGE_KEY, JSON.stringify(_prefs.dir));
  _putUiKey('comicReaderDirPrefs', _prefs.dir);
}

function _extractMeta(file) {
  const meta = file?.source_metadata || {};
  const extra = meta.extra || {};
  const pageCount = Number(extra.page_count || 0);
  // Clamp to [1, pageCount]. Without this, a stale extra.current_page that
  // exceeds the real page count (e.g. after upstream removed pages) opens
  // the reader at a non-existent page, which then shows a load error and
  // makes forward-nav look broken.
  let currentPage = Math.max(1, Math.round(Number(extra.current_page || 0)) || 1);
  if (pageCount > 0) currentPage = Math.min(currentPage, pageCount);
  const isFinished = !!extra.is_finished;

  // Pretty title hierarchy: prefer series + volume/chapter, fall back to
  // the file name. Works for both Komga (volumes) and Suwayomi (chapters).
  const seriesName = extra.series_name || '';
  const volume = extra.volume || (extra.volume_sort != null ? `Vol. ${extra.volume_sort}` : '');
  const chapter = extra.chapter_number != null
    ? `Ch. ${extra.chapter_number}`
    : (extra.chapter_index != null ? `Ch. ${extra.chapter_index + 1}` : '');
  const chapterName = extra.chapter_name || '';

  let subtitle = '';
  if (volume) subtitle = typeof volume === 'number' ? `Vol. ${volume}` : String(volume).startsWith('Vol') ? volume : `Vol. ${volume}`;
  else if (chapter) subtitle = chapterName ? `${chapter} · ${chapterName}` : chapter;

  return {
    fileId: file.id,
    title: seriesName || file.name || 'Untitled',
    subtitle,
    pageCount,
    currentPage,
    isFinished,
    provider: meta.provider || '',
  };
}

function _pageUrl(fileId, page, opts = {}) {
  const params = new URLSearchParams();
  params.set('page', String(page));
  if (opts.thumb) params.set('thumb', '1');
  else params.set('quality', opts.quality || 'raw');
  return `/api/media/comic/page/${encodeURIComponent(fileId)}?${params}`;
}

// Push progress best-effort — failure doesn't block reading. Debounced
// at the caller so we don't flood the server with a POST per page flip.
async function _pushProgress(fileId, page, pageCount) {
  if (!pageCount) return;
  const isFinished = page >= pageCount;
  try {
    await fetch(`/api/media/progress/${encodeURIComponent(fileId)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        // Audio-shaped protocol — ``current_time_s`` carries the page
        // number, ``duration_s`` carries the total page count. The
        // Komga / Suwayomi provider.push_progress translates on the way
        // to the upstream server.
        current_time_s: page,
        duration_s: pageCount,
        is_finished: isFinished,
      }),
    });
  } catch { /* best-effort */ }
}

// -------------------------------------------------------------------------
// Reader preferences — two layers:
//
//  1. Global defaults — what a user's next-new-series reader opens with.
//     Stored under flat localStorage keys. This is what changes when the
//     user flips F / W / R on any reader.
//
//  2. Per-series overrides — a user who configures Berserk as RTL paged
//     should not have to re-set it on every chapter re-open, and their
//     Solo Leveling reader should still default to LTR webtoon. Stored
//     under a single `comic-reader-series-prefs-v1` map keyed by
//     `series_id`. A series can carry any subset of keys; missing keys
//     fall back to the global default.
//
// Resolution on mount: per-series override → global default → hardcoded
// fallback. Write-through: any in-reader setting change updates BOTH
// the per-series record AND the global default (so the user's last
// preference becomes the starting point for new series they open).
// -------------------------------------------------------------------------

const _FIT_STORAGE_KEY        = 'comic-reader-fit-v1';
const _MODE_STORAGE_KEY       = 'comic-reader-mode-v1';
const _BG_STORAGE_KEY         = 'comic-reader-bg-v1';
const _CROP_STORAGE_KEY       = 'comic-reader-crop-v1';
const _AUTOSCROLL_STORAGE_KEY = 'comic-reader-autoscroll-v1';
const _SERIES_PREFS_KEY       = 'comic-reader-series-prefs-v1';

const _FIT_MODES  = ['fit-page', 'fit-width', 'original'];
const _READ_MODES = ['paged', 'dual', 'webtoon'];
const _BG_MODES   = ['dark', 'black', 'sepia', 'paper'];

// Cooldown for the webtoon "scroll-past-end" auto-advance. Prevents a
// runaway cascade if a layout race ever slips past the page-observer +
// scrollTop gates: even with every other guard defeated, the sentinel
// can fire at most once per cooldown window. Keeps any residual bug
// contained to a single chapter skip rather than a multi-chapter
// cascade. Manual chapter-next (button click, `]` key) and the
// page-end auto-fire (user pressing next on the last page) are
// deliberately NOT gated — those are explicit user actions and silent-
// blocking them would feel broken.
const _SENTINEL_COOLDOWN_MS = 60_000;
const _FIT_LABELS = {
  'fit-page':  'Fit page',
  'fit-width': 'Fit width',
  'original':  'Original',
};
const _MODE_LABELS = {
  'paged':   'Paged',
  'dual':    'Spread',
  'webtoon': 'Webtoon',
};
const _BG_LABELS = {
  'dark':  'Dark',
  'black': 'Black',
  'sepia': 'Sepia',
  'paper': 'Paper',
};
// Auto-scroll (webtoon) — continuous hands-free glide, ported from the
// cast receiver's autoplay engine (ui/cast-comic/cast-comic.js,
// startContinuousScroll). px/sec presets match the cast surface's
// Slow/Med/Fast so a speed that feels right on the TV feels identical
// here.
const _AUTOSCROLL_MODES  = ['off', 'slow', 'medium', 'fast'];
const _AUTOSCROLL_SPEEDS = { off: 0, slow: 120, medium: 240, fast: 400 };
const _AUTOSCROLL_LABELS = { off: 'Off', slow: 'Slow', medium: 'Medium', fast: 'Fast' };

// ── Server-synced pref store ─────────────────────────────────────────────
// Reader prefs (mode/fit/direction/background/crop/auto-scroll) are per-user
// and server-synced via the ui config store, so a user's paged/webtoon choice
// follows them across devices (CLAUDE.md: default to server-side persistence).
// localStorage is the OFFLINE cache + the migration source for prefs set before
// sync existed. An in-memory cache backs the synchronous _load*/_save* the
// reader already calls; _ensurePrefsLoaded() (awaited in openComicReader) seeds
// it once per session. Writes update the cache + localStorage immediately and
// debounce a PUT to the server.
// ``installDir`` is the server's ``comic_default_reading_direction`` — the
// bottom of the direction chain, replacing the hardcoded 'ltr' that used to sit
// there. Deliberately NOT stored inside ``global``: that blob gets PUT back to
// the ui config store, and echoing a server setting into a user pref blob would
// freeze whatever value happened to be live the first time the reader opened.
const _prefs = { global: {}, series: {}, dir: {}, installDir: 'ltr', loaded: false };
let _prefsLoading = null;

function _lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function _lsSet(k, v) { try { localStorage.setItem(k, v); } catch { /* quota/private */ } }
function _lsJson(k) { try { const r = localStorage.getItem(k); return r ? JSON.parse(r) : null; } catch { return null; } }
function _parseJson(raw) { try { return raw ? JSON.parse(raw) : null; } catch { return null; } }

// Debounced per-key PUT to the ui config store (one flight per key).
const _putTimers = {};
function _putUiKey(key, value) {
  clearTimeout(_putTimers[key]);
  _putTimers[key] = setTimeout(() => {
    fetch('/api/config/ui', {
      method: 'PUT',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: JSON.stringify(value) }),
    }).catch(() => { /* offline — localStorage cache stands */ });
  }, 400);
}

// Gather any pre-sync flat-key localStorage values into a global-prefs object.
function _migrateGlobalFromLs() {
  const g = {};
  const fit = _lsGet(_FIT_STORAGE_KEY);  if (_FIT_MODES.includes(fit)) g.fit = fit;
  const mode = _lsGet(_MODE_STORAGE_KEY); if (_READ_MODES.includes(mode)) g.mode = mode;
  const bg = _lsGet(_BG_STORAGE_KEY);    if (_BG_MODES.includes(bg)) g.background = bg;
  const crop = _lsGet(_CROP_STORAGE_KEY); if (crop === '1' || crop === '0') g.crop = crop === '1';
  const as = _lsGet(_AUTOSCROLL_STORAGE_KEY); if (_AUTOSCROLL_MODES.includes(as)) g.autoscroll = as;
  return g;
}

// Seed the in-memory cache once: localStorage first (instant/offline), then
// overlay the server's authoritative values; if the server has nothing but
// localStorage does, push a one-time migration up.
async function _ensurePrefsLoaded() {
  if (_prefs.loaded) return;
  if (_prefsLoading) return _prefsLoading;
  _prefsLoading = (async () => {
    _prefs.global = _migrateGlobalFromLs();
    _prefs.series = _lsJson(_SERIES_PREFS_KEY) || {};
    _prefs.dir = _lsJson(_DIR_STORAGE_KEY) || {};
    try {
      const resp = await fetch('/api/config/ui', { credentials: 'same-origin' });
      if (resp.ok) {
        const body = await resp.json();
        const g = _parseJson(body.comicReaderPrefs);
        const s = _parseJson(body.comicReaderSeriesPrefs);
        const d = _parseJson(body.comicReaderDirPrefs);
        if (g && typeof g === 'object') _prefs.global = { ..._prefs.global, ...g };
        if (s && typeof s === 'object') _prefs.series = s;
        if (d && typeof d === 'object') _prefs.dir = d;
        if (!body.comicReaderPrefs && Object.keys(_prefs.global).length) _putUiKey('comicReaderPrefs', _prefs.global);
        if (!body.comicReaderSeriesPrefs && Object.keys(_prefs.series).length) _putUiKey('comicReaderSeriesPrefs', _prefs.series);
        if (!body.comicReaderDirPrefs && Object.keys(_prefs.dir).length) _putUiKey('comicReaderDirPrefs', _prefs.dir);
      }
    } catch { /* offline — localStorage-seeded cache stands */ }
    try {
      const cfg = await fetch('/api/config/section/comic', { credentials: 'same-origin' })
        .then(r => (r.ok ? r.json() : null));
      const d = cfg?.comic_default_reading_direction;
      if (d === 'ltr' || d === 'rtl') _prefs.installDir = d;
    } catch { /* offline — 'ltr' stands, same as the server's own default */ }
    _prefs.loaded = true;
  })();
  return _prefsLoading;
}

function _loadFit() {
  return _FIT_MODES.includes(_prefs.global.fit) ? _prefs.global.fit : 'fit-page';
}
function _saveFit(fit) {
  _prefs.global.fit = fit;
  _lsSet(_FIT_STORAGE_KEY, fit);
  _putUiKey('comicReaderPrefs', _prefs.global);
}
function _loadMode() {
  return _READ_MODES.includes(_prefs.global.mode) ? _prefs.global.mode : 'paged';
}
function _saveMode(mode) {
  _prefs.global.mode = mode;
  _lsSet(_MODE_STORAGE_KEY, mode);
  _putUiKey('comicReaderPrefs', _prefs.global);
}
function _loadBg() {
  return _BG_MODES.includes(_prefs.global.background) ? _prefs.global.background : 'dark';
}
function _saveBg(bg) {
  _prefs.global.background = bg;
  _lsSet(_BG_STORAGE_KEY, bg);
  _putUiKey('comicReaderPrefs', _prefs.global);
}
function _loadCrop() {
  return _prefs.global.crop === true;
}
function _saveCrop(on) {
  _prefs.global.crop = !!on;
  _lsSet(_CROP_STORAGE_KEY, on ? '1' : '0');
  _putUiKey('comicReaderPrefs', _prefs.global);
}
function _loadAutoScroll() {
  return _AUTOSCROLL_MODES.includes(_prefs.global.autoscroll) ? _prefs.global.autoscroll : 'off';
}
function _saveAutoScroll(v) {
  _prefs.global.autoscroll = v;
  _lsSet(_AUTOSCROLL_STORAGE_KEY, v);
  _putUiKey('comicReaderPrefs', _prefs.global);
}

/** Read the per-series override record, or null if none. */
function _loadSeriesPrefs(seriesId) {
  if (!seriesId) return null;
  const rec = _prefs.series[seriesId];
  return rec && typeof rec === 'object' ? rec : null;
}

/** Merge a partial record into the per-series override map. */
function _patchSeriesPrefs(seriesId, patch) {
  if (!seriesId || !patch) return;
  _prefs.series[seriesId] = { ...(_prefs.series[seriesId] || {}), ...patch };
  _lsSet(_SERIES_PREFS_KEY, JSON.stringify(_prefs.series));
  _putUiKey('comicReaderSeriesPrefs', _prefs.series);
}

/** Resolve the effective preference for one key, following the fallback
 *  chain series → global → default. Exported shape keeps call sites tidy. */
function _resolvePref(key, seriesId, globalValue, validValues) {
  const seriesPrefs = _loadSeriesPrefs(seriesId);
  const fromSeries = seriesPrefs?.[key];
  if (fromSeries && validValues.includes(fromSeries)) return fromSeries;
  if (globalValue && validValues.includes(globalValue)) return globalValue;
  return validValues[0];
}

/** Direction gets its own resolver because its chain is one link longer than
 *  the others: series → this file → last explicit choice → INSTALL DEFAULT.
 *  The last link is the whole point — ``_resolvePref`` bottoms out at
 *  ``validValues[0]``, which for direction meant a silent 'ltr' that undid a
 *  manga reader's setting every time the first three links were empty. */
function _resolveDir(seriesId, fileId) {
  const fromSeries = _loadSeriesPrefs(seriesId)?.direction;
  if (fromSeries === 'ltr' || fromSeries === 'rtl') return fromSeries;
  return _loadDir(fileId) || _loadLastDir() || _prefs.installDir || 'ltr';
}

/** Boolean variant — same fallback chain for a yes/no preference. */
function _resolveBoolPref(key, seriesId, globalValue) {
  const seriesPrefs = _loadSeriesPrefs(seriesId);
  const fromSeries = seriesPrefs?.[key];
  if (fromSeries === true || fromSeries === false) return fromSeries;
  return !!globalValue;
}


// -------------------------------------------------------------------------
// Border detection — canvas-based. Samples the four corner pixels to find
// the dominant "background" color, then walks each edge inward until a
// row/column contains anything outside the threshold. Returns inset
// fractions, or null when the border recovery isn't meaningful.
//
// Costs: one getImageData on a 128px downscale (~100KB, <5ms). Called once
// per (file, page) and cached — no per-frame overhead.
// -------------------------------------------------------------------------

const _CROP_THRESHOLD_PER_CHANNEL = 18;  // tolerance for "same color as border"
const _CROP_MIN_TOTAL = 0.02;             // <2% recovery isn't worth re-encoding
const _CROP_MAX_TOTAL = 0.55;             // >55% means the scan is probably a splash page — skip

function _detectBorders(img) {
  if (!img?.naturalWidth || !img?.naturalHeight) return null;
  const scale = Math.min(128 / img.naturalWidth, 128 / img.naturalHeight, 1);
  const w = Math.max(2, Math.round(img.naturalWidth * scale));
  const h = Math.max(2, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  try {
    ctx.drawImage(img, 0, 0, w, h);
  } catch { return null; }
  let data;
  // getImageData throws on cross-origin-tainted canvases. Same-origin
  // proxy endpoint serves the pages so this should be fine, but catch
  // defensively — a thrown detect shouldn't crash the reader.
  try { data = ctx.getImageData(0, 0, w, h).data; }
  catch { return null; }

  const px = (x, y) => {
    const i = (y * w + x) * 4;
    return [data[i], data[i + 1], data[i + 2]];
  };
  const corners = [px(0, 0), px(w - 1, 0), px(0, h - 1), px(w - 1, h - 1)];
  const median = (idx) => {
    const s = corners.map(c => c[idx]).sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  };
  const bg = [median(0), median(1), median(2)];
  const isBorder = (x, y) => {
    const c = px(x, y);
    return Math.abs(c[0] - bg[0]) + Math.abs(c[1] - bg[1]) + Math.abs(c[2] - bg[2])
      < _CROP_THRESHOLD_PER_CHANNEL * 3;
  };
  const rowIsBorder = (y) => {
    for (let x = 0; x < w; x++) if (!isBorder(x, y)) return false;
    return true;
  };
  const colIsBorder = (x) => {
    for (let y = 0; y < h; y++) if (!isBorder(x, y)) return false;
    return true;
  };

  let top = 0, bot = h - 1, left = 0, right = w - 1;
  while (top < h && rowIsBorder(top))    top++;
  while (bot > top && rowIsBorder(bot))  bot--;
  while (left < w && colIsBorder(left))  left++;
  while (right > left && colIsBorder(right)) right--;

  const tF = top / h;
  const bF = (h - 1 - bot) / h;
  const lF = left / w;
  const rF = (w - 1 - right) / w;
  const total = tF + bF + lF + rF;
  if (total < _CROP_MIN_TOTAL) return null;
  if (total > _CROP_MAX_TOTAL) return null;
  // Per-side cap. A single side trimming >30% of a dimension is almost
  // always the detector misreading a uniform-color top panel (splash
  // pages, full-bleed dark backgrounds, color manga with a wide
  // letterbox header) as "border" and cutting content. The total-only
  // ceiling lets a 35%+0%+0%+0% crop through, which is exactly that
  // failure mode. Real letterboxing is usually symmetric AND smaller.
  if (tF > 0.30 || bF > 0.30 || lF > 0.30 || rF > 0.30) return null;
  return { top: tF, right: rF, bottom: bF, left: lF, bg };
}

/** Produce a cropped dataURL from an already-loaded image. Returns the
 *  original URL if no meaningful crop was found — the caller swaps
 *  unconditionally so there's no "detect + apply" race. */
function _cropImageToDataUrl(img, insets) {
  const srcW = img.naturalWidth;
  const srcH = img.naturalHeight;
  const sx = Math.floor(insets.left * srcW);
  const sy = Math.floor(insets.top * srcH);
  const sw = Math.floor((1 - insets.left - insets.right) * srcW);
  const sh = Math.floor((1 - insets.top - insets.bottom) * srcH);
  if (sw < 64 || sh < 64) return null;
  const canvas = document.createElement('canvas');
  canvas.width = sw;
  canvas.height = sh;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
  try { return canvas.toDataURL('image/jpeg', 0.92); }
  catch { return null; }
}


class ComicReader {
  constructor(file, { siblings = null } = {}) {
    this._file = file;
    this._meta = _extractMeta(file);
    this._pageCount = this._meta.pageCount;
    // Re-read guard: if the chapter's saved position is at (or past) the
    // end — either via is_finished or current_page >= page_count — open
    // at page 1 instead. Otherwise the user lands on the last page and
    // their first "next" press gets interpreted as "advance to next
    // chapter", producing the exact page→chapter jump bug users report.
    // The actual "press next at end" two-press flow still works normally
    // once they've read through legitimately.
    const saved = this._meta.currentPage;
    const atEndOnOpen = this._pageCount > 0 && saved >= this._pageCount;
    const looksFinished = this._meta.isFinished || atEndOnOpen;
    this._page = looksFinished ? 1 : saved;
    // Resume-protection gate. Webtoon mode lays out images with
    // ``loading='lazy'`` + ``height: auto``, so on a fresh mount every
    // undecoded image has 0 height — the resume scroll lands near 0
    // and the IntersectionObserver's first delivery reports "page 1"
    // as dominant. That observer callback would otherwise immediately
    // ``_schedulePushProgress()``, writing page=1 over the saved page
    // a few seconds later (debounce). Same family as the audio race.
    //
    // Starts ``false`` only when there's a real saved page to land on.
    // The observer flips it to ``true`` once it sees a page ≥ saved,
    // and explicit user navigation (``_goTo``) flips it too. While
    // ``false``, ``_schedulePushProgress`` and the unmount sendBeacon
    // both no-op — better to skip a progress write than corrupt the
    // saved position with a transient pre-resume layout artifact.
    this._sawResumePage = !(this._page > 1 && !looksFinished);
    // series_id — resolves preferences that should persist across all
    // chapters in a series (direction, mode, fit, background). Null when
    // the caller opens a reader without series context (flat Files click).
    this._seriesId = file?.series_id || '';
    // Resolve effective settings: per-series override → global default.
    // Direction still has a per-file fallback for users who set RTL on
    // one chapter long before the series concept existed.
    this._dir = _resolveDir(this._seriesId, file.id);
    this._fit = _resolvePref('fit',       this._seriesId, _loadFit(),        _FIT_MODES);
    this._mode = _resolvePref('mode',     this._seriesId, _loadMode(),       _READ_MODES);
    this._bg  = _resolvePref('background',this._seriesId, _loadBg(),         _BG_MODES);
    this._cropBorders = _resolveBoolPref('crop', this._seriesId, _loadCrop());
    // Detection cache: key is `${fileId}|${page}`, value is a cropped
    // dataURL. Bounded implicitly by page turnover — we clear on
    // chapter transition so memory doesn't grow unboundedly.
    this._cropCache = new Map();
    // Sibling chapters (same series) — lets us flow one chapter into
    // the next without dismissing the reader. When not passed, we skip
    // the chapter-nav chrome buttons (single-chapter mode). The caller
    // in comics.js always passes this; direct-from-Files-panel opens
    // can omit it, in which case the user still gets an in-chapter
    // reader but no cross-chapter navigation.
    this._siblings = Array.isArray(siblings) ? siblings : null;
    this._siblingIndex = this._findSiblingIndex(file);
    this._overlay = null;
    this._stage = null;
    this._img = null;
    this._loadingToken = 0;                     // race guard for in-flight fetches
    this._idleTimer = null;
    this._boundKey = null;
    this._progressTimer = null;
    this._surface = null;
    this._surfaceTimer = null;
    this._boundSurfaceScroll = null;
    this._prefetchImgs = new Set();             // keep refs so browser keeps decoded bitmap
    // Webtoon-mode state. Rebuilt on chapter transition.
    this._webtoonObserver = null;
    this._webtoonPages = [];                    // <img> refs, 1-indexed aligned at [i] = page i+1
    // Auto-scroll (webtoon) — the chosen SPEED is now remembered (per-series
    // → global, server-synced like the other reader prefs). But we still don't
    // auto-START on the initial mount: a reader that lurches into motion the
    // instant it opens is a surprise. _autoScrollSuppressMountStart gates only
    // that first render; chapter transitions and paged→webtoon switches resume
    // the glide as before, so binge-reading still flows hands-free.
    this._autoScroll = _resolvePref('autoscroll', this._seriesId, _loadAutoScroll(), _AUTOSCROLL_MODES);
    this._autoScrollSuppressMountStart = true;
    this._autoScrollRaf = 0;
    this._autoScrollLastTs = 0;
    this._autoScrollHeld = false;               // finger down on the strip → glide pauses
    this._wakeLock = null;
    // Sentinel observer fires the next-chapter transition once the
    // "continue" block below the last page scrolls into view. Separate
    // from the page observer because its threshold + lifecycle differ
    // (one-shot, fires + tears down). Null when there is no next chapter.
    this._sentinelObserver = null;
    // True once the user has scrolled into the home stretch of the
    // current chapter — the page observer flips this when a page in
    // the last ~15% of the strip becomes dominant. Sentinel firing is
    // gated on this so the auto-continue can only happen when the user
    // has actually consumed most of the chapter, not because lazy-loaded
    // images give the strip 0 height at mount and put the sentinel in
    // viewport before anything's been read. Reset per chapter.
    this._reachedContinuePoint = false;
    // Timestamp of the last sentinel-driven auto-advance. Persists
    // across chapter transitions so a sentinel firing in chapter B
    // shortly after a sentinel firing in chapter A is detected as a
    // cascade and refused. Manual transitions don't update this.
    this._lastSentinelAdvanceAt = 0;
    // Two-press edge-wrap flag — Tachiyomi-style. When the user presses
    // next on the last page (or prev on page 1), we don't immediately
    // jump chapters; we toast "press again" and set this. A second press
    // within the timeout performs the chapter transition; any other
    // navigation (moving to a different page) resets it. Prevents the
    // "re-open a read chapter → first click skips whole chapter" trap
    // that happens because saved lastPageRead puts you at the edge.
    this._edgeFlag = 'none';                    // 'none' | 'start' | 'end'
    this._edgeFlagTimer = null;
    // One-shot guard for next-chapter prefetch — fires when the user
    // reaches the last few pages so the transition feels instant.
    // Reset on every chapter change.
    this._nextChapterPrefetched = false;
    // Zoom + pan state. Driven by touch pinch, mouse Ctrl+wheel, and
    // double-click toggle. Reset on page / mode / chapter change so a
    // zoomed-in view doesn't leak between pages (surprising — user
    // expects a fresh page). Applies to paged + dual; webtoon relies on
    // native scroll so we leave its transform alone.
    this._zoom = { scale: 1, tx: 0, ty: 0 };
    this._pointers = new Map();                 // pointerId → {x, y, type}
    this._gesture = 'none';                     // 'none' | 'swipe' | 'pan' | 'pinch'
    this._pinchRef = null;                      // {d0, s0, tx0, ty0, cx, cy}
    this._panRef = null;                        // {x0, y0, tx0, ty0}
    this._swipeRef = null;                      // {x0, y0, moved}
    this._suppressNextClick = false;            // set on swipe-end; click→tap-zone suppressed once
  }

  _findSiblingIndex(file) {
    if (!this._siblings || !file?.id) return -1;
    return this._siblings.findIndex(s => s?.id === file.id);
  }

  _hasPrevChapter() {
    return !!this._siblings && this._siblingIndex > 0;
  }
  _hasNextChapter() {
    return !!this._siblings
      && this._siblingIndex >= 0
      && this._siblingIndex < this._siblings.length - 1;
  }

  mount() {
    if (this._overlay) return;
    const m = this._meta;
    const reduceMotion = window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const overlay = document.createElement('div');
    overlay.className = 'comic-reader-overlay'
      + (reduceMotion ? ' comic-reader-reduced-motion' : '')
      + ` mode-${this._mode}`
      + ` ${this._fit}`
      + ` dir-${this._dir}`
      + ` bg-${this._bg}`;
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-label', `Reading ${m.title}`);
    overlay.tabIndex = 0;
    overlay.innerHTML = `
      <div class="comic-reader-chrome comic-reader-chrome-top" data-zone="chrome-top">
        <button type="button" class="comic-reader-btn comic-reader-close"
                aria-label="Close reader" title="Close (Esc)">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
        <button type="button" class="comic-reader-titles comic-reader-titles-btn"
                aria-label="Open library — chapters and series info"
                title="Library — chapters &amp; info (L)"
                aria-expanded="false"
                data-action="open-nav-drawer">
          <div class="comic-reader-title">${escapeHtml(m.title)}</div>
          ${m.subtitle ? `<div class="comic-reader-subtitle">${escapeHtml(m.subtitle)}</div>` : ''}
          <svg class="comic-reader-titles-chevron" width="10" height="10"
               viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"
               aria-hidden="true">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </button>
        <div class="comic-reader-top-tools">
          <button type="button" class="comic-reader-btn comic-reader-mode-toggle"
                  aria-label="Reading mode" title="Cycle paged / spread / webtoon (W)">
            <span class="comic-reader-mode-label">${_MODE_LABELS[this._mode]}</span>
          </button>
          <button type="button" class="comic-reader-btn comic-reader-fit-toggle"
                  aria-label="Fit mode" title="Cycle fit modes (F)">
            <span class="comic-reader-fit-label">${_FIT_LABELS[this._fit]}</span>
          </button>
          <button type="button" class="comic-reader-btn comic-reader-dir-toggle"
                  aria-label="Reading direction"
                  title="Toggle left-to-right / right-to-left (R)">
            <span class="comic-reader-dir-label">${this._dir === 'rtl' ? 'RTL' : 'LTR'}</span>
          </button>
          <button type="button" class="comic-reader-btn comic-reader-surface-toggle"
                  aria-label="Send to TV" title="Send to TV">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="4" width="18" height="12" rx="2"/>
              <path d="M8 20h8"/>
              <path d="M12 16v4"/>
            </svg>
          </button>
          <button type="button" class="comic-reader-btn comic-reader-listen"
                  aria-label="Listen — voiced motion-comic"
                  title="Listen — narrate this comic with pan &amp; scan">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 10v4h4l5 5V5L7 10H3z"/>
              <path d="M16 8a5 5 0 0 1 0 8"/>
            </svg>
          </button>
          <button type="button" class="comic-reader-btn comic-reader-minimize"
                  aria-label="Minimize reader" title="Minimize — resume later (M)">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M5 12h14"/>
            </svg>
          </button>
          <button type="button" class="comic-reader-btn comic-reader-settings-toggle"
                  aria-label="Reader settings" title="Reader settings (S)">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
          </button>
        </div>
      </div>

      ${this._renderNavDrawer()}
      ${this._renderSettingsDrawer()}

      <div class="comic-reader-stage" data-zone="stage">
        <img class="comic-reader-img" alt="" draggable="false">
        <div class="comic-reader-dual" data-zone="dual">
          <img class="comic-reader-dual-img comic-reader-dual-left" alt="" draggable="false">
          <img class="comic-reader-dual-img comic-reader-dual-right" alt="" draggable="false">
        </div>
        <div class="comic-reader-webtoon" data-zone="webtoon"></div>
        <div class="comic-reader-loading" aria-hidden="true">
          <div class="comic-reader-spinner"></div>
        </div>
        <div class="comic-reader-error" hidden>
          <p class="comic-reader-error-msg"></p>
          <button type="button" class="btn btn-sm comic-reader-retry">Retry</button>
        </div>
      </div>

      <!-- Always-visible thin progress bar. Stays put when chrome auto-hides
           so the reader surface never loses sense-of-progress during
           distraction-free reading. -->
      <div class="comic-reader-progress-persistent" role="progressbar"
           aria-valuemin="1" aria-valuemax="${this._pageCount || 1}" aria-valuenow="${this._page}">
        <div class="comic-reader-progress-persistent-fill"></div>
      </div>

      <div class="comic-reader-chrome comic-reader-chrome-bottom" data-zone="chrome-bottom">
        <button type="button" class="comic-reader-btn comic-reader-ch-prev"
                aria-label="Previous chapter" title="Previous chapter ([)"
                ${this._hasPrevChapter() ? '' : 'disabled'}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="11 17 6 12 11 7"/><polyline points="18 17 13 12 18 7"/>
          </svg>
        </button>
        <button type="button" class="comic-reader-btn comic-reader-prev"
                aria-label="Previous page" title="← or swipe right">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </button>
        <div class="comic-reader-progress">
          <div class="comic-reader-page-counter">
            <span class="comic-reader-page-current">${this._page}</span>
            <span class="comic-reader-page-sep">/</span>
            <span class="comic-reader-page-total">${this._pageCount || '?'}</span>
          </div>
          <div class="comic-reader-progress-bar" role="slider"
               tabindex="0"
               aria-label="Scrub pages"
               aria-valuemin="1"
               aria-valuemax="${this._pageCount || 1}"
               aria-valuenow="${this._page}"
               data-zone="scrubber">
            <div class="comic-reader-progress-fill"></div>
            <div class="comic-reader-scrubber-handle" aria-hidden="true"></div>
          </div>
          ${this._siblings ? `<div class="comic-reader-chapter-position">${this._chapterPositionLabel()}</div>` : ''}
        </div>
        <button type="button" class="comic-reader-btn comic-reader-next"
                aria-label="Next page" title="→ or Space or swipe left">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="9 18 15 12 9 6"/>
          </svg>
        </button>
        <button type="button" class="comic-reader-btn comic-reader-ch-next"
                aria-label="Next chapter" title="Next chapter (])"
                ${this._hasNextChapter() ? '' : 'disabled'}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="13 17 18 12 13 7"/><polyline points="6 17 11 12 6 7"/>
          </svg>
        </button>
      </div>

      <!-- Scrubber thumbnail preview. Hidden until the user starts
           dragging on the progress bar; positioned absolute so it can
           float above the chrome without affecting layout. -->
      <div class="comic-reader-scrubber-preview" data-zone="scrubber-preview" hidden>
        <div class="comic-reader-scrubber-preview-frame">
          <img class="comic-reader-scrubber-preview-img" alt="">
        </div>
        <div class="comic-reader-scrubber-preview-label">
          <span class="comic-reader-scrubber-preview-page">1</span>
          <span class="comic-reader-scrubber-preview-sep">/</span>
          <span class="comic-reader-scrubber-preview-total">${this._pageCount || '?'}</span>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    this._overlay = overlay;
    this._stage = overlay.querySelector('.comic-reader-stage');
    this._img = overlay.querySelector('.comic-reader-img');
    document.body.classList.add('comic-reader-open');

    // Visibility: trigger a paint + then add class so CSS transitions
    // animate the fade-in rather than flashing at final opacity.
    requestAnimationFrame(() => overlay.classList.add('visible'));

    this._wireEvents();
    if (this._mode === 'webtoon') {
      this._renderWebtoon();
    } else {
      this._loadPage(this._page);
    }
    // Initial mount render is done — from here on, webtoon renders (chapter
    // transitions, paged→webtoon switch) may auto-resume the saved glide.
    this._autoScrollSuppressMountStart = false;
    overlay.focus();
    // Fire-and-forget manifest refresh — hydrates page_count / is_finished
    // from the upstream provider so stale catalog data doesn't trap the
    // user into a false "end of chapter" edge arming. Runs after the
    // first page has already started loading so reader is usable
    // immediately; manifest just back-fills better counts when ready.
    this._refreshManifest();
    // Warm the next chapter while the user reads this one. Suwayomi's
    // prepare_chapter mutation goes out to the source extension and
    // typically takes 5–10s; firing it now means by the time the user
    // reaches the end of this chapter, the next one is already cached
    // upstream and the transition is instant. Cheap if there's no next
    // chapter (early-return inside _prefetchNextChapter).
    this._prefetchNextChapter();
  }

  async _refreshManifest() {
    const fileId = this._file?.id;
    if (!fileId) return;
    // Snapshot identity — if the user transitions chapters or closes the
    // reader while we're waiting, don't patch the now-stale file.
    const openedFile = this._file;
    const failGracefully = (msg) => {
      // If we have nothing to show (webtoon waiting on page_count, no
      // images loaded yet), surface an error rather than leave the user
      // staring at a permanent spinner. Mode preference stays intact.
      if (this._mode === 'webtoon' && (this._pageCount || 0) <= 0) {
        this._hideLoading();
        this._showError(msg);
      }
    };
    try {
      const resp = await fetch(
        `/api/media/comic/manifest/${encodeURIComponent(fileId)}`,
      );
      if (!resp.ok) {
        failGracefully("Couldn't load chapter manifest. Try opening it again.");
        return;
      }
      const data = await resp.json();
      if (this._file !== openedFile) return;      // user navigated away
      if (!this._overlay) return;                 // reader unmounted
      const freshCount = Number(data.page_count) || 0;
      if (freshCount <= 0) {
        failGracefully("This chapter hasn't been prepared by the provider yet — try again in a moment.");
        return;
      }
      if (freshCount === this._pageCount) return; // already accurate
      // Update the local metadata. `_page` stays where it is unless it'd
      // be past the new end (freshly-detected page count could be lower
      // than the stale stored value, e.g. provider removed pages).
      this._pageCount = freshCount;
      this._meta.pageCount = freshCount;
      if (this._page > freshCount) this._page = freshCount;
      // Sync the DOM bits that show pageCount so chrome + scrubber range
      // both reflect the new truth.
      const totalEl = this._overlay.querySelector('.comic-reader-page-total');
      if (totalEl) totalEl.textContent = String(freshCount);
      const previewTotal = this._overlay.querySelector('.comic-reader-scrubber-preview-total');
      if (previewTotal) previewTotal.textContent = String(freshCount);
      const bar = this._overlay.querySelector('.comic-reader-progress-bar');
      if (bar) bar.setAttribute('aria-valuemax', String(freshCount));
      const persistentBar = this._overlay.querySelector('.comic-reader-progress-persistent');
      if (persistentBar) persistentBar.setAttribute('aria-valuemax', String(freshCount));
      // Webtoon needs a full re-render to materialise the newly-known
      // pages. Paged/dual are fine — only the chrome math changes.
      if (this._mode === 'webtoon') this._renderWebtoon();
      else this._updatePageChrome();
      this._scheduleSurfaceSync({ immediate: true });
    } catch {
      // Network failure. If we have a usable cached count we just keep
      // it and let the existing edge-arm guards handle any stale state.
      // If we have nothing (webtoon stuck on the loading spinner because
      // page_count was 0), surface the failure so the user knows to
      // retry or open another chapter.
      failGracefully("Couldn't reach the provider — try opening this chapter again.");
    }
  }

  /**
   * Hand off to the bottom-docked mini-player and tear down the full
   * overlay. The dock carries just enough context (file, siblings,
   * zoom, webtoon scroll) to reopen the reader exactly where it was.
   * Page / is_finished are already on the server via _pushProgress,
   * so we don't double-track them in the snapshot.
   */
  async minimize() {
    if (!this._overlay) return;
    const uiState = {
      zoom: { ...this._zoom },
      webtoonScrollY: this._mode === 'webtoon' && this._stage
        ? this._stage.scrollTop
        : 0,
    };
    // Comics ride the provider-backed cover proxy, not the generic
    // thumbnail service — comic chapter files aren't `image/*`, so
    // the mime-based default producer would 404. This mirrors what
    // comics.js and the Files grid already use for cover art.
    const meta = this._file?.source_metadata || {};
    const hasCover = !!(meta.has_cover || meta.cover_url);
    const coverUrl = hasCover && this._file?.id
      ? `/api/media/cover/${encodeURIComponent(this._file.id)}`
      : '';
    // Bake the current page state into the file object the mini-player
    // hands back on Resume. Without this, the file we pass is a
    // snapshot from when the reader first mounted — its
    // ``source_metadata.extra.current_page`` is the value the chapter
    // was loaded with (typically 0 / page 1), not what the user is
    // actually on. ``_extractMeta`` reads from that field on construct
    // and would otherwise reset the user to page 1 every Resume.
    //
    // The unmount sendBeacon already pushes the same numbers to the
    // server (and the server-side mirror lands them in extra too), so
    // re-opening from the Files grid afterward also works. This block
    // covers the in-memory minimize → resume path which doesn't hit
    // the server before reopening.
    const freshExtra = {
      ...(meta.extra && typeof meta.extra === 'object' ? meta.extra : {}),
      current_page: this._page,
      page_count: this._pageCount || (meta.extra?.page_count || 0),
      is_finished: this._pageCount > 0 && this._page >= this._pageCount,
    };
    const fileWithFreshProgress = {
      ...this._file,
      source_metadata: { ...meta, extra: freshExtra },
    };
    const ctx = {
      file: fileWithFreshProgress,
      siblings: this._siblings,
      title: this._meta.title,
      subtitle: this._meta.subtitle,
      page: this._page,
      pageCount: this._pageCount,
      coverUrl,
      uiState,
    };
    try {
      const mod = await import('../comic-mini-player.js');
      mod.showComicMini(ctx);
      // Minimize means "set this aside, back to what I was doing" — and
      // the bubble is designed to float over the home surface (chat),
      // not the Files browser the reader happened to be opened from.
      // Close the Files panel so the user lands home with the bubble;
      // no-op when Files isn't open (e.g. opened from a continue rail).
      // Skipped on mini-player failure: stranding the user on home with
      // no recall affordance would be worse than staying in Files.
      try {
        const files = await import('../files/index.js');
        files.closeFiles?.();
      } catch { /* files surface not loaded — nothing to close */ }
    } catch (err) {
      console.error('[comic-reader] mini-player unavailable:', err);
      // If the mini-player module failed to load, fall through to a
      // normal unmount so the user isn't trapped in the reader.
    }
    this.unmount();
  }

  /**
   * Restore transient UI state after a mount triggered by the mini-
   * player's Resume action. Called from openComicReader when a
   * ``resume`` option is passed. Page is already on the server; we
   * only round-trip zoom and webtoon scroll position.
   */
  _restoreResumeState(uiState) {
    if (!uiState || !this._overlay) return;
    if (uiState.zoom && this._mode !== 'webtoon') {
      this._zoom = { ...uiState.zoom };
      this._applyZoomTransform?.();
    }
    if (uiState.webtoonScrollY && this._mode === 'webtoon' && this._stage) {
      // Wait for webtoon images to layout before restoring scroll.
      requestAnimationFrame(() => {
        if (this._stage) this._stage.scrollTop = uiState.webtoonScrollY;
      });
    }
  }

  async startSurfaceHandoff(options = {}) {
    return this._startSurfaceHandoff(options);
  }

  _surfaceScrollRatio() {
    if (this._mode !== 'webtoon' || !this._stage) {
      return 0;
    }
    const pageEl = this._webtoonPages[this._page - 1];
    if (!pageEl) return 0;
    const overflow = Math.max(0, pageEl.offsetHeight - this._stage.clientHeight);
    if (!overflow) return 0;
    const y = this._stage.scrollTop - pageEl.offsetTop;
    return Math.max(0, Math.min(1, y / overflow));
  }

  _surfaceStripScrollRatio() {
    if (this._mode !== 'webtoon' || !this._stage) return 0;
    const maxScroll = Math.max(0, this._stage.scrollHeight - this._stage.clientHeight);
    if (!maxScroll) return 0;
    return Math.max(0, Math.min(1, this._stage.scrollTop / maxScroll));
  }

  _surfaceStatePatch() {
    const ratio = this._surfaceScrollRatio();
    const stripRatio = this._surfaceStripScrollRatio();
    return {
      reader: {
        file_id: this._file?.id || '',
        page: this._page,
        page_count: this._pageCount || 0,
        scroll_ratio: Number(ratio.toFixed(5)),
        strip_scroll_ratio: Number(stripRatio.toFixed(5)),
        mode: this._mode,
        direction: this._dir,
      },
      controller: {
        viewport: {
          width: Math.round(this._stage?.clientWidth || window.innerWidth || 0),
          height: Math.round(this._stage?.clientHeight || window.innerHeight || 0),
        },
      },
    };
  }

  _scheduleSurfaceSync({ immediate = false } = {}) {
    if (!this._surface?.sessionId || !this._overlay) return;
    if (this._surfaceTimer) clearTimeout(this._surfaceTimer);
    this._surfaceTimer = setTimeout(() => {
      this._surfaceTimer = null;
      this._pushSurfaceSync();
    }, immediate ? 0 : 220);
  }

  async _pushSurfaceSync() {
    const surface = this._surface;
    if (!surface?.sessionId || !this._overlay) return;
    try {
      const data = await patchSurfaceState(surface.sessionId, {
        patch: this._surfaceStatePatch(),
        sourceParticipantId: surface.participantId,
      });
      if (data?.session?.revision != null && this._surface === surface) {
        surface.revision = data.session.revision;
      }
    } catch (err) {
      console.warn('[comic-reader] surface sync failed:', err);
    }
  }

  /**
   * Cast the current comic to a paired Augmentum TV via the cast-picker.
   *
   * Routes through openCastPicker (the standard cast affordance used by
   * every content surface — browse video, audio player, image viewer)
   * so the user sees one consistent picker UX no matter where they cast
   * from. Sends the cast-comic surface URL with the file id; cast-comic
   * fetches metadata server-side and picks up the user's last-read
   * page from there (the comic-reader writes progress on every page
   * turn, so by the time the user hits Cast the server side is fresh).
   *
   * The legacy ``_startSurfaceHandoff`` path (Google Cast SDK +
   * Bluetooth surface session) is preserved for users who target
   * Chromecast-Premium / Bluetooth-paired surfaces — they can wire a
   * separate trigger or fall back to it explicitly. The default Cast
   * button now routes through the augmentum substrate which is what
   * every other surface in the app uses.
   */
  // Voice Cast editor — tapping Listen expands a panel to choose the active
  // cast (five register buckets the reader casts each line into) and start
  // narration. Casts are a reusable, server-persisted library, so a cast
  // defined once follows the user across comics, sessions, and devices. Never
  // auto-picks: slots default to "Default voice" until the user chooses.
  async _toggleNarrationVoices(anchor) {
    if (!this._file?.id) {
      showToast('No comic loaded — open a chapter first.', 'info', 2000);
      return;
    }
    // Toggle closed if already open.
    const open = this._overlay?.querySelector('.comic-reader-voice-panel');
    if (open) { open.remove(); return; }
    if (this._narrationBusy) return;

    const ref = this._file.id;
    const panel = document.createElement('div');
    panel.className = 'comic-reader-voice-panel';
    panel.innerHTML = '<div class="comic-reader-voice-loading">Loading voices…</div>';
    (this._overlay || document.body).appendChild(panel);
    try {
      const r = anchor.getBoundingClientRect();
      panel.style.top = `${Math.round(r.bottom + 8)}px`;
      panel.style.right = `${Math.round(window.innerWidth - r.right)}px`;
    } catch { /* fixed-position fallback via CSS */ }

    const close = () => {
      panel.remove();
      document.removeEventListener('mousedown', onDoc, true);
      document.removeEventListener('keydown', onKey, true);
    };
    const onDoc = (e) => {
      if (!panel.contains(e.target) && e.target !== anchor && !anchor.contains(e.target)) close();
    };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('mousedown', onDoc, true);
    document.addEventListener('keydown', onKey, true);

    // Voices + the cast library + this comic's already-recorded cast.
    let voices = [];
    let cfg = {};
    let recordedCast = {};
    try {
      const mc = await import('../model-cache.js');
      voices = await mc.getVoices();
    } catch { /* no TTS provider — selects fall back to Default */ }
    try {
      cfg = await fetch('/api/config/ui').then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
    } catch { /* no server config */ }
    try {
      const st = await fetch(`/api/comic-narration/${encodeURIComponent(ref)}`)
        .then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
      recordedCast = (st && typeof st.voice_cast === 'object' && st.voice_cast) || {};
      if (st && st.voice) recordedCast.narrator = st.voice;
    } catch { /* first run — no server record yet */ }
    if (!panel.isConnected) return;   // closed while loading

    const s = getSettings?.() || {};
    const readerDefault = s.readerTtsVoice || s.voiceDefaultVoice || '';
    const newId = () => (crypto?.randomUUID?.() || `cast_${Date.now()}`);

    // Parse the library; seed a first cast (pre-filled with this comic's
    // recorded cast, if any) when empty. Slots start blank = Default voice.
    let casts = [];
    try { casts = JSON.parse(cfg.comicVoiceCasts || '[]'); } catch { /* ignore */ }
    if (!Array.isArray(casts)) casts = [];
    if (!casts.length) {
      casts = [{ id: newId(), name: 'My Cast', slots: { ...recordedCast } }];
    }
    let activeId = cfg.comicVoiceCastActive || casts[0].id;
    if (!casts.some((c) => c.id === activeId)) activeId = casts[0].id;

    const SLOTS = [
      ['narrator', 'Narrator'],
      ['m_low', 'Male · low'],
      ['m_high', 'Male · high'],
      ['f_low', 'Female · low'],
      ['f_high', 'Female · high'],
    ];
    const voiceOpts = (selected) => {
      let html = `<option value="">Default${readerDefault ? ' voice' : ''}</option>`;
      for (const v of (Array.isArray(voices) ? voices : [])) {
        const rawId = typeof v === 'string' ? v : (v.id || v.name || v.voice_id || '');
        if (!rawId) continue;
        const provId = (typeof v === 'object' && v.provider_id) ? v.provider_id : '';
        const val = provId ? `${provId}::${rawId}` : rawId;
        const label = typeof v === 'string' ? v : (v.name || v.id || rawId);
        html += `<option value="${escapeHtml(val)}"${val === selected ? ' selected' : ''}>${escapeHtml(label)}</option>`;
      }
      return html;
    };
    const activeCast = () => casts.find((c) => c.id === activeId) || casts[0];

    // Persist the cast library + active id server-side on EVERY mutation
    // (new / delete / rename / slot change / switch) — not just on Listen —
    // so a cast survives closing the panel and reloading. Debounced so the
    // rename keystrokes + rapid slot edits coalesce into one PUT.
    let _persistTimer = null;
    const persist = () => {
      if (_persistTimer) clearTimeout(_persistTimer);
      _persistTimer = setTimeout(() => {
        fetch('/api/config/ui', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            comicVoiceCasts: JSON.stringify(casts),
            comicVoiceCastActive: activeId,
          }),
        }).catch(() => { /* best-effort */ });
      }, 400);
    };

    const render = () => {
      const cast = activeCast();
      const slots = cast.slots || {};
      panel.innerHTML = `
        <div class="comic-reader-voice-title">Voice Cast</div>
        <div class="comic-reader-voice-castbar">
          <select class="comic-reader-voice-castsel">
            ${casts.map((c) => `<option value="${escapeHtml(c.id)}"${c.id === activeId ? ' selected' : ''}>${escapeHtml(c.name || 'Cast')}</option>`).join('')}
          </select>
          <button type="button" class="comic-reader-voice-new" title="New cast">＋</button>
          <button type="button" class="comic-reader-voice-del" title="Delete cast"${casts.length <= 1 ? ' disabled' : ''}>🗑</button>
        </div>
        <input class="comic-reader-voice-name" type="text" value="${escapeHtml(cast.name || '')}" placeholder="Cast name" />
        ${SLOTS.map(([key, label]) => `
          <label class="comic-reader-voice-row">
            <span>${label}</span>
            <select data-slot="${key}">${voiceOpts(slots[key] || '')}</select>
          </label>`).join('')}
        <button type="button" class="comic-reader-voice-go">Listen</button>`;

      // Slot edits mutate the active cast in place.
      panel.querySelectorAll('select[data-slot]').forEach((sel) => {
        sel.addEventListener('change', () => {
          const c = activeCast();
          c.slots = c.slots || {};
          c.slots[sel.dataset.slot] = sel.value;
          persist();
        });
      });
      panel.querySelector('.comic-reader-voice-name')?.addEventListener('input', (e) => {
        activeCast().name = e.target.value;
        persist();
      });
      panel.querySelector('.comic-reader-voice-castsel')?.addEventListener('change', (e) => {
        activeId = e.target.value; persist(); render();
      });
      panel.querySelector('.comic-reader-voice-new')?.addEventListener('click', () => {
        const c = { id: newId(), name: `Cast ${casts.length + 1}`, slots: {} };
        casts.push(c); activeId = c.id; persist(); render();
        panel.querySelector('.comic-reader-voice-name')?.focus();
      });
      panel.querySelector('.comic-reader-voice-del')?.addEventListener('click', () => {
        if (casts.length <= 1) return;
        casts = casts.filter((c) => c.id !== activeId);
        activeId = casts[0].id; persist(); render();
      });
      panel.querySelector('.comic-reader-voice-go')?.addEventListener('click', () => {
        const cast = activeCast();
        const slotsNow = cast.slots || {};
        // Persist the library + active id server-side (best-effort) so the cast
        // is reusable everywhere; the run proceeds regardless.
        fetch('/api/config/ui', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            comicVoiceCasts: JSON.stringify(casts),
            comicVoiceCastActive: activeId,
          }),
        }).catch(() => { /* best-effort */ });
        close();
        this._openNarration(anchor, {
          voice: slotsNow.narrator || '',
          voice_cast: {
            m_low: slotsNow.m_low || '', m_high: slotsNow.m_high || '',
            f_low: slotsNow.f_low || '', f_high: slotsNow.f_high || '',
          },
        });
      });
    };
    render();
  }

  // Listen — voiced motion-comic. Kicks off (or resumes) narration synthesis
  // on the server, polls progress, then mounts the pan-and-scan player over
  // the reader. Idempotent: a finished narration opens instantly.
  async _openNarration(anchor, voices = null) {
    if (!this._file?.id) {
      showToast('No comic loaded — open a chapter first.', 'info', 2000);
      return;
    }
    const ref = this._file.id;
    const btn = anchor || this._overlay?.querySelector('.comic-reader-listen');
    if (this._narrationBusy) return;
    this._narrationBusy = true;
    if (btn) btn.classList.add('is-busy');
    try {
      // Streaming: mount as soon as ≥1 page is ready, not when the whole
      // chapter finishes. Fetch current state first.
      let st = await fetch(`/api/comic-narration/${encodeURIComponent(ref)}`)
        .then((r) => (r.ok ? r.json() : { status: 'none', ready_pages: 0 }))
        .catch(() => ({ status: 'none', ready_pages: 0 }));

      // Cache off (while transcription quality is being tuned): a finished
      // narration is never replayed, it's re-read. Skipping the direction
      // prompt below is deliberate — there's nothing to choose between when
      // the old recording isn't a candidate.
      if (st.cache_enabled === false && st.status !== 'running' && st.status !== 'pending') {
        st = { status: 'none', ready_pages: 0 };
      }

      // A narration already on disk was recorded in ONE reading direction, and
      // that direction is baked into every line of it — an LTR pass over manga
      // reads the panels backwards, which sounds like scrambled dialogue
      // rather than like an error. If it doesn't match how the user is reading
      // now, say so and let them choose; never silently play the wrong one and
      // never silently spend a chapter re-recording.
      // Both sides are already resolved — this._dir through _resolveDir, and
      // st.reading_direction through the server's normalizer — so neither needs
      // an 'ltr' floor here. Adding one back would reintroduce a phantom
      // mismatch prompt whenever the install default is RTL.
      const dirNow = this._dir;
      if (st.ready_pages >= 1 && st.reading_direction && st.reading_direction !== dirNow) {
        const was = st.reading_direction === 'rtl' ? 'right-to-left' : 'left-to-right';
        const now = dirNow === 'rtl' ? 'right-to-left' : 'left-to-right';
        const choice = await new Promise((resolve) => {
          showChoiceToast(
            `This narration was read ${was}, but you're reading ${now}.`,
            [
              { label: `Re-record ${now}`, primary: true, onClick: () => resolve('rerecord') },
              { label: 'Play it anyway', onClick: () => resolve('asis') },
            ],
            { description: 'Re-recording reads the whole chapter again.', onDismiss: () => resolve(null) },
          );
        });
        if (choice === null) return;
        if (choice === 'rerecord') st = { status: 'none', ready_pages: 0 };
      }

      if (!(st.ready_pages >= 1)) {
        // Kick off synthesis unless it's already in flight.
        if (st.status !== 'running' && st.status !== 'pending') {
          const post = (extra = {}) => fetch('/api/comic-narration/begin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              comic_ref: ref,
              reading_direction: this._dir,
              // Read from where the user IS, wrapping to the start afterwards.
              // Opening to page 17 and waiting out sixteen pages you've
              // already read is the whole reason this is sent.
              start_page: Math.max(0, (this._page || 1) - 1),
              // Voice Cast from the editor. `voice` is the narrator / default
              // (falls back to the reader's TTS voice, like the EPUB read-aloud
              // bar); the register buckets swap in per line and fall back to
              // `voice` server-side when a slot is left as Default.
              voice: (voices?.voice) || (() => {
                const s = getSettings?.() || {};
                return s.readerTtsVoice || s.voiceDefaultVoice || '';
              })(),
              voice_cast: voices?.voice_cast || {},
              ...extra,
            }),
          });
          let begin = await post();
          if (!begin.ok) {
            const msg = await begin.json().catch(() => ({}));
            const detail = msg.detail;
            // The OCR sidecar is started by hand, so "it's not running" is a
            // routine state, not an error — offer the two real options instead
            // of a dead end. Reading whole pages still works, just worse.
            if (detail && detail.code === 'boxed_reading_unavailable') {
              const pick = await new Promise((resolve) => {
                showChoiceToast(detail.message, [
                  { label: 'I started it — retry', primary: true, onClick: () => resolve('retry') },
                  { label: 'Read whole pages anyway', onClick: () => resolve('degraded') },
                ], { onDismiss: () => resolve(null) });
              });
              if (pick === null) return;
              begin = await post(pick === 'degraded' ? { accept_degraded: true } : {});
            }
            if (!begin.ok) {
              const again = await begin.json().catch(() => ({}));
              const text = typeof again.detail === 'string'
                ? again.detail
                : (again.detail?.message || 'Couldn’t start narration.');
              showToast(text, 'error', 4000);
              return;
            }
          }
        }
        showToast('Narrating this comic — first page in a moment.', 'info', 3000);
        st = await this._pollUntilFirstPage(ref);
      }

      if (!st || !(st.ready_pages >= 1)) {
        showToast(
          st && st.status === 'failed'
            ? (st.error || 'Narration failed.')
            : 'Couldn’t start narration.',
          'error', 4000,
        );
        return;
      }
      this._mountNarrationBar(st, ref);
    } finally {
      this._narrationBusy = false;
      if (btn) btn.classList.remove('is-busy');
    }
  }

  async _pollUntilFirstPage(ref, { intervalMs = 2000, timeoutMs = 600000 } = {}) {
    // Resolve as soon as page 1 is synthesized (ready_pages >= 1) — the player
    // streams the rest itself. Also returns on terminal failure.
    const started = Date.now();
    // eslint-disable-next-line no-constant-condition
    while (Date.now() - started < timeoutMs) {
      await new Promise((res) => setTimeout(res, intervalMs));
      const st = await fetch(`/api/comic-narration/${encodeURIComponent(ref)}`)
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null);
      if (!st) continue;
      if (st.ready_pages >= 1 || st.status === 'failed') return st;
      // Live progress on the Listen button's title.
      const btn = this._overlay?.querySelector('.comic-reader-listen');
      if (btn && st.total_pages) {
        btn.title = `Narrating… page ${st.processed_pages || 0}/${st.total_pages}`;
      }
    }
    return { status: 'timeout', ready_pages: 0 };
  }

  /* Narration docks INTO the reader rather than mounting over it. The reader
   * keeps the page image, the page cursor and every navigation control; the
   * bar only plays the audio for whichever page the reader is showing. See
   * narration-bar.js for why the old takeover overlay was the wrong model
   * (it desynced the page cursor and took navigation away from the user). */
  _mountNarrationBar(payload, ref) {
    this._unmountNarration();
    this._narrationBar = mountNarrationBar(this._overlay || document.body, payload, {
      // Polled for newly-synthesized pages while the user reads.
      pollUrl: `/api/comic-narration/${encodeURIComponent(ref)}`,
      // Auto-turn asks the READER to turn — the bar never moves the cursor
      // itself, so there stays exactly one source of truth for the page.
      requestPage: (page1) => this._goTo(page1),
      // Webtoon is a continuous scroll, so crossing a page boundary is a
      // position, not a decision — the bar queues the turn and lets the
      // current page finish speaking. Paged/dual turn on a deliberate act,
      // where switching immediately is the expected response. Read as a
      // callback (not a captured value) because W toggles the mode while the
      // narration bar is still mounted.
      isContinuous: () => this._mode === 'webtoon',
      onClose: () => this._unmountNarration(),
    });
    this._narrationBar?.start(this._page);
  }

  _unmountNarration() {
    try { this._narrationBar?.destroy(); } catch { /* noop */ }
    this._narrationBar = null;
  }

  async _castToTV(anchor) {
    if (!this._file?.id) {
      showToast('No comic loaded — open a chapter first.', 'info', 2000);
      return;
    }
    openCastPicker({
      anchor: anchor || this._overlay,
      capability: 'display.web_show@1',
      content: {
        contentUrl: `/ui/cast-comic/?id=${encodeURIComponent(this._file.id)}`,
        title: this._meta?.title || 'Comic',
        posterUrl: this._meta?.coverUrl || '',
        contentKey: `comic:${this._file.id}`,
        fileId: this._file.id,
        metadata: {
          currentPage: this._page,
          pageCount: this._pageCount,
          mode: this._mode,
          direction: this._dir,
          source: 'comic-reader',
        },
      },
    });
  }

  async _startSurfaceHandoff(options = {}) {
    if (!this._file?.id) return null;
    const button = this._overlay?.querySelector('.comic-reader-surface-toggle');
    const wasDisabled = !!button?.disabled;
    if (button) button.disabled = true;

    let bluetoothTarget = null;
    let bluetoothAttempted = false;
    let castAttempted = false;
    try {
      if (options.bluetooth === true && bluetoothHandoffAvailable()) {
        bluetoothAttempted = true;
        try {
          bluetoothTarget = await requestSurfaceBluetoothTarget({
            namePrefix: options.namePrefix || 'Augmentum',
            acceptAllDevices: !!options.acceptAllDevices,
          });
        } catch (err) {
          if (isBluetoothHandoffBlockedError(err)) {
            rememberBluetoothHandoffBlocked();
            bluetoothAttempted = false;
          }
          console.info('[comic-reader] Bluetooth surface target unavailable:', err);
        }
      }

      showToast('Preparing TV surface...', 'info', 1800);
      const result = await createComicSurfaceHandoff({
        file: this._file,
        page: this._page,
        pageCount: this._pageCount,
        scrollRatio: this._surfaceScrollRatio(),
        mode: this._mode,
        direction: this._dir,
        targetLabel: options.targetLabel || 'TV',
        targetIp: options.targetIp || '',
        bluetoothMtu: options.bluetoothMtu || 185,
      });
      const session = result.session || {};
      this._surface = {
        sessionId: session.id,
        participantId: result.participantId,
        revision: session.revision ?? 0,
        handoff: result.handoff,
      };
      this._scheduleSurfaceSync({ immediate: true });

      if (options.cast !== false && surfaceCastConfigured(options)) {
        castAttempted = true;
        try {
          const sent = await sendHandoffOverCast(result.handoff, {
            receiverApplicationId: options.receiverApplicationId || options.castAppId || '',
            label: `${session.title || 'Comic'} on Augmentum`,
          });
          showToast(sent.receiverName ? `Casting to ${sent.receiverName}` : 'Casting to TV', 'success', 2600);
          return result;
        } catch (err) {
          console.info('[comic-reader] Cast surface target unavailable:', err);
        }
      }

      if (bluetoothTarget) {
        try {
          const sent = await sendHandoffOverBluetooth(result.handoff, { target: bluetoothTarget });
          showToast(sent.deviceName ? `TV linked: ${sent.deviceName}` : 'TV linked', 'success', 2500);
          return result;
        } catch (err) {
          console.warn('[comic-reader] Bluetooth handoff write failed:', err);
        }
      }

      try {
        await copySurfaceReceiverUrl(result.handoff);
        const suffix = bluetoothAttempted || castAttempted ? ' instead' : '';
        showToast(`TV receiver link copied${suffix}`, 'success', 3000);
      } catch {
        showToast('TV surface ready; receiver link is in AugmentumSurfaceHandoff.lastComic', 'info', 4200);
      }
      window.AugmentumSurfaceHandoff = window.AugmentumSurfaceHandoff || {};
      window.AugmentumSurfaceHandoff.lastComic = result;
      return result;
    } catch (err) {
      showToast(`TV handoff failed: ${err.message || err}`, 'error', 4500);
      return null;
    } finally {
      try { bluetoothTarget?.disconnect?.(); } catch {}
      if (button) button.disabled = wasDisabled;
    }
  }

  unmount() {
    if (!this._overlay) return;

    this._unmountNarration();

    // One last progress push on close, synchronous via sendBeacon so we
    // don't get cancelled by the navigation/teardown. Best-effort — if
    // the transport fails the last page-flip push already covered us.
    // Skip when the resume scroll hasn't landed yet — pushing the
    // transient ``this._page`` here would clobber the saved position
    // for users who minimize and reopen quickly (the bug the
    // ``_sawResumePage`` gate guards against).
    try {
      if (this._sawResumePage && this._pageCount && navigator.sendBeacon) {
        const body = new Blob([JSON.stringify({
          current_time_s: this._page,
          duration_s: this._pageCount,
          is_finished: this._page >= this._pageCount,
        })], { type: 'application/json' });
        navigator.sendBeacon(
          `/api/media/progress/${encodeURIComponent(this._file.id)}`,
          body,
        );
      }
    } catch { /* best-effort */ }

    this._clearIdleTimer();
    this._clearEdgeTimer();
    this._teardownWebtoon();
    if (this._boundKey) document.removeEventListener('keydown', this._boundKey);
    if (this._progressTimer) clearTimeout(this._progressTimer);
    if (this._boundSurfaceScroll && this._stage) {
      this._stage.removeEventListener('scroll', this._boundSurfaceScroll);
    }
    if (this._surfaceTimer) clearTimeout(this._surfaceTimer);
    this._surface = null;
    this._surfaceTimer = null;
    this._boundSurfaceScroll = null;
    // Pointer handlers live on the overlay and leave with it.
    this._pointers.clear();
    this._overlay.classList.remove('visible');
    // Wait for the CSS fade-out before removing the node. 180ms matches
    // the .visible transition so we don't orphan a half-faded overlay.
    const node = this._overlay;
    this._overlay = null;
    this._prefetchImgs.clear();
    setTimeout(() => {
      node.remove();
      document.body.classList.remove('comic-reader-open');
    }, 180);
  }

  _wireEvents() {
    const overlay = this._overlay;

    // Buttons
    overlay.querySelector('.comic-reader-close')
      .addEventListener('click', () => this.unmount());
    overlay.querySelector('.comic-reader-prev')
      .addEventListener('click', () => this._goPrev());
    overlay.querySelector('.comic-reader-next')
      .addEventListener('click', () => this._goNext());
    overlay.querySelector('.comic-reader-ch-prev')
      ?.addEventListener('click', () => this._goPrevChapter());
    overlay.querySelector('.comic-reader-ch-next')
      ?.addEventListener('click', () => this._goNextChapter());
    overlay.querySelector('.comic-reader-dir-toggle')
      .addEventListener('click', () => this._toggleDir());
    overlay.querySelector('.comic-reader-surface-toggle')
      ?.addEventListener('click', (e) => this._castToTV(e.currentTarget));
    overlay.querySelector('.comic-reader-fit-toggle')
      ?.addEventListener('click', () => this._cycleFit());
    overlay.querySelector('.comic-reader-mode-toggle')
      ?.addEventListener('click', () => this._toggleMode());
    overlay.querySelector('.comic-reader-listen')
      ?.addEventListener('click', (e) => this._toggleNarrationVoices(e.currentTarget));
    overlay.querySelector('.comic-reader-minimize')
      ?.addEventListener('click', () => this.minimize());
    overlay.querySelector('.comic-reader-settings-toggle')
      ?.addEventListener('click', () => this._toggleDrawer());
    overlay.querySelector('.comic-reader-drawer-close')
      ?.addEventListener('click', () => this._closeDrawer());
    overlay.querySelector('[data-zone="drawer-scrim"]')
      ?.addEventListener('click', () => this._closeDrawer());
    // Nav drawer (left) — title block trigger, close button, scrim.
    overlay.querySelector('[data-action="open-nav-drawer"]')
      ?.addEventListener('click', () => this._toggleNavDrawer());
    overlay.querySelector('.comic-reader-nav-close')
      ?.addEventListener('click', () => this._closeNavDrawer());
    overlay.querySelector('[data-zone="nav-drawer-scrim"]')
      ?.addEventListener('click', () => this._closeNavDrawer());
    // Delegated segment/swatch clicks inside the drawer. One handler
    // covers mode / direction / fit / bg — data-attributes route to
    // the right setter so adding more groups in the future is a
    // one-line addition here plus a case in the switch below.
    overlay.querySelector('.comic-reader-drawer')
      ?.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-drawer-group]');
        if (!btn || btn.disabled) return;
        const group = btn.dataset.drawerGroup;
        const value = btn.dataset.drawerValue;
        switch (group) {
          case 'mode': this._setMode(value); break;
          case 'dir':  this._setDir(value); break;
          case 'fit':  this._setFit(value); break;
          case 'bg':   this._setBg(value); break;
          case 'crop': this._setCrop(!this._cropBorders); break;
          case 'autoscroll': this._setAutoScroll(value); break;
        }
      });
    overlay.querySelector('.comic-reader-retry')
      .addEventListener('click', () => {
        this._hideError();
        if (this._mode === 'webtoon') {
          // If we're still missing a page_count, _renderWebtoon would
          // just loop on the loading spinner. Re-fire the manifest
          // refresh so the upstream provider gets another shot at
          // preparing the chapter.
          if ((this._pageCount || 0) <= 0) {
            this._showLoading();
            this._refreshManifest();
          } else {
            this._renderWebtoon();
          }
        } else {
          this._loadPage(this._page);
        }
      });

    // Tap on image center: toggle chrome. Tap on left/right thirds:
    // navigate. Touch-friendly equivalent of keyboard nav. Swipes suppress
    // the click once via _suppressNextClick so a drag doesn't double-fire
    // into a tap zone after the swipe handler already navigated.
    this._stage.addEventListener('click', (e) => {
      if (this._suppressNextClick) {
        this._suppressNextClick = false;
        return;
      }
      // Zoomed: tap pans via pointer drag, no tap-zone nav. Prevents a
      // quick drag-release from landing on the wrong page.
      if (this._zoom.scale > 1.01) return;
      const rect = this._stage.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const third = rect.width / 3;
      if (x < third)            this._goPrev();
      else if (x > 2 * third)   this._goNext();
      else                      this._toggleChrome();
    });

    // Double-click / double-tap: toggle zoom between 1.0 and 2.5, focused
    // on the tap point. Matches Apple Photos / Instagram convention.
    this._stage.addEventListener('dblclick', (e) => {
      if (this._mode === 'webtoon') return;
      const rect = this._stage.getBoundingClientRect();
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      if (this._zoom.scale > 1.01) this._resetZoom();
      else this._setZoomAround(2.5, cx, cy);
    });

    // Ctrl/⌘ + wheel: desktop zoom, focused on cursor. Bare wheel is
    // intentionally left alone — in paged mode it does nothing (a brief
    // nudge not worth a scale change); in webtoon it drives native
    // scrolling, which is the mode's whole point.
    this._stage.addEventListener('wheel', (e) => {
      if (this._mode === 'webtoon') return;
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const rect = this._stage.getBoundingClientRect();
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      // deltaY < 0 = wheel up = zoom in. Small exponential step so
      // repeated ticks feel smooth without overshooting.
      const factor = Math.exp(-e.deltaY * 0.0015);
      const next = Math.max(1, Math.min(5, this._zoom.scale * factor));
      this._setZoomAround(next, cx, cy);
    }, { passive: false });

    this._boundSurfaceScroll = () => this._scheduleSurfaceSync();
    this._stage.addEventListener('scroll', this._boundSurfaceScroll, { passive: true });

    // Auto-scroll hold-to-pause: a finger resting on the strip pauses
    // the glide (so reading a dense panel or native touch-scrolling
    // doesn't fight the engine); lifting it resumes. Touch/pen only —
    // mouse users park the cursor over the page constantly, and their
    // wheel input composes fine with the per-frame scrollBy.
    const holdStart = (e) => {
      if (e.pointerType !== 'touch' && e.pointerType !== 'pen') return;
      this._autoScrollHeld = true;
    };
    const holdEnd = () => { this._autoScrollHeld = false; };
    this._stage.addEventListener('pointerdown', holdStart, { passive: true });
    this._stage.addEventListener('pointerup', holdEnd, { passive: true });
    this._stage.addEventListener('pointercancel', holdEnd, { passive: true });

    // Keyboard
    this._boundKey = (e) => this._onKey(e);
    document.addEventListener('keydown', this._boundKey);

    // Pointer gestures — unified swipe / pinch / pan. Touch + pen drive
    // the gesture layer; mouse swipes aren't a convention so we let the
    // click handler above carry mouse nav. Wiring is on the overlay so
    // fingers that drift off the image (onto letterbox) keep tracking.
    this._wirePointerGestures();
    this._wireScrubber();

    // Idle chrome auto-hide: any pointer movement on the overlay
    // reveals the chrome, and the timer re-arms. Respects the initial
    // "just opened" moment — chrome stays visible for 2s at mount.
    overlay.addEventListener('mousemove', () => this._revealChrome());
    overlay.addEventListener('touchstart', () => this._revealChrome(), { passive: true });
    this._revealChrome();
  }

  _onKey(e) {
    // Bail on any modifier combination — respects user's browser shortcuts
    // (Ctrl+F etc) and avoids eating Cmd+A copies in case users accidentally
    // trigger the reader with text focus.
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    // Escape closes the nav drawer first, then settings drawer, then
    // the reader — so a user in any drawer doesn't lose their reading
    // position to one press.
    if (e.key === 'Escape') {
      e.preventDefault();
      if (this._overlay?.classList.contains('nav-drawer-open')) this._closeNavDrawer();
      else if (this._overlay?.classList.contains('drawer-open')) this._closeDrawer();
      else this.unmount();
    }
    else if (e.key === 'ArrowRight')          { e.preventDefault(); this._dirForward(); }
    else if (e.key === 'ArrowLeft')           { e.preventDefault(); this._dirBackward(); }
    // Vertical nav keys make sense in webtoon (where scrolling IS the
    // primitive). In paged mode they'd be confusing — leave them alone so
    // the browser's default (nothing) runs.
    else if (e.key === 'ArrowDown' && this._mode === 'webtoon') { e.preventDefault(); this._goNext(); }
    else if (e.key === 'ArrowUp'   && this._mode === 'webtoon') { e.preventDefault(); this._goPrev(); }
    else if (e.key === 'PageDown')             { e.preventDefault(); this._goNext(); }
    else if (e.key === 'PageUp')               { e.preventDefault(); this._goPrev(); }
    else if (e.key === ' ')                   { e.preventDefault(); this._goNext(); }
    else if (e.key === 'Home')                { e.preventDefault(); this._goTo(1); }
    else if (e.key === 'End' && this._pageCount) {
      e.preventDefault(); this._goTo(this._pageCount);
    }
    else if (e.key === 'r' || e.key === 'R')  { e.preventDefault(); this._toggleDir(); }
    else if (e.key === 'f' || e.key === 'F')  { e.preventDefault(); this._cycleFit(); }
    else if (e.key === 'w' || e.key === 'W')  { e.preventDefault(); this._toggleMode(); }
    else if (e.key === 's' || e.key === 'S')  { e.preventDefault(); this._toggleDrawer(); }
    else if (e.key === 'l' || e.key === 'L')  { e.preventDefault(); this._toggleNavDrawer(); }
    else if (e.key === 'm' || e.key === 'M')  { e.preventDefault(); this.minimize(); }
    else if (e.key === 'c' || e.key === 'C')  { e.preventDefault(); this._setCrop(!this._cropBorders); }
    else if (e.key === 'a' || e.key === 'A')  { e.preventDefault(); this._cycleAutoScroll(); }
    else if (e.key === '?')                    { e.preventDefault(); this._openDrawer(); /* drawer has shortcuts section */ }
    // Chapter navigation — [ and ] feel natural next to the arrow keys
    // and match the convention from Tachiyomi / Mihon.
    else if (e.key === '[')                    { e.preventDefault(); this._goPrevChapter(); }
    else if (e.key === ']')                    { e.preventDefault(); this._goNextChapter(); }
  }

  _dirForward() {
    // Right-arrow physically means "forward in reading order" — which is
    // "next" for LTR and "previous" for RTL. Mirrors how physical manga
    // pages work: you flip to the right to read the previous page.
    if (this._dir === 'rtl') this._goPrev();
    else                     this._goNext();
  }

  _dirBackward() {
    if (this._dir === 'rtl') this._goNext();
    else                     this._goPrev();
  }

  _goNext() {
    // Webtoon nav is scroll-driven — a key press advances ~one viewport
    // rather than forcing a whole-page jump (which on manhwa is often
    // 2-3 visual panels). Edge-wrap still fires when we're already at
    // the bottom so the two-press chapter handoff works.
    if (this._mode === 'webtoon') {
      if (this._isWebtoonAtEnd()) return this._armOrFireEdge('end');
      this._edgeFlag = 'none';
      this._webtoonScrollBy(1);
      return;
    }
    // Dual mode shows pairs like (38, 39), so the "last rendered page" is
    // _page + 1 (except for the cover). Comparing _page alone against
    // pageCount misses the odd-count case: pageCount=39, _page=38 renders
    // (38, 39) but _page < pageCount, so _goNext tries to advance to 40
    // and _goTo blocks it — leaving the user stuck with no edge-arm.
    if (this._pageCount > 0) {
      const rightmost = this._mode === 'dual'
        ? (this._page === 1 ? 1 : this._page + 1)
        : this._page;
      if (rightmost >= this._pageCount) return this._armOrFireEdge('end');
    }
    this._edgeFlag = 'none';
    this._goTo(this._page + this._pageStep(true));
  }

  _goPrev() {
    if (this._mode === 'webtoon') {
      if (this._isWebtoonAtStart()) return this._armOrFireEdge('start');
      this._edgeFlag = 'none';
      this._webtoonScrollBy(-1);
      return;
    }
    // Dual mode: the "leftmost" shown page is _page (or 1 for the cover).
    // page 1 = cover; page 2 = pair (2,3). Prev from page 2 lands on 1.
    // Arming here only when _page <= 1 (mirrors paged).
    if (this._page <= 1) return this._armOrFireEdge('start');
    this._edgeFlag = 'none';
    this._goTo(this._page - this._pageStep(false));
  }

  _isWebtoonAtEnd() {
    const s = this._stage;
    if (!s) return false;
    return (s.scrollTop + s.clientHeight) >= (s.scrollHeight - 4);
  }
  _isWebtoonAtStart() {
    return (this._stage?.scrollTop ?? 0) <= 4;
  }
  /** Scroll the webtoon stage by ~90% of its viewport. Leaves a little
   *  overlap so the reader can see where they were — matches how PDF
   *  viewers handle Space/PageDown. */
  _webtoonScrollBy(direction) {
    const s = this._stage;
    if (!s) return;
    const step = Math.max(120, Math.round(s.clientHeight * 0.9));
    s.scrollBy({ top: direction * step, behavior: 'smooth' });
  }
  _webtoonScrollToPage(page) {
    const target = this._webtoonPages[page - 1];
    if (!target) return;
    target.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }

  // --- Scrubber -----------------------------------------------------
  //
  // The progress bar doubles as a slider. Drag (mouse / touch / pen) to
  // scrub through pages; a floating thumbnail preview follows the pointer
  // and shows the page at that position. Release commits via _goTo.
  // Keyboard arrows on the focused bar also step one page at a time.
  //
  // Thumbnails use the same /comic/page proxy with ``thumb=1`` so
  // Komga can serve its lighter per-page thumbnails while Suwayomi
  // just falls back to the normal page image. A 180ms idle debounce
  // prevents thrashing the upstream provider while the user is
  // dragging rapidly.

  _wireScrubber() {
    const bar = this._overlay?.querySelector('[data-zone="scrubber"]');
    if (!bar) return;
    const preview = this._overlay.querySelector('[data-zone="scrubber-preview"]');
    const previewImg = preview?.querySelector('.comic-reader-scrubber-preview-img');
    const previewPage = preview?.querySelector('.comic-reader-scrubber-preview-page');
    const previewTotal = preview?.querySelector('.comic-reader-scrubber-preview-total');

    let active = false;
    let hoverPage = this._page;
    let loadDebounce = null;
    let loadedPage = -1;

    const fractionFromX = (clientX) => {
      const rect = bar.getBoundingClientRect();
      const frac = (clientX - rect.left) / Math.max(1, rect.width);
      return Math.max(0, Math.min(1, frac));
    };
    const pageFromFraction = (frac) => {
      const total = this._pageCount || 1;
      if (total <= 1) return 1;
      return Math.max(1, Math.min(total, Math.round(frac * (total - 1)) + 1));
    };
    const positionPreview = (clientX) => {
      if (!preview) return;
      // Position preview above the bar, centered on the cursor. Clamp to
      // the viewport so the card never clips on edge-of-screen drags.
      const rect = bar.getBoundingClientRect();
      const pv = preview.getBoundingClientRect();
      const halfW = (pv.width || 180) / 2;
      const targetX = Math.max(8, Math.min(
        window.innerWidth - (pv.width || 180) - 8,
        clientX - halfW,
      ));
      const targetY = Math.max(8, rect.top - (pv.height || 220) - 12);
      preview.style.left = `${targetX}px`;
      preview.style.top = `${targetY}px`;
    };
    const scheduleThumb = (page) => {
      if (!previewImg) return;
      clearTimeout(loadDebounce);
      loadDebounce = setTimeout(() => {
        if (!active) return;
        if (page === loadedPage) return;
        loadedPage = page;
        // Use the same page endpoint; CSS scales the image. If the page
        // image is already decoded in the preload pool (or HTTP cache),
        // this is a free render.
        previewImg.src = _pageUrl(this._file.id, page, { thumb: true });
      }, 120);
    };
    const updateForClientX = (clientX) => {
      const frac = fractionFromX(clientX);
      const page = pageFromFraction(frac);
      hoverPage = page;
      if (previewPage) previewPage.textContent = String(page);
      if (previewTotal) previewTotal.textContent = String(this._pageCount || '?');
      positionPreview(clientX);
      scheduleThumb(page);
      // Update the visible fill + aria so the user sees the knob track
      // their finger even before they release. _updatePageChrome resets
      // this to the real page on cancel / commit.
      const fillPct = this._pageCount > 1
        ? ((page - 1) / (this._pageCount - 1)) * 100
        : 0;
      const fill = this._overlay.querySelector('.comic-reader-progress-fill');
      if (fill) fill.style.width = `${fillPct}%`;
      bar.style.setProperty('--scrubber-handle-pct', `${fillPct}%`);
      bar.setAttribute('aria-valuenow', String(page));
    };
    const start = (e) => {
      if (!this._pageCount || this._pageCount <= 1) return;
      active = true;
      bar.setPointerCapture?.(e.pointerId);
      bar.classList.add('scrubbing');
      preview?.removeAttribute('hidden');
      loadedPage = -1;
      // Immediate thumb for the starting position — the 120ms debounce
      // only kicks in once the user starts dragging.
      clearTimeout(loadDebounce);
      if (previewImg) {
        const p = pageFromFraction(fractionFromX(e.clientX));
        previewImg.src = _pageUrl(this._file.id, p, { thumb: true });
        loadedPage = p;
      }
      updateForClientX(e.clientX);
      e.preventDefault();
    };
    const move = (e) => {
      if (!active) return;
      updateForClientX(e.clientX);
    };
    const finish = (e) => {
      if (!active) return;
      active = false;
      bar.releasePointerCapture?.(e.pointerId);
      bar.classList.remove('scrubbing');
      preview?.setAttribute('hidden', '');
      clearTimeout(loadDebounce);
      if (hoverPage && hoverPage !== this._page) {
        // Commit — chrome chrome + progress will re-sync via _goTo.
        this._goTo(hoverPage);
      } else {
        // No-op commit — restore the bar to reflect actual page since
        // we mutated fill width during drag.
        this._updatePageChrome();
      }
    };
    const cancel = () => {
      if (!active) return;
      active = false;
      bar.classList.remove('scrubbing');
      preview?.setAttribute('hidden', '');
      clearTimeout(loadDebounce);
      this._updatePageChrome();  // snap back to real page
    };

    bar.addEventListener('pointerdown', start);
    bar.addEventListener('pointermove', move);
    bar.addEventListener('pointerup', finish);
    bar.addEventListener('pointercancel', cancel);

    // Keyboard: left/right on the focused bar = one-page step. Stop
    // propagation so the reader's global handler doesn't also fire.
    bar.addEventListener('keydown', (e) => {
      if (!this._pageCount) return;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        e.stopPropagation();
        const delta = e.key === 'ArrowRight' ? 1 : -1;
        const next = Math.max(1, Math.min(this._pageCount, this._page + delta));
        this._goTo(next);
      } else if (e.key === 'Home') {
        e.preventDefault(); e.stopPropagation();
        this._goTo(1);
      } else if (e.key === 'End') {
        e.preventDefault(); e.stopPropagation();
        this._goTo(this._pageCount);
      }
    });
  }

  // --- Zoom + pointer gestures ---------------------------------------
  //
  // Pinch-zoom is paged/dual only — webtoon's native vertical scroll is
  // the mode's whole point and we don't want to fight it. Transforms go
  // on the zoom target (single img or the dual wrapper) with default
  // transform-origin (center), using the formula
  //     tx' = f*tx + (1-f)*cx    with f = newScale / oldScale
  // to keep the pinch/wheel focal point pinned in screen space while the
  // image scales around it.

  _zoomTarget() {
    if (this._mode === 'dual') {
      return this._overlay?.querySelector('[data-zone="dual"]');
    }
    return this._img;
  }

  _applyZoomTransform() {
    const el = this._zoomTarget();
    if (!el) return;
    const { scale, tx, ty } = this._zoom;
    if (scale === 1 && tx === 0 && ty === 0) {
      el.style.transform = '';
    } else {
      el.style.transform = `translate(${tx.toFixed(1)}px, ${ty.toFixed(1)}px) scale(${scale.toFixed(3)})`;
    }
    // Transition off during active gesture so pinch/pan tracks 1:1 with
    // the finger; we re-enable it when the gesture ends (on reset the
    // CSS default handles the snap-back fade).
    el.style.transition = (this._gesture === 'pinch' || this._gesture === 'pan')
      ? 'none' : '';
    this._overlay?.classList.toggle('is-zoomed', scale > 1.01);
  }

  _resetZoom() {
    this._zoom = { scale: 1, tx: 0, ty: 0 };
    const el = this._zoomTarget();
    if (el) {
      el.style.transition = '';
      el.style.transform = '';
    }
    this._overlay?.classList.remove('is-zoomed');
  }

  _setZoomAround(nextScale, cx, cy) {
    const clamped = Math.max(1, Math.min(5, nextScale));
    const f = clamped / this._zoom.scale;
    this._zoom.tx = f * this._zoom.tx + (1 - f) * cx;
    this._zoom.ty = f * this._zoom.ty + (1 - f) * cy;
    this._zoom.scale = clamped;
    if (clamped <= 1.001) {
      // Snap to clean 1.0 so _resetZoom reliably removes the transform
      // and CSS defaults take over (no lingering sub-pixel offset).
      this._resetZoom();
    } else {
      this._applyZoomTransform();
    }
  }

  _wirePointerGestures() {
    const overlay = this._overlay;
    if (!overlay) return;
    const pointers = this._pointers;

    const stageRect = () => this._stage.getBoundingClientRect();
    const centerFrame = (clientX, clientY) => {
      const r = stageRect();
      return { x: clientX - r.left - r.width / 2, y: clientY - r.top - r.height / 2 };
    };

    const isGestureTarget = (el) => {
      // Chrome / drawer / buttons handle their own pointer semantics.
      // Everything inside the stage is fair game for gestures.
      if (!el) return false;
      if (el.closest('.comic-reader-chrome')) return false;
      if (el.closest('.comic-reader-drawer')) return false;
      if (el.closest('.comic-reader-drawer-scrim')) return false;
      return true;
    };

    overlay.addEventListener('pointerdown', (e) => {
      // Gesture layer is for touch + pen. Mouse nav goes through click
      // (tap zones) + wheel (zoom). Letting mouse pointerdown start a
      // pan/swipe would surprise users accustomed to tap-zone flow.
      if (e.pointerType !== 'touch' && e.pointerType !== 'pen') return;
      if (!isGestureTarget(e.target)) return;
      if (this._mode === 'webtoon') return;  // native scroll owns the primitive

      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      try { overlay.setPointerCapture?.(e.pointerId); } catch { /* older engines */ }

      if (pointers.size === 2) {
        // Second finger down — enter pinch. Cancel any pan/swipe state.
        const pts = [...pointers.values()];
        const dx = pts[1].x - pts[0].x;
        const dy = pts[1].y - pts[0].y;
        const mid = centerFrame((pts[0].x + pts[1].x) / 2, (pts[0].y + pts[1].y) / 2);
        this._pinchRef = {
          d0: Math.hypot(dx, dy) || 1,
          s0: this._zoom.scale,
          tx0: this._zoom.tx,
          ty0: this._zoom.ty,
          cx: mid.x,
          cy: mid.y,
        };
        this._gesture = 'pinch';
        this._suppressNextClick = true;
      } else if (pointers.size === 1) {
        if (this._zoom.scale > 1.01) {
          // Zoomed → drag = pan.
          this._gesture = 'pan';
          this._panRef = {
            x0: e.clientX, y0: e.clientY,
            tx0: this._zoom.tx, ty0: this._zoom.ty,
          };
        } else {
          // Unzoomed → drag = swipe (for page flip).
          this._gesture = 'swipe';
          this._swipeRef = { x0: e.clientX, y0: e.clientY, moved: false };
        }
      }
    });

    overlay.addEventListener('pointermove', (e) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (this._gesture === 'pinch' && pointers.size >= 2 && this._pinchRef) {
        const pts = [...pointers.values()].slice(0, 2);
        const dx = pts[1].x - pts[0].x;
        const dy = pts[1].y - pts[0].y;
        const dist = Math.hypot(dx, dy) || 1;
        const ratio = dist / this._pinchRef.d0;
        const newScale = Math.max(1, Math.min(5, this._pinchRef.s0 * ratio));
        const f = newScale / this._pinchRef.s0;
        this._zoom.scale = newScale;
        this._zoom.tx = f * this._pinchRef.tx0 + (1 - f) * this._pinchRef.cx;
        this._zoom.ty = f * this._pinchRef.ty0 + (1 - f) * this._pinchRef.cy;
        this._applyZoomTransform();
      } else if (this._gesture === 'pan' && this._panRef) {
        this._zoom.tx = this._panRef.tx0 + (e.clientX - this._panRef.x0);
        this._zoom.ty = this._panRef.ty0 + (e.clientY - this._panRef.y0);
        this._applyZoomTransform();
      } else if (this._gesture === 'swipe' && this._swipeRef) {
        const dx = e.clientX - this._swipeRef.x0;
        const dy = e.clientY - this._swipeRef.y0;
        if (Math.abs(dx) > 10 || Math.abs(dy) > 10) this._swipeRef.moved = true;
      }
    });

    const finishGesture = (e) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.delete(e.pointerId);
      try { overlay.releasePointerCapture?.(e.pointerId); } catch { /* no-op */ }

      if (this._gesture === 'swipe' && this._swipeRef && pointers.size === 0) {
        const dx = e.clientX - this._swipeRef.x0;
        const dy = e.clientY - this._swipeRef.y0;
        const adx = Math.abs(dx), ady = Math.abs(dy);
        if (adx > 40 && adx > ady) {
          // RTL: swipe right → next page (physical manga page flip).
          const swipedNext = this._dir === 'rtl' ? dx > 0 : dx < 0;
          if (swipedNext) this._goNext();
          else            this._goPrev();
          this._suppressNextClick = true;
        }
      } else if (this._gesture === 'pinch' && pointers.size < 2) {
        // Second finger lifted mid-pinch. If the remaining finger's still
        // down we continue as a pan (zoomed) or accept the scale as-is.
        if (this._zoom.scale <= 1.05) {
          this._resetZoom();
        }
        if (pointers.size === 1) {
          // Hand off to pan — seed the ref from the surviving pointer so
          // the user can keep dragging without a lift-and-retap.
          const remaining = [...pointers.values()][0];
          this._gesture = 'pan';
          this._panRef = {
            x0: remaining.x, y0: remaining.y,
            tx0: this._zoom.tx, ty0: this._zoom.ty,
          };
        }
      }

      if (pointers.size === 0) {
        this._gesture = 'none';
        this._pinchRef = null;
        this._panRef = null;
        this._swipeRef = null;
        // Re-enable CSS transition post-gesture for the next paint.
        this._applyZoomTransform();
      }
    };
    overlay.addEventListener('pointerup', finishGesture);
    overlay.addEventListener('pointercancel', finishGesture);
  }

  /** How many pages one navigation step advances. Dual-page mode pairs
   *  (2,3), (4,5)... with the cover on page 1 alone — so from page 1
   *  forward is +1 (to 2), then +2 steps thereafter. Symmetric for prev. */
  _pageStep(forward) {
    if (this._mode !== 'dual') return 1;
    if (forward) {
      // After the cover (page 1), all subsequent forward steps are +2
      return this._page === 1 ? 1 : 2;
    } else {
      // Back from 3 → 2, back from 4 → 2, back from 2 → 1, back from 1 → N/A
      // Effectively: snap-back to a pair's LEFT (lower) page. The pair
      // layout is (2,3), (4,5)..., so given page p ≥ 2 the "pair left"
      // is p if p is even, p-1 if p is odd. Step backward is one pair width.
      if (this._page <= 2) return 1;  // page 2 steps back to page 1 (cover)
      // step back to the previous pair's left page
      const currentPairLeft = this._page - (this._page % 2 === 0 ? 0 : 1);
      const prevPairLeft = currentPairLeft - 2;
      return this._page - prevPairLeft;
    }
  }

  /** End-of-chapter: forward fires immediately (the open-at-end guard in
   *  the constructor + manifest-driven freshness already prevent the
   *  "first press skips a chapter" trap that two-press was protecting
   *  against). Backward keeps two-press so an accidental prev at the
   *  top of a chapter doesn't rewind without confirmation. */
  _armOrFireEdge(edge) {
    const hasAdjacent = edge === 'end'
      ? this._hasNextChapter()
      : this._hasPrevChapter();
    if (!hasAdjacent) {
      // Nothing to wrap into — show a one-shot "you're at the boundary"
      // nudge so the user knows the button registered.
      showToast(
        edge === 'end' ? 'Last page of the last chapter.' : 'First page of the first chapter.',
        'info', 1400,
      );
      return;
    }
    if (edge === 'end') {
      // Auto-flow into the next chapter — natural reading shouldn't
      // require a confirmation tap. The transition itself fires a toast
      // with the new chapter title so the user has feedback.
      this._clearEdgeTimer();
      this._edgeFlag = 'none';
      this._goNextChapter();
      return;
    }
    if (this._edgeFlag === edge) {
      // Second press at the start edge — fire the backward transition.
      this._clearEdgeTimer();
      this._edgeFlag = 'none';
      this._goPrevChapter({ startAtEnd: true });
      return;
    }
    // First press at the start edge — arm + toast for confirmation.
    this._edgeFlag = edge;
    showToast(
      `Start of chapter — press again for ${this._prevChapterLabel() || 'previous chapter'}`,
      'info', 2200,
    );
    this._clearEdgeTimer();
    this._edgeFlagTimer = setTimeout(() => {
      this._edgeFlag = 'none';
      this._edgeFlagTimer = null;
    }, 3200);
  }

  _clearEdgeTimer() {
    if (this._edgeFlagTimer) {
      clearTimeout(this._edgeFlagTimer);
      this._edgeFlagTimer = null;
    }
  }

  // --- Navigation drawer (left side) ---------------------------------
  //
  // Library / chapters / series info / go-to / mark-as-read controls.
  // Mirrors the right-side settings drawer in geometry + animation but
  // owns navigation rather than reader preferences. Mutually exclusive
  // with the settings drawer — opening one closes the other so on phones
  // (where both consume 92vw) there is no impossible overlap.
  //
  // Sections are rendered as empty containers here and populated by
  // future steps (series info, chapter list, go-to, mark-as-read). The
  // scaffold ships first so each follow-up step is a tight diff.

  _renderNavDrawer() {
    return `
      <aside class="comic-reader-nav-drawer" data-zone="nav-drawer"
             aria-hidden="true" aria-label="Reader library">
        <header class="comic-reader-nav-header">
          <h2>Library</h2>
          <button type="button" class="comic-reader-btn comic-reader-nav-close"
                  aria-label="Close library">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </header>
        <section class="comic-reader-nav-series" data-zone="nav-series"></section>
        <section class="comic-reader-nav-goto"   data-zone="nav-goto"></section>
        <section class="comic-reader-nav-chapters" data-zone="nav-chapters"></section>
      </aside>
      <div class="comic-reader-nav-drawer-scrim" data-zone="nav-drawer-scrim"
           aria-hidden="true"></div>
    `;
  }

  _openNavDrawer() {
    if (!this._overlay) return;
    // Mutual exclusion: never have both drawers open simultaneously.
    if (this._overlay.classList.contains('drawer-open')) this._closeDrawer();
    this._overlay.classList.add('nav-drawer-open');
    const drawer = this._overlay.querySelector('.comic-reader-nav-drawer');
    drawer?.setAttribute('aria-hidden', 'false');
    const titleBtn = this._overlay.querySelector('.comic-reader-titles-btn');
    titleBtn?.setAttribute('aria-expanded', 'true');
    // Suppress chrome auto-hide while the drawer is open — same reasoning
    // as the settings drawer; the user is engaging with controls and
    // the ambient UI shouldn't collapse under them.
    this._clearIdleTimer();
  }

  _closeNavDrawer() {
    if (!this._overlay) return;
    this._overlay.classList.remove('nav-drawer-open');
    const drawer = this._overlay.querySelector('.comic-reader-nav-drawer');
    drawer?.setAttribute('aria-hidden', 'true');
    const titleBtn = this._overlay.querySelector('.comic-reader-titles-btn');
    titleBtn?.setAttribute('aria-expanded', 'false');
    this._revealChrome();
  }

  _toggleNavDrawer() {
    if (!this._overlay) return;
    if (this._overlay.classList.contains('nav-drawer-open')) this._closeNavDrawer();
    else                                                      this._openNavDrawer();
  }

  // --- Settings drawer + help overlay --------------------------------

  _renderSettingsDrawer() {
    const seriesLabel = this._meta?.title
      ? escapeHtml(this._meta.title)
      : 'this series';
    const dualSupported = this._supportsDualPage();
    const segProps = {
      mode: '_mode', fit: '_fit', dir: '_dir', bg: '_bg',
      autoscroll: '_autoScroll',
    };
    const seg = (group, value, label, extra = '') => `
      <button class="comic-reader-drawer-seg${this[segProps[group]] === value ? ' active' : ''}"
              data-drawer-group="${group}" data-drawer-value="${escapeHtml(value)}"
              type="button"${extra}>
        ${escapeHtml(label)}
      </button>
    `;
    return `
      <aside class="comic-reader-drawer" data-zone="drawer" aria-hidden="true"
             aria-label="Reader settings">
        <header class="comic-reader-drawer-header">
          <h2>Reader</h2>
          <button type="button" class="comic-reader-btn comic-reader-drawer-close"
                  aria-label="Close settings">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </header>
        <p class="comic-reader-drawer-note">
          Preferences save per series — they'll stick on ${seriesLabel}
          and carry across its chapters.
        </p>

        <section class="comic-reader-drawer-section">
          <h3>Reading mode</h3>
          <div class="comic-reader-drawer-segment">
            ${seg('mode', 'paged', 'Paged')}
            ${seg('mode', 'dual', 'Spread', dualSupported ? '' : ' disabled title="Viewport too narrow"')}
            ${seg('mode', 'webtoon', 'Webtoon')}
          </div>
        </section>

        <section class="comic-reader-drawer-section">
          <h3>Direction</h3>
          <div class="comic-reader-drawer-segment">
            ${seg('dir', 'ltr', 'Left → Right')}
            ${seg('dir', 'rtl', 'Right ← Left')}
          </div>
          <p class="comic-reader-drawer-hint">
            Manga is traditionally right-to-left; webtoons and Western comics are left-to-right.
          </p>
        </section>

        <section class="comic-reader-drawer-section">
          <h3>Fit</h3>
          <div class="comic-reader-drawer-segment">
            ${seg('fit', 'fit-page', 'Page')}
            ${seg('fit', 'fit-width', 'Width')}
            ${seg('fit', 'original', 'Original')}
          </div>
        </section>

        <section class="comic-reader-drawer-section">
          <h3>Auto-scroll</h3>
          <div class="comic-reader-drawer-segment">
            ${seg('autoscroll', 'off', 'Off')}
            ${seg('autoscroll', 'slow', 'Slow')}
            ${seg('autoscroll', 'medium', 'Med')}
            ${seg('autoscroll', 'fast', 'Fast')}
          </div>
          <p class="comic-reader-drawer-hint">
            Hands-free reading in Webtoon mode — the strip glides at a steady
            speed and rolls into the next chapter. Hold a finger on the page
            to pause; resumes on release.
          </p>
        </section>

        <section class="comic-reader-drawer-section">
          <h3>Page</h3>
          <button type="button"
                  class="comic-reader-drawer-toggle${this._cropBorders ? ' on' : ''}"
                  data-drawer-group="crop" data-drawer-value="toggle"
                  aria-pressed="${this._cropBorders ? 'true' : 'false'}">
            <span class="comic-reader-drawer-toggle-label">Trim white borders</span>
            <span class="comic-reader-drawer-toggle-pill" aria-hidden="true">
              <span class="comic-reader-drawer-toggle-knob"></span>
            </span>
          </button>
          <p class="comic-reader-drawer-hint">
            Detects and trims solid edges around scanned pages so content fills more of the viewport.
          </p>
        </section>

        <section class="comic-reader-drawer-section">
          <h3>Background</h3>
          <div class="comic-reader-drawer-swatches">
            <button class="comic-reader-drawer-swatch bg-swatch-dark${this._bg === 'dark' ? ' active' : ''}"
                    data-drawer-group="bg" data-drawer-value="dark" type="button"
                    aria-label="Dark"><span></span>Dark</button>
            <button class="comic-reader-drawer-swatch bg-swatch-black${this._bg === 'black' ? ' active' : ''}"
                    data-drawer-group="bg" data-drawer-value="black" type="button"
                    aria-label="Black"><span></span>Black</button>
            <button class="comic-reader-drawer-swatch bg-swatch-sepia${this._bg === 'sepia' ? ' active' : ''}"
                    data-drawer-group="bg" data-drawer-value="sepia" type="button"
                    aria-label="Sepia"><span></span>Sepia</button>
            <button class="comic-reader-drawer-swatch bg-swatch-paper${this._bg === 'paper' ? ' active' : ''}"
                    data-drawer-group="bg" data-drawer-value="paper" type="button"
                    aria-label="Paper"><span></span>Paper</button>
          </div>
        </section>

        <section class="comic-reader-drawer-section">
          <h3>Shortcuts</h3>
          <dl class="comic-reader-drawer-shortcuts">
            <dt><kbd>←</kbd> <kbd>→</kbd></dt><dd>Previous / next page</dd>
            <dt><kbd>Space</kbd> / <kbd>PgDn</kbd></dt><dd>Next page (scrolls in webtoon)</dd>
            <dt><kbd>PgUp</kbd></dt><dd>Previous page</dd>
            <dt><kbd>Home</kbd> <kbd>End</kbd></dt><dd>First / last page</dd>
            <dt><kbd>[</kbd> <kbd>]</kbd></dt><dd>Previous / next chapter</dd>
            <dt><kbd>F</kbd></dt><dd>Cycle fit mode</dd>
            <dt><kbd>W</kbd></dt><dd>Cycle reading mode</dd>
            <dt><kbd>R</kbd></dt><dd>Toggle direction</dd>
            <dt><kbd>C</kbd></dt><dd>Trim borders on / off</dd>
            <dt><kbd>A</kbd></dt><dd>Cycle auto-scroll speed (webtoon)</dd>
            <dt><kbd>L</kbd></dt><dd>Library — chapters &amp; series info</dd>
            <dt><kbd>S</kbd></dt><dd>Open settings</dd>
            <dt><kbd>?</kbd></dt><dd>Show shortcuts</dd>
            <dt><kbd>Esc</kbd></dt><dd>Close</dd>
          </dl>
        </section>
      </aside>
      <div class="comic-reader-drawer-scrim" data-zone="drawer-scrim" aria-hidden="true"></div>
    `;
  }

  _openDrawer() {
    if (!this._overlay) return;
    // Mutual exclusion: never have both drawers open simultaneously.
    if (this._overlay.classList.contains('nav-drawer-open')) this._closeNavDrawer();
    this._overlay.classList.add('drawer-open');
    const drawer = this._overlay.querySelector('.comic-reader-drawer');
    drawer?.setAttribute('aria-hidden', 'false');
    // Prevent chrome auto-hide while the drawer is open — the user is
    // actively fiddling with settings and doesn't want the ambient UI
    // collapsing under them.
    this._clearIdleTimer();
  }
  _closeDrawer() {
    if (!this._overlay) return;
    this._overlay.classList.remove('drawer-open');
    const drawer = this._overlay.querySelector('.comic-reader-drawer');
    drawer?.setAttribute('aria-hidden', 'true');
    this._revealChrome();  // re-arm chrome auto-hide
  }
  _toggleDrawer() {
    if (!this._overlay) return;
    if (this._overlay.classList.contains('drawer-open')) this._closeDrawer();
    else                                                  this._openDrawer();
  }

  /** Sync the drawer's active segment pills when a setting changes via
   *  any path (chrome toggle, keyboard shortcut, drawer click). Called
   *  by every `_setX` method so all three entry points stay consistent. */
  _updateDrawerActive(group, value) {
    const drawer = this._overlay?.querySelector('.comic-reader-drawer');
    if (!drawer) return;
    drawer.querySelectorAll(`[data-drawer-group="${group}"]`).forEach(el => {
      el.classList.toggle('active', el.dataset.drawerValue === value);
    });
  }

  _nextChapterLabel() {
    if (!this._hasNextChapter()) return '';
    const next = this._siblings[this._siblingIndex + 1];
    return _extractMeta(next).subtitle || 'next chapter';
  }
  _prevChapterLabel() {
    if (!this._hasPrevChapter()) return '';
    const prev = this._siblings[this._siblingIndex - 1];
    return _extractMeta(prev).subtitle || 'previous chapter';
  }

  _goPrevChapter({ startAtEnd = false } = {}) {
    if (!this._hasPrevChapter()) return;
    const prev = this._siblings[this._siblingIndex - 1];
    this._transitionToChapter(prev, { startAtEnd });
  }

  _goNextChapter() {
    if (!this._hasNextChapter()) return;
    const next = this._siblings[this._siblingIndex + 1];
    this._transitionToChapter(next, { startAtEnd: false });
  }

  _transitionToChapter(nextFile, { startAtEnd = false } = {}) {
    // Flush any pending progress push for the chapter we're LEAVING. The
    // debounce would normally fire eventually, but a quick flip-through
    // would drop the final state. Using sendBeacon keeps it synchronous.
    this._flushProgressSync();
    if (this._surface?.sessionId) {
      if (this._surfaceTimer) clearTimeout(this._surfaceTimer);
      this._surface = null;
      this._surfaceTimer = null;
      showToast('TV sync paused for the next chapter; send it to TV again when ready.', 'info', 2600);
    }
    // New chapter, fresh edge state — don't carry over the "armed at end"
    // flag from the previous chapter.
    this._edgeFlag = 'none';
    this._clearEdgeTimer();

    // Cancel any in-flight page load so its onload handler doesn't swap
    // the stale image into the new chapter's stage.
    this._loadingToken++;
    if (this._progressTimer) { clearTimeout(this._progressTimer); this._progressTimer = null; }
    this._teardownWebtoon();
    this._prefetchImgs.clear();
    this._cropCache.clear();   // dataURLs belong to the outgoing chapter
    this._nextChapterPrefetched = false;   // re-arm for the new chapter's end
    this._reachedContinuePoint = false;    // re-arm for the new chapter's end

    this._file = nextFile;
    this._meta = _extractMeta(nextFile);
    this._pageCount = this._meta.pageCount;
    this._page = startAtEnd && this._pageCount ? this._pageCount : 1;
    // Chapter transitions always land at page 1 (or the end, for
    // backward navigation). No resume scroll to wait for, so the gate
    // starts open.
    this._sawResumePage = true;
    // Same resolution as opening the reader (series → file → last explicit
    // choice). This used to read the per-FILE pref only, so turning to the next
    // chapter of a series set to RTL silently dropped it back to LTR — throwing
    // away a choice the user had made explicitly, on the series, moments ago.
    this._dir = _resolveDir(this._seriesId, nextFile.id);
    this._siblingIndex = this._findSiblingIndex(nextFile);

    // Re-render the chrome bits that carry chapter identity. Title,
    // subtitle, counter, progress, chapter-nav button disabled state.
    this._rerenderChapterChrome();

    if (this._mode === 'webtoon') this._renderWebtoon();
    else                          this._loadPage(this._page);

    // Same stale-metadata concerns apply to the chapter we're entering —
    // refresh its manifest so the first-press-on-open path doesn't hit
    // an edge arm on bad data.
    this._refreshManifest();
    // Mount-time-equivalent prefetch for the chapter AFTER this one.
    // _nextChapterPrefetched was reset above, so this re-arms for the
    // newly-current chapter's successor. Keeps the pipeline filled —
    // if the user keeps reading sequentially, every transition lands on
    // a chapter Suwayomi already prepared.
    this._prefetchNextChapter();

    // Gentle nudge so the user sees the new chapter loaded
    if (this._meta.subtitle) showToast(this._meta.subtitle, 'info', 1200);
  }

  _rerenderChapterChrome() {
    const o = this._overlay;
    if (!o) return;
    const m = this._meta;
    const titleEl = o.querySelector('.comic-reader-title');
    const subEl = o.querySelector('.comic-reader-subtitle');
    if (titleEl) titleEl.textContent = m.title;
    if (subEl) {
      subEl.textContent = m.subtitle || '';
      subEl.style.display = m.subtitle ? '' : 'none';
    } else if (m.subtitle) {
      // Previous chapter had no subtitle so the DOM node never rendered;
      // inject one so the new chapter's subtitle surfaces.
      const titles = o.querySelector('.comic-reader-titles');
      if (titles) {
        const div = document.createElement('div');
        div.className = 'comic-reader-subtitle';
        div.textContent = m.subtitle;
        titles.appendChild(div);
      }
    }
    const totalEl = o.querySelector('.comic-reader-page-total');
    if (totalEl) totalEl.textContent = this._pageCount || '?';
    // Direction change across chapters (if user set RTL per-file) reflects
    // in the overlay class and the dir label.
    o.classList.toggle('dir-rtl', this._dir === 'rtl');
    o.classList.toggle('dir-ltr', this._dir === 'ltr');
    const dirLabel = o.querySelector('.comic-reader-dir-label');
    if (dirLabel) dirLabel.textContent = this._dir === 'rtl' ? 'RTL' : 'LTR';
    // Chapter-nav button enable/disable
    const prevBtn = o.querySelector('.comic-reader-ch-prev');
    const nextBtn = o.querySelector('.comic-reader-ch-next');
    if (prevBtn) prevBtn.disabled = !this._hasPrevChapter();
    if (nextBtn) nextBtn.disabled = !this._hasNextChapter();
    const posEl = o.querySelector('.comic-reader-chapter-position');
    if (posEl) posEl.textContent = this._chapterPositionLabel();
    this._updatePageChrome();
  }

  _chapterPositionLabel() {
    if (!this._siblings || this._siblingIndex < 0) return '';
    return `Chapter ${this._siblingIndex + 1} of ${this._siblings.length}`;
  }

  _flushProgressSync() {
    // Force-push what we have, bypassing the debounce. sendBeacon
    // survives the page transition without waiting for a response.
    if (this._progressTimer) {
      clearTimeout(this._progressTimer);
      this._progressTimer = null;
    }
    if (!this._pageCount) return;
    try {
      if (navigator.sendBeacon) {
        const body = new Blob([JSON.stringify({
          current_time_s: this._page,
          duration_s: this._pageCount,
          is_finished: this._page >= this._pageCount,
        })], { type: 'application/json' });
        navigator.sendBeacon(
          `/api/media/progress/${encodeURIComponent(this._file.id)}`,
          body,
        );
      } else {
        _pushProgress(this._file.id, this._page, this._pageCount);
      }
    } catch { /* best-effort */ }
  }

  _cycleFit() {
    const i = _FIT_MODES.indexOf(this._fit);
    this._setFit(_FIT_MODES[(i + 1) % _FIT_MODES.length]);
    showToast(_FIT_LABELS[this._fit], 'info', 900);
  }

  _setFit(fit) {
    if (!_FIT_MODES.includes(fit)) return;
    this._fit = fit;
    _saveFit(fit);
    _patchSeriesPrefs(this._seriesId, { fit });
    // CSS hooks live at `.fit-page` / `.fit-width` / `.original` (the raw
    // mode token — note `original` is namespaced by the overlay class, not
    // by a `fit-` prefix). Don't double-prefix; the previous form produced
    // `fit-fit-page` / `fit-original` and matched nothing, leaving every
    // fit mode to fall back to the default `.comic-reader-img` rule that
    // CSS comments specifically call out as buggy ("half-clipped renders").
    for (const m of _FIT_MODES) this._overlay.classList.toggle(m, m === fit);
    const label = this._overlay.querySelector('.comic-reader-fit-label');
    if (label) label.textContent = _FIT_LABELS[fit];
    this._updateDrawerActive('fit', fit);
  }

  _toggleMode() {
    // Cycle: paged → dual → webtoon → paged. Dual is skipped on narrow
    // viewports where the spread wouldn't fit comfortably.
    const order = this._supportsDualPage() ? _READ_MODES : ['paged', 'webtoon'];
    const i = order.indexOf(this._mode);
    const next = order[(i + 1) % order.length];
    this._setMode(next);
    showToast(`${_MODE_LABELS[next]} mode`, 'info', 900);
  }

  _setMode(mode) {
    if (!_READ_MODES.includes(mode)) return;
    // Guard: dual-page requires a wide-enough viewport. Fall back to paged
    // with a toast so the user understands why their selection didn't
    // stick — respects the intent without silently swallowing a tap.
    if (mode === 'dual' && !this._supportsDualPage()) {
      mode = 'paged';
      showToast('Spread mode needs a wider viewport — using paged', 'warning', 1400);
    }
    this._mode = mode;
    _saveMode(mode);
    _patchSeriesPrefs(this._seriesId, { mode });
    // Zoom state belongs to a specific render target; switching modes
    // re-homes the transform. Reset before we swap classes so the old
    // target's inline transform gets cleared.
    this._resetZoom();
    for (const m of _READ_MODES) this._overlay.classList.toggle(`mode-${m}`, m === mode);
    const label = this._overlay.querySelector('.comic-reader-mode-label');
    if (label) label.textContent = _MODE_LABELS[mode];
    this._updateDrawerActive('mode', mode);
    // Re-render for the new mode. Webtoon builds the stacked strip;
    // dual + paged share the single-img stage but render differently
    // per _loadPage which is mode-aware.
    if (mode === 'webtoon') {
      this._renderWebtoon();
    } else {
      this._teardownWebtoon();
      this._loadPage(this._page);
    }
    this._scheduleSurfaceSync({ immediate: true });
  }

  /** Dual-page mode is only useful when two manga pages fit side-by-side.
   *  Anything under ~900px total stage width becomes cramped. Mobile
   *  phones and narrow panel windows collapse to single-page automatically. */
  _supportsDualPage() {
    const w = this._stage?.clientWidth || window.innerWidth || 0;
    return w >= 900;
  }

  _setBg(bg) {
    if (!_BG_MODES.includes(bg)) return;
    this._bg = bg;
    _saveBg(bg);
    _patchSeriesPrefs(this._seriesId, { background: bg });
    for (const b of _BG_MODES) this._overlay.classList.toggle(`bg-${b}`, b === bg);
    this._updateDrawerActive('bg', bg);
  }

  _setCrop(on) {
    this._cropBorders = !!on;
    _saveCrop(this._cropBorders);
    _patchSeriesPrefs(this._seriesId, { crop: this._cropBorders });
    // Reload whatever's showing so the effect lands immediately. Toggling
    // off clears the cache so we re-show originals (not stale crops).
    if (!this._cropBorders) this._cropCache.clear();
    const toggleBtn = this._overlay?.querySelector('[data-drawer-group="crop"]');
    if (toggleBtn) {
      toggleBtn.classList.toggle('on', this._cropBorders);
      toggleBtn.setAttribute('aria-pressed', this._cropBorders ? 'true' : 'false');
    }
    if (this._mode === 'webtoon') {
      this._renderWebtoon();
    } else {
      this._loadPage(this._page);
    }
    showToast(this._cropBorders ? 'Trimming borders' : 'Borders restored', 'info', 900);
  }

  _renderWebtoon() {
    // Build the stacked vertical feed. Every page renders full-width;
    // the stage scrolls. An IntersectionObserver updates _page to match
    // whichever page is currently dominant in the viewport, so progress
    // and the counter stay in sync with scroll position.
    const container = this._overlay.querySelector('.comic-reader-webtoon');
    if (!container) return;
    container.innerHTML = '';
    this._webtoonPages = [];
    this._teardownWebtoon({ keepContainer: true });

    const total = this._pageCount || 0;
    if (total <= 0) {
      // Webtoon needs a known page count so we can build the full strip.
      // Suwayomi / Komga populate this after the chapter is first
      // downloaded; brand-new or not-yet-decoded chapters arrive with
      // page_count=0. The manifest refresh that mount() / chapter
      // transition fires will populate it shortly, then re-call this
      // method — at which point we hit the strip-build branch below.
      //
      // Critical that we DON'T flip the overlay to mode-paged here: it
      // produces a display where the screen shows a single page but
      // the mode toggle still reads "Webtoon" (because this._mode is
      // unchanged). Users naturally click the toggle to "fix" the
      // mismatch, which actually CYCLES THEIR PREFERENCE OFF webtoon
      // and persists paged. Showing the loading spinner instead makes
      // the transient state read as "loading", not "wrong mode".
      this._showLoading();
      return;
    }
    this._hideLoading();
    // Reassert webtoon classes — a prior fallback (or chapter transition
    // following one) may have left `mode-paged` on the overlay, which
    // would hide the strip we're about to build. Without this, the user
    // sees a paged image until they manually toggle modes.
    this._overlay.classList.remove('mode-paged', 'mode-dual');
    this._overlay.classList.add('mode-webtoon');
    for (let i = 1; i <= total; i++) {
      const img = document.createElement('img');
      img.className = 'comic-reader-webtoon-page';
      img.dataset.page = String(i);
      img.alt = '';
      img.loading = 'lazy';                      // browser-native lazy load
      img.decoding = 'async';
      img.draggable = false;
      // Placeholder intrinsic dimensions so undecoded images reserve a
      // realistic chunk of vertical space — manga / webtoon pages cluster
      // around a 2:3 portrait ratio, so we seed every <img> with width=800
      // height=1200 as attributes (NOT CSS; HTML attrs only set the
      // initial aspect ratio and yield to the natural dimensions once the
      // image decodes). Without this, ``loading='lazy'`` + CSS
      // ``height: auto`` makes every undecoded page 0px tall, the whole
      // strip is 0px tall on mount, and the resume ``scrollIntoView`` /
      // ``scrollTop = webtoonScrollY`` both clamp to ~0 — the "resume
      // lands on page 1 even though state says page N" UX bug.
      // Same trick every modern image-heavy site uses to prevent CLS.
      img.width = 800;
      img.height = 1200;
      // Failure handling. ``loading='lazy'`` images that 404 or stall
      // mid-decode would otherwise leave a broken-image icon in the
      // strip with no way back. Mark the slot with a failed class so
      // CSS can show the alt text + retry affordance, keep the
      // placeholder dimensions so the strip layout doesn't collapse,
      // and wire a click handler that re-fires the load. Loading is
      // ``lazy``, so a re-set of src only actually hits the network
      // when the slot scrolls back into view — exactly when the user
      // can see the retry result.
      const pageIdx = i;
      img.alt = `Page ${pageIdx}`;
      img.addEventListener('error', () => {
        if (!container.contains(img)) return;
        img.classList.add('comic-reader-webtoon-page-failed');
        img.alt = `Page ${pageIdx} couldn't load — tap to retry`;
      });
      img.addEventListener('click', (e) => {
        // Loaded fine — let the click bubble to the stage handler so
        // the tap-zone nav (prev/next/toggle-chrome) keeps working.
        if (!img.classList.contains('comic-reader-webtoon-page-failed')) return;
        e.stopPropagation();
        img.classList.remove('comic-reader-webtoon-page-failed');
        img.alt = `Page ${pageIdx}`;
        // Clearing then re-setting the same URL forces a fresh load
        // attempt even if the browser cached the previous failure.
        const retryUrl = _pageUrl(this._file.id, pageIdx);
        img.removeAttribute('src');
        img.src = retryUrl;
      });
      const url = _pageUrl(this._file.id, i);
      const cacheKey = `${this._file.id}|${i}`;
      const cached = this._cropBorders ? this._cropCache.get(cacheKey) : null;
      if (cached) {
        img.src = cached;
      } else if (this._cropBorders) {
        // Load the raw image via a detached probe so onload runs even
        // when browser native lazy-load hasn't decoded the visible <img>
        // yet. Swap only when detection finds something useful.
        img.crossOrigin = 'anonymous';
        img.src = url;
        img.addEventListener('load', () => {
          if (!this._cropBorders) return;
          const insets = _detectBorders(img);
          if (!insets) return;
          const cropped = _cropImageToDataUrl(img, insets);
          if (!cropped) return;
          this._cropCache.set(cacheKey, cropped);
          img.src = cropped;
        }, { once: true });
      } else {
        img.src = url;
      }
      container.appendChild(img);
      this._webtoonPages.push(img);
    }

    // Continue-into-next-chapter sentinel. Sits below the last page so
    // the user has to scroll past the chapter to reveal it. Once it's
    // mostly visible, fire the transition — that's "continued scrolling
    // pulls in the next chapter" without needing a button press. Last
    // chapter of a series gets a static end card with no observer.
    if (this._hasNextChapter()) {
      const sentinel = document.createElement('div');
      sentinel.className = 'comic-reader-webtoon-end-sentinel';
      const label = escapeHtml(this._nextChapterLabel() || 'next chapter');
      sentinel.innerHTML = `
        <div class="comic-reader-webtoon-end-arrow" aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="7 13 12 18 17 13"/>
            <polyline points="7 6 12 11 17 6"/>
          </svg>
        </div>
        <div class="comic-reader-webtoon-end-message">Continuing to ${label}</div>
      `;
      container.appendChild(sentinel);
      if ('IntersectionObserver' in window) {
        // Sentinel fires the next-chapter transition. Four gates:
        //
        // 1. ``intersectionRatio >= 0.6`` — sentinel is genuinely visible,
        //    not just edge-touching the viewport.
        //
        // 2. ``_reachedContinuePoint`` — the page observer has reported a
        //    page in the last ~15% of the strip as dominant. Without
        //    this, lazy-loaded images leave the strip 0px tall on mount
        //    and the sentinel sits in viewport immediately, cascading
        //    through chapters in seconds.
        //
        // 3. ``scrollTop > 0`` — backup for ultra-short chapters where
        //    Math.ceil(pageCount * 0.85) rounds down to 1, making the
        //    home-stretch flag flip on initial render. Requires the
        //    user to have moved the strip at all.
        //
        // 4. ``_SENTINEL_COOLDOWN_MS since last sentinel advance`` —
        //    final brake against any residual cascade. Even if every
        //    other gate is somehow bypassed, the sentinel can fire at
        //    most once per cooldown window across the whole reader
        //    session. The timestamp persists across chapter transitions
        //    so a chapter-A sentinel followed by a chapter-B sentinel
        //    within 60s is detected and refused. Cooldown skips don't
        //    disconnect — observer keeps watching, fires correctly when
        //    the cooldown expires and the user crosses a threshold.
        //
        // Disconnect-before-fire so a re-trigger during teardown can't
        // double-advance.
        const so = new IntersectionObserver((entries) => {
          for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            if (entry.intersectionRatio < 0.6) continue;
            if (!this._reachedContinuePoint) continue;
            if ((this._stage?.scrollTop ?? 0) <= 0) continue;
            if (Date.now() - this._lastSentinelAdvanceAt < _SENTINEL_COOLDOWN_MS) continue;
            this._lastSentinelAdvanceAt = Date.now();
            so.disconnect();
            this._sentinelObserver = null;
            this._goNextChapter();
            break;
          }
        }, { root: this._stage, threshold: [0.3, 0.6, 0.9] });
        so.observe(sentinel);
        this._sentinelObserver = so;
      }
    } else {
      const endCard = document.createElement('div');
      endCard.className = 'comic-reader-webtoon-end-sentinel comic-reader-webtoon-end-final';
      endCard.innerHTML = `<div class="comic-reader-webtoon-end-message">End of series</div>`;
      container.appendChild(endCard);
    }

    // Scroll to user's saved progress on initial render — so reopening
    // a chapter mid-read lands them close to where they left off.
    //
    // Two-pass scroll: the first pass runs against the placeholder
    // 2:3 layout (from the HTML width/height attrs above), which lands
    // approximately on the saved page. The second pass re-runs once
    // the target image has actually decoded — its natural aspect
    // ratio may differ from the placeholder, so without this the
    // user can land half a page off. A series of load events along
    // the way (each image as it decodes shifts everything below it)
    // would be too noisy; we only re-anchor when the target itself
    // resolves. Bounded by overlay-still-mounted + same-target guards
    // so a chapter transition during decoding doesn't fight the new
    // chapter's own resume scroll.
    if (this._page > 1) {
      const target = this._webtoonPages[this._page - 1];
      if (target) {
        const anchor = () => {
          if (!this._overlay || !this._stage) return;
          if (this._webtoonPages[this._page - 1] !== target) return;
          // Only re-anchor if the viewport is still BEFORE the target.
          // ``_restoreResumeState`` may have already placed scrollTop
          // inside the target page using the finer-grained
          // ``webtoonScrollY`` signal; clobbering that would lose the
          // user's intra-page position. The pathological case we're
          // catching here is the strip being 0px tall before decode,
          // which clamps every scroll to ~0 (above the target).
          if (this._stage.scrollTop + 4 < target.offsetTop) {
            target.scrollIntoView({ block: 'start' });
          }
        };
        requestAnimationFrame(anchor);
        if (!target.complete || !target.naturalHeight) {
          target.addEventListener('load', anchor, { once: true });
        }
      }
    }

    // Observer watches every page; whichever has the largest visible
    // area becomes the "current" page. Cheap — 30-100 observed targets.
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        // Pick the entry with the highest intersection ratio; that's
        // the page the reader is "on" right now.
        let best = null;
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          if (!best || entry.intersectionRatio > best.intersectionRatio) best = entry;
        }
        if (!best) return;
        const page = Number(best.target.dataset.page);
        // Resume-protection: while the saved-page scroll hasn't landed
        // yet, ignore observer reports that land below it. Lazy-loaded
        // 0-height images make page 1 look dominant at scrollTop≈0 and
        // would otherwise clobber the saved page via the push that
        // ``_schedulePushProgress`` queues. Once we see a page at or
        // past the saved target, the resume has landed and normal
        // behavior resumes.
        if (!this._sawResumePage) {
          if (page >= this._meta.currentPage) {
            this._sawResumePage = true;
          } else {
            return;
          }
        }
        if (page && page !== this._page) {
          this._page = page;
          this._updatePageChrome();
          this._schedulePushProgress();
          this._scheduleSurfaceSync();
          // Flip the continue-eligibility flag once the user has
          // read into the home stretch (~85%+ of pages). Two gates:
          //
          // 1. ``page >= 85% threshold`` — semantic "user has consumed
          //    most of the chapter" signal.
          //
          // 2. ``scrollTop > 0`` — strict guard against initial-layout
          //    chaos. Lazy-loaded webtoon images can decode out of DOM
          //    order; if image 28 decodes before image 1, it briefly
          //    becomes the dominant intersection at scrollTop 0 and
          //    fires this callback with `page = 28`. Without the
          //    scrollTop gate the flag would latch true on first paint
          //    and the sentinel could fire as soon as anything else
          //    nudges the strip. Requiring the user to have actually
          //    scrolled forces a real human signal at flip time.
          if (
            this._pageCount > 0
            && page >= Math.ceil(this._pageCount * 0.85)
            && (this._stage?.scrollTop ?? 0) > 0
          ) {
            this._reachedContinuePoint = true;
          }
        }
      }, {
        root: this._stage,
        threshold: [0.3, 0.6, 0.9],
      });
      for (const img of this._webtoonPages) observer.observe(img);
      this._webtoonObserver = observer;
    }

    // Resume the auto-scroll glide if one is configured — this is what
    // carries it across chapter transitions (teardown stopped it, the
    // rebuilt strip restarts it) and what kicks it in when the user
    // picks a speed from paged mode and then switches to webtoon. Skipped
    // on the initial mount so a remembered speed doesn't start moving the
    // instant the reader opens.
    if (this._autoScrollPxPerSec() && !this._autoScrollSuppressMountStart) {
      this._startAutoScroll();
    }
  }

  _teardownWebtoon({ keepContainer = false } = {}) {
    // Engine stops on every teardown; _renderWebtoon restarts it at the
    // end of a successful rebuild, so chapter transitions glide straight
    // through while mode switches and unmount stop cleanly.
    this._stopAutoScroll();
    if (this._webtoonObserver) {
      this._webtoonObserver.disconnect();
      this._webtoonObserver = null;
    }
    if (this._sentinelObserver) {
      this._sentinelObserver.disconnect();
      this._sentinelObserver = null;
    }
    this._webtoonPages = [];
    if (!keepContainer) {
      const container = this._overlay?.querySelector('.comic-reader-webtoon');
      if (container) container.innerHTML = '';
    }
  }

  // --- Auto-scroll (webtoon) ------------------------------------------
  // Continuous-glide engine ported from the cast receiver
  // (cast-comic.js startContinuousScroll): an rAF loop adds
  // velocity × Δt to the stage's scrollTop each frame. The existing
  // end-sentinel handles rolling into the next chapter — the glide
  // just keeps feeding it scroll position. When the strip bottoms out
  // without a transition (last chapter, or the sentinel's cascade
  // cooldown refused), the engine switches itself off rather than
  // spinning at the bottom holding a wake lock.

  _autoScrollPxPerSec() {
    return _AUTOSCROLL_SPEEDS[this._autoScroll] || 0;
  }

  _setAutoScroll(value) {
    if (!_AUTOSCROLL_MODES.includes(value)) return;
    this._autoScroll = value;
    this._updateDrawerActive('autoscroll', value);
    // Persist the speed (per-series → global), same write-through as mode/fit.
    _saveAutoScroll(value);
    _patchSeriesPrefs(this._seriesId, { autoscroll: value });
    if (!this._autoScrollPxPerSec()) {
      this._stopAutoScroll();
      showToast('Auto-scroll off', 'info', 900);
      return;
    }
    if (this._mode !== 'webtoon') {
      // Remember the choice; _renderWebtoon starts the engine when the
      // user lands in webtoon mode.
      showToast(`Auto-scroll ${_AUTOSCROLL_LABELS[value].toLowerCase()} — starts in Webtoon mode`, 'info', 1600);
      return;
    }
    showToast(`Auto-scroll: ${_AUTOSCROLL_LABELS[value].toLowerCase()}`, 'info', 900);
    this._startAutoScroll();
  }

  _cycleAutoScroll() {
    const idx = _AUTOSCROLL_MODES.indexOf(this._autoScroll);
    const next = _AUTOSCROLL_MODES[(idx + 1) % _AUTOSCROLL_MODES.length];
    this._setAutoScroll(next);
  }

  _startAutoScroll() {
    if (this._mode !== 'webtoon') return;
    if (!this._autoScrollPxPerSec()) return;
    if (this._autoScrollRaf) return;             // already running
    this._requestWakeLock();
    this._autoScrollLastTs = performance.now();
    const tick = (now) => {
      this._autoScrollRaf = 0;
      if (!this._overlay || this._mode !== 'webtoon') return;
      const pxPerSec = this._autoScrollPxPerSec();
      if (!pxPerSec) return;
      // Δt clamp (100ms) — a backgrounded tab's first frame back would
      // otherwise jump the strip by the entire time away.
      const dt = Math.min(now - this._autoScrollLastTs, 100);
      this._autoScrollLastTs = now;
      const s = this._stage;
      if (s && !this._autoScrollHeld) {
        const px = pxPerSec * (dt / 1000);
        if (px > 0) s.scrollBy({ top: px, behavior: 'auto' });
        // Bottomed out on a real (laid-out) strip with no transition
        // pending — switch off. The 64px scrollable floor mirrors the
        // cast receiver's WEBTOON_SCROLLABLE_MIN_PX guard: an empty or
        // still-loading strip has scrollHeight ≈ clientHeight and must
        // not read as "finished".
        if (
          s.scrollHeight - s.clientHeight >= 64
          && s.scrollTop + s.clientHeight >= s.scrollHeight - 4
        ) {
          this._autoScroll = 'off';
          this._updateDrawerActive('autoscroll', 'off');
          this._stopAutoScroll();
          showToast(
            this._hasNextChapter()
              ? 'Auto-scroll paused at the end of the chapter'
              : 'Auto-scroll finished — end of series',
            'info', 1800,
          );
          return;
        }
      }
      this._autoScrollRaf = requestAnimationFrame(tick);
    };
    this._autoScrollRaf = requestAnimationFrame(tick);
  }

  _stopAutoScroll() {
    if (this._autoScrollRaf) {
      cancelAnimationFrame(this._autoScrollRaf);
      this._autoScrollRaf = 0;
    }
    this._releaseWakeLock();
  }

  // Wake lock — keeps the screen on while the strip glides unattended
  // (same rationale as the cast receiver: auto-advance with no touch
  // input looks idle to the OS). Best-effort; unsupported browsers just
  // fall back to their normal screen timeout.
  async _requestWakeLock() {
    if (this._wakeLock) return;
    if (!('wakeLock' in navigator)) return;
    try {
      this._wakeLock = await navigator.wakeLock.request('screen');
      this._wakeLock.addEventListener('release', () => { this._wakeLock = null; });
    } catch { /* permission denied / low battery — harmless */ }
  }

  async _releaseWakeLock() {
    if (!this._wakeLock) return;
    try { await this._wakeLock.release(); } catch { /* already released */ }
    this._wakeLock = null;
  }

  _goTo(page) {
    if (page < 1) return;
    if (this._pageCount && page > this._pageCount) return;
    // Explicit user navigation — release the resume gate so subsequent
    // pushes flow normally. Covers prev/next/scrubber/home/end paths
    // since they all funnel through ``_goTo``.
    this._sawResumePage = true;
    if (this._mode === 'webtoon') {
      // Scroll instead of load — Home/End and future scrubber jumps flow
      // through here. The IntersectionObserver syncs _page back once the
      // smooth scroll lands, so we don't fight it by setting _page first.
      this._webtoonScrollToPage(page);
      return;
    }
    if (page === this._page) return;
    this._page = page;
    this._loadPage(page);
    this._updatePageChrome();
    this._schedulePushProgress();
    this._scheduleSurfaceSync({ immediate: true });
  }

  _schedulePushProgress() {
    // Resume-protection gate: don't persist while we're still waiting
    // for the saved-page scroll to land. See ``_sawResumePage`` in the
    // constructor for why.
    if (!this._sawResumePage) return;
    // Debounced progress push — avoids a POST per arrow-key mash. A
    // 1200ms window catches both rapid flipping (user lands on the page
    // they want) and normal reading cadence.
    if (this._progressTimer) clearTimeout(this._progressTimer);
    this._progressTimer = setTimeout(() => {
      _pushProgress(this._file.id, this._page, this._pageCount);
      this._progressTimer = null;
    }, 1200);
  }

  _loadPage(page) {
    const token = ++this._loadingToken;
    this._showLoading();
    this._hideError();
    // New page → fresh zoom. A paged reader expects each page to start
    // fit-to-viewport, not inherit the previous page's pinch state.
    this._resetZoom();

    // Preload neighbors FIRST so they start downloading in parallel
    // with the main page — user sees the current page as fast as
    // possible but the next click is already warm in the HTTP cache.
    this._prefetchNeighbors(page);

    if (this._mode === 'dual') {
      this._loadDualPages(page, token);
    } else {
      this._loadSinglePage(page, token);
    }
  }

  _loadSinglePage(page, token) {
    const url = _pageUrl(this._file.id, page);
    const cacheKey = `${this._file.id}|${page}`;
    const cached = this._cropBorders ? this._cropCache.get(cacheKey) : null;
    if (cached) {
      if (token !== this._loadingToken) return;
      this._img.src = cached;
      this._hideLoading();
      return;
    }
    const img = new Image();
    img.decoding = 'async';
    img.crossOrigin = 'anonymous';  // needed for canvas getImageData (crop)
    img.onload = () => {
      if (token !== this._loadingToken) return;
      if (this._cropBorders) {
        // Apply crop synchronously — detection is fast (<5ms) and keeps
        // the displayed frame's identity predictable for pinch/preload.
        const insets = _detectBorders(img);
        if (insets) {
          const cropped = _cropImageToDataUrl(img, insets);
          if (cropped) {
            this._cropCache.set(cacheKey, cropped);
            this._img.src = cropped;
            this._hideLoading();
            return;
          }
        }
      }
      this._img.src = url;
      this._img.decode?.().catch(() => {});
      this._hideLoading();
    };
    img.onerror = () => {
      if (token !== this._loadingToken) return;
      this._hideLoading();
      this._showError(`Couldn't load page ${page}. The provider may be offline or the chapter not downloaded.`);
    };
    img.src = url;
  }

  /** Render a side-by-side spread. Pair rules:
   *   - Page 1 (the cover) is always solo, right slot blank.
   *   - Pairs: (2,3), (4,5), ... so from page N (even, N>=2) we show N + N+1.
   *   - If the user navigated to page 3 by other means (keyboard specific),
   *     we display (2,3) so the layout stays consistent.
   *   - At the end of chapter if pageCount is odd, the final page goes
   *     in the left slot alone.
   *   - RTL reverses LEFT/RIGHT so the "next" page in reading order
   *     is on the appropriate side.
   */
  _loadDualPages(page, token) {
    const total = this._pageCount || 0;
    let leftPage, rightPage;
    if (page === 1) {
      leftPage = 1; rightPage = null;
    } else {
      // Normalize to pair start: even page is the pair-left, odd page
      // belongs to the pair whose left is the prior even page.
      const pairLeft = page % 2 === 0 ? page : page - 1;
      leftPage = pairLeft;
      rightPage = (total > 0 && pairLeft + 1 > total) ? null : pairLeft + 1;
      // Keep the internal page pointer aligned with what's actually
      // showing — the LEFT-side page is canonical "current".
      this._page = pairLeft;
    }

    const dual = this._overlay.querySelector('[data-zone="dual"]');
    const imgLeftEl = dual.querySelector('.comic-reader-dual-left');
    const imgRightEl = dual.querySelector('.comic-reader-dual-right');
    // Map logical left/right to DOM order based on reading direction.
    // In RTL, the DOM-right slot renders the LOWER page number (which is
    // the "next" page visually when you read right-to-left). We swap via
    // CSS (flex-direction: row-reverse) rather than here; the data stays
    // clean. Both slots get populated; CSS flips the order for RTL.
    const applyTo = (el, p) => {
      // Clear any previous failure state — same slot may be reused
      // when navigating between pairs.
      el.classList.remove('failed-slot');
      el.removeAttribute('title');
      el.onclick = null;
      if (p == null) {
        el.removeAttribute('src');
        el.classList.add('empty-slot');
        return;
      }
      el.classList.remove('empty-slot');
      const url = _pageUrl(this._file.id, p);
      const cacheKey = `${this._file.id}|${p}`;
      const cached = this._cropBorders ? this._cropCache.get(cacheKey) : null;
      if (cached) { el.src = cached; return; }
      const probe = new Image();
      probe.decoding = 'async';
      probe.crossOrigin = 'anonymous';
      probe.onload = () => {
        if (token !== this._loadingToken) return;
        if (this._cropBorders) {
          const insets = _detectBorders(probe);
          if (insets) {
            const cropped = _cropImageToDataUrl(probe, insets);
            if (cropped) {
              this._cropCache.set(cacheKey, cropped);
              el.src = cropped;
              return;
            }
          }
        }
        el.src = url;
      };
      probe.onerror = () => {
        if (token !== this._loadingToken) return;
        // Distinct ``failed-slot`` (vs ``empty-slot`` for intentional
        // blank halves like page-1's right slot) so the user sees a
        // clear "this half didn't load" affordance instead of an
        // ambiguous blank. Click re-runs applyTo to retry just this
        // slot — sibling slot stays put. stopPropagation so the
        // retry click doesn't ALSO trigger the stage's tap-zone nav.
        el.removeAttribute('src');
        el.classList.add('failed-slot');
        el.title = `Page ${p} couldn't load — click to retry`;
        el.onclick = (e) => { e.stopPropagation(); applyTo(el, p); };
      };
      probe.src = url;
    };

    applyTo(imgLeftEl, leftPage);
    applyTo(imgRightEl, rightPage);
    // One-shot "both slots done" signal — hide loading overlay once
    // either slot has fired (avoids lingering spinner when right slot
    // is intentionally empty on page 1 or final odd page).
    requestAnimationFrame(() => this._hideLoading());
  }

  _prefetchNeighbors(page) {
    const wanted = new Set();
    // Look ahead more than behind — forward preload is the usual case.
    for (let delta = -1; delta <= 2; delta++) {
      const p = page + delta;
      if (p < 1 || p === page) continue;
      if (this._pageCount && p > this._pageCount) continue;
      wanted.add(p);
    }
    // Warm: create Image() objects for pages we want; drop refs for pages
    // outside the window so the browser can evict decoded bitmaps.
    const kept = new Set();
    for (const img of this._prefetchImgs) {
      const prefetchPage = Number(img.dataset.page);
      if (wanted.has(prefetchPage)) {
        kept.add(prefetchPage);
      }
    }
    this._prefetchImgs = new Set(
      [...this._prefetchImgs].filter(i => kept.has(Number(i.dataset.page))),
    );
    for (const p of wanted) {
      if (kept.has(p)) continue;
      const img = new Image();
      img.dataset.page = String(p);
      img.decoding = 'async';
      img.src = _pageUrl(this._file.id, p);
      this._prefetchImgs.add(img);
    }
  }

  /** Warm the next chapter on the upstream provider while the user
   *  reads the current one. Fires on reader mount and on every chapter
   *  transition, so by the time the user reaches the end of chapter N
   *  the upstream provider has already prepared chapter N+1 (Suwayomi's
   *  ``fetchChapterPages`` mutation typically takes 5–10s — running it
   *  in the background covers that latency). One-shot per current
   *  chapter, gated by ``_nextChapterPrefetched`` (reset on transition).
   *  Skips if there is no next chapter. */
  _prefetchNextChapter() {
    if (this._nextChapterPrefetched) return;
    if (!this._hasNextChapter()) return;
    const next = this._siblings[this._siblingIndex + 1];
    if (!next?.id) return;
    this._nextChapterPrefetched = true;
    // Manifest call hydrates upstream's accurate page_count for the next
    // chapter — Suwayomi runs prepare_chapter as a side effect, so the
    // first images are ready by the time we ask for them.
    fetch(
      `/api/media/comic/manifest/${encodeURIComponent(next.id)}`,
    ).catch(() => { /* best-effort */ });
    // Webtoon needs a tall first image visible immediately; paged just
    // needs page 1. Two for paged catches users who flip past page 1
    // before the second prefetch lands.
    const warmPages = this._mode === 'webtoon' ? 3 : 2;
    for (let i = 1; i <= warmPages; i++) {
      const img = new Image();
      img.decoding = 'async';
      img.src = _pageUrl(next.id, i);
      // Hold a reference so the browser keeps the decoded bitmap until
      // we transition. _transitionToChapter clears this set as part of
      // its teardown, so memory doesn't grow across chapters.
      this._prefetchImgs.add(img);
    }
  }

  _updatePageChrome() {
    if (!this._overlay) return;
    const currEl = this._overlay.querySelector('.comic-reader-page-current');
    if (currEl) currEl.textContent = String(this._page);
    // Scrubber-consistent math: page 1 of N sits at 0%, page N at 100%.
    // Using the previous `page / total` formula made page 1 sit at a
    // small non-zero fill, which fought the scrubber's 0% drag-target.
    const pct = this._pageCount > 1
      ? Math.max(0, Math.min(100, ((this._page - 1) / (this._pageCount - 1)) * 100))
      : (this._pageCount === 1 ? 100 : 0);
    const fill = this._overlay.querySelector('.comic-reader-progress-fill');
    const bar = this._overlay.querySelector('.comic-reader-progress-bar');
    if (fill) fill.style.width = `${pct}%`;
    if (bar) {
      bar.setAttribute('aria-valuenow', String(this._page));
      bar.style.setProperty('--scrubber-handle-pct', `${pct}%`);
    }
    // Persistent progress bar — always visible, stays in sync with the
    // auto-hiding chrome counter so a distraction-free reader still
    // carries sense of chapter progress.
    const persistent = this._overlay.querySelector('.comic-reader-progress-persistent-fill');
    if (persistent) persistent.style.width = `${pct}%`;
    const persistentBar = this._overlay.querySelector('.comic-reader-progress-persistent');
    if (persistentBar) persistentBar.setAttribute('aria-valuenow', String(this._page));
    // Single choke point for "the visible page changed" — paged `_goTo`, the
    // webtoon IntersectionObserver and the dual-page pair math all land here.
    // Narration follows the reader from this one call; `setPage` ignores a
    // repeat of the page it's already on, so the swipe-drag snap-back calls
    // that also reach here cost nothing.
    this._narrationBar?.setPage(this._page);
  }

  _toggleDir() {
    this._setDir(this._dir === 'rtl' ? 'ltr' : 'rtl');
    showToast(`Reading ${this._dir === 'rtl' ? 'right-to-left' : 'left-to-right'}`, 'info', 900);
  }

  /** Apply a reading direction and write it through all three layers
   *  (per-file legacy, per-series, global default). Reused by the
   *  direction toggle in chrome, the settings drawer segment, and the
   *  R shortcut — single code path keeps the persistence invariant. */
  _setDir(dir) {
    if (dir !== 'ltr' && dir !== 'rtl') return;
    this._dir = dir;
    _saveDir(this._file.id, dir);
    _saveLastDir(dir);
    _patchSeriesPrefs(this._seriesId, { direction: dir });
    this._overlay.classList.toggle('dir-rtl', dir === 'rtl');
    this._overlay.classList.toggle('dir-ltr', dir === 'ltr');
    const label = this._overlay.querySelector('.comic-reader-dir-label');
    if (label) label.textContent = dir === 'rtl' ? 'RTL' : 'LTR';
    this._updateDrawerActive('dir', dir);
    this._scheduleSurfaceSync({ immediate: true });
  }

  _showLoading() {
    this._overlay?.classList.add('is-loading');
  }
  _hideLoading() {
    this._overlay?.classList.remove('is-loading');
  }
  _showError(msg) {
    const box = this._overlay?.querySelector('.comic-reader-error');
    const p = this._overlay?.querySelector('.comic-reader-error-msg');
    if (box) box.hidden = false;
    if (p) p.textContent = msg;
  }
  _hideError() {
    const box = this._overlay?.querySelector('.comic-reader-error');
    if (box) box.hidden = true;
  }

  _revealChrome() {
    if (!this._overlay) return;
    this._overlay.classList.remove('chrome-hidden');
    this._clearIdleTimer();
    this._idleTimer = setTimeout(() => {
      this._overlay?.classList.add('chrome-hidden');
    }, 2200);
  }

  _toggleChrome() {
    if (!this._overlay) return;
    if (this._overlay.classList.contains('chrome-hidden')) {
      this._revealChrome();
    } else {
      this._overlay.classList.add('chrome-hidden');
      this._clearIdleTimer();
    }
  }

  _clearIdleTimer() {
    if (this._idleTimer) {
      clearTimeout(this._idleTimer);
      this._idleTimer = null;
    }
  }
}


/**
 * Open the comic reader for a file_index entry. Primary entry point —
 * called from ``comics.js`` (series drill-down, which passes siblings
 * so the reader can flow across chapters), ``files/actions.js::
 * activateFile`` (Media tiles, flat Files grid, continue rail), and
 * Discovery (lean ``{id, kind}`` stub).
 *
 * ``siblings``: optional array of file_index entries representing the
 * full chapter list for this file's series, ordered by source_order
 * ascending. Enables prev/next-chapter nav inside the reader.
 *
 * When the caller doesn't pass siblings, we self-resolve them from the
 * entry's ``series_id`` before mounting — otherwise every chapter
 * opened outside the comics drill-in reads as a one-chapter series
 * ("End of series" after chapter 1, no auto-advance). The same fetch
 * the drill-in uses (/api/files/comics/series/{id}/chapters) keeps the
 * ordering contract identical across surfaces.
 */
export async function openComicReader(file, { siblings = null, resume = null } = {}) {
  if (!file || !file.id) return;
  // Load server-synced reader prefs before constructing — the constructor
  // reads mode/fit/direction/bg/crop/auto-scroll synchronously from the cache.
  await _ensurePrefsLoaded();
  // Companion presence: this comic is now "what I'm reading".
  import('../architect-observer.js')
    .then(m => m.reportAttention('surface.comic.opened', {
      label: file.display_name || file.name || '',
      kind: 'comic',
      ref: String(file.id),
    }))
    .catch(() => {});

  let entry = file;
  let sibs = Array.isArray(siblings) ? siblings : null;
  if (!sibs) {
    try {
      // Lean callers (Discovery passes {id, kind}) don't carry
      // series_id — resolve the full row first. Entries fetched via
      // /api/files/entry/ already include it.
      if (entry.series_id === undefined) {
        const resp = await fetch(`/api/files/entry/${encodeURIComponent(entry.id)}`);
        if (resp.ok) {
          const row = await resp.json();
          if (row?.id) entry = row;
        }
      }
      if (entry.series_id) {
        const resp = await fetch(
          `/api/files/comics/series/${encodeURIComponent(entry.series_id)}/chapters?limit=2000`,
        );
        if (resp.ok) {
          const data = await resp.json();
          if (Array.isArray(data?.files) && data.files.length) {
            sibs = data.files;
            // Use the sibling row as the open entry so progress/meta
            // come from the same (fresher) snapshot the prev/next
            // chapters will use.
            const own = sibs.find((c) => c?.id === entry.id);
            if (own) entry = own;
          }
        }
      }
    } catch (err) {
      // Continuity is an enhancement — a failed resolve still opens
      // the chapter, just without cross-chapter nav.
      console.warn('[comic-reader] sibling resolve failed:', err);
    }
  }

  const reader = new ComicReader(entry, { siblings: sibs });
  reader.mount();
  if (resume) reader._restoreResumeState(resume);
  return reader;
}
