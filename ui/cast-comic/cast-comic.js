/**
 * cast-comic.js — TV comic/manga reader.
 *
 * Renders comic pages from the Augmentum proxy onto a casting TV.
 * Designed to feel like a polished reader app, not a slideshow:
 *
 *   - Robust per-page loading (token + timeout + retry + cache-bust),
 *     so one flaky page never poisons the reader. Each page request
 *     gets a fresh <img> built from a Blob URL, eliminating the
 *     double-buffer state-pollution failure where every other page
 *     would stick on "Loading…".
 *
 *   - Adaptive layout: after sampling the first few pages we pick the
 *     best mode — single-page on portrait / square-aspect TVs, dual-
 *     page (two facing pages) on widescreen TVs reading portrait
 *     manga, or webtoon (continuous vertical scroll) for tall-strip
 *     content typical of Suwayomi-served webtoons.
 *
 *   - Auto border crop: trims uniform-color margins around scanned
 *     pages so artwork uses more of the screen. Per-side cap prevents
 *     the detector from misreading uniform splash panels as borders.
 *
 *   - Wake Lock: keeps the TV awake while reading; auto-released on
 *     unload or when autoplay stops.
 *
 *   - Persistent chapter progress bar + autoplay countdown — the
 *     reader always tells you where you are without needing the HUD.
 *
 * Patch protocol (postMessage from cast-receiver):
 *   {page_idx: N}             — absolute page (1-indexed)
 *   {page_delta: N}           — relative move (any integer)
 *   {jump: 'first'|'last'}    — endpoints
 *   {autoplay_ms: N}          — 0 disables; otherwise per-page dwell
 *   {paused: bool}            — pause/resume autoplay
 *   {mode: 'single'|'dual'|'webtoon'|'auto'}  — layout override
 *   {fit: 'smart'|'width'|'height'|'native'}  — fit override
 *   {reading_direction: 'ltr'|'rtl'}          — for dual mode pairing
 *   {border_crop: bool}       — toggle auto crop
 *   {retry: true}             — re-fetch current page (clears failure)
 */


import { createNarrationClock } from '../scripts/comic-reader/narration-clock.js';
import { AudioBus } from '../scripts/audio-bus.js';
import * as musicSource from '../scripts/music-source.js';

// ── Config ─────────────────────────────────────────────────────

const PARAMS = new URLSearchParams(location.search);
// Reassigned by ``transitionToChapter`` when the cast surface auto-
// advances to the next chapter in a series. Treated as immutable by
// every other call site — only the chapter-transition path rewrites it.
let FILE_ID = (PARAMS.get('id') || '').trim();

const LOAD_TIMEOUT_MS = 12000;       // per-attempt fetch timeout
const RETRY_BACKOFF_MS = [800, 1800, 4000];  // 3 retries, ~7s budget
const PREFETCH_AHEAD = 2;            // pages requested ahead of current
const PREFETCH_BEHIND = 1;           // pages kept warm behind
const SPINNER_DELAY_MS = 400;        // delay before showing spinner
const HUD_FADE_MS = 2500;            // HUD visible after a turn
const CACHE_MAX_PAGES = 30;          // LRU bound on decoded pages
const WEBTOON_PREFETCH_AHEAD = 5;    // pages warmed ahead of currentPage
                                     // on every scroll-tracker tick
const CROP_THRESHOLD_PER_CHANNEL = 14;
const CROP_MIN_TOTAL = 0.04;
const CROP_MAX_TOTAL = 0.55;
const CROP_PER_SIDE_CAP = 0.30;
const AUTOPLAY_PROGRESS_HZ = 20;     // countdown bar update rate
const ASPECT_SAMPLE_COUNT = 3;       // pages sampled before mode auto-pick
const STRIP_ASPECT_THRESHOLD = 0.6;  // h/w > 1/0.6 → webtoon strip
const PORTRAIT_ASPECT_THRESHOLD = 0.85; // w/h < this → portrait page


// ── DOM refs ───────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const stage = $('[data-cc-stage]');
const hud = $('[data-cc-hud]');
const titleEl = $('[data-cc-title]');
const pageEl = $('[data-cc-page]');
const totalEl = $('[data-cc-total]');
const modeChipEl = $('[data-cc-mode-chip]');
const speedChipEl = $('[data-cc-speed-chip]');
const progressFill = $('[data-cc-progress-fill]');
const autoplayStrip = $('[data-cc-autoplay-strip]');
const autoplayFill = $('[data-cc-autoplay-fill]');
const spinner = $('[data-cc-spinner]');
const errorOverlay = $('[data-cc-error]');
const errorText = $('[data-cc-error-text]');
const retryBtn = $('[data-cc-retry]');


// ── Helpers ────────────────────────────────────────────────────

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    if (signal) {
      signal.addEventListener('abort', () => {
        clearTimeout(t);
        reject(new DOMException('aborted', 'AbortError'));
      }, { once: true });
    }
  });
}

function pageUrl(fileId, n, attempt = 0) {
  // Hint the server at the TV's effective resolution. Providers that
  // honour size hints (future Komga `dimension=`, Pillow resize layer)
  // can serve a smaller payload; today the param is forwarded and
  // ignored upstream, which is harmless. The DPR multiply makes 4K
  // panels actually request 4K imagery instead of 1080p upscaled.
  const w = Math.round((window.innerWidth || 1920) * (window.devicePixelRatio || 1));
  const base = `/api/media/comic/page/${encodeURIComponent(fileId)}`
    + `?page=${n}&w=${w}`;
  // Cache-bust only on retries so the happy path stays cacheable. The
  // Cache-Control: max-age=3600 the server sets is good — we just need
  // to bypass the *negative* cache when the first attempt hit a 5xx.
  return attempt > 0 ? `${base}&_=${Date.now()}` : base;
}


// ── Page loader ────────────────────────────────────────────────
// Per-page idempotent fetch with timeout + retry. Returns
// { img, naturalWidth, naturalHeight, aspectRatio, blobUrl }.

const inflight = new Map();    // pageNum → { controller, promise }
const blobUrls = new Map();    // pageNum → ObjectURL (revoked on evict)

async function fetchPage(pageNum, { signal } = {}) {
  if (inflight.has(pageNum)) return inflight.get(pageNum).promise;
  const controller = new AbortController();
  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener('abort', () => controller.abort(), { once: true });
    }
  }
  const promise = loadWithRetry(pageNum, controller.signal);
  inflight.set(pageNum, { controller, promise });
  promise.finally(() => {
    if (inflight.get(pageNum)?.controller === controller) {
      inflight.delete(pageNum);
    }
  });
  return promise;
}

async function loadWithRetry(pageNum, signal) {
  let lastErr;
  for (let attempt = 0; attempt <= RETRY_BACKOFF_MS.length; attempt++) {
    if (signal.aborted) throw new DOMException('aborted', 'AbortError');
    try {
      return await loadOnce(pageNum, attempt, signal);
    } catch (err) {
      lastErr = err;
      if (err.name === 'AbortError') throw err;
      if (err.permanent) throw err;
      if (attempt < RETRY_BACKOFF_MS.length) {
        try { await sleep(RETRY_BACKOFF_MS[attempt], signal); }
        catch (e) { if (e.name === 'AbortError') throw e; }
      }
    }
  }
  throw lastErr;
}

async function loadOnce(pageNum, attempt, signal) {
  const url = pageUrl(FILE_ID, pageNum, attempt);
  // Combine the caller's abort signal with a fresh per-attempt timeout.
  // We can't trust the browser's internal timeout to fire — TV-class
  // WebKit builds sometimes hang HTTP requests indefinitely. The 12s
  // budget plus 3 retries (each cache-busting) is generous enough for
  // a slow Komga page generation but bounded enough to recover before
  // the user notices.
  const timeoutCtl = new AbortController();
  const timer = setTimeout(() => timeoutCtl.abort(), LOAD_TIMEOUT_MS);
  const composite = anySignal([signal, timeoutCtl.signal]);
  try {
    const resp = await fetch(url, {
      credentials: 'same-origin',
      signal: composite,
      cache: attempt > 0 ? 'reload' : 'default',
    });
    if (!resp.ok) {
      const err = new Error(`HTTP ${resp.status}`);
      err.name = 'PageError';
      // 4xx (except 408 Request Timeout, 429 Too Many) are permanent —
      // retrying gives nothing. 5xx + 408 + 429 are transient.
      err.permanent = resp.status >= 400 && resp.status < 500
        && resp.status !== 408 && resp.status !== 429;
      throw err;
    }
    const blob = await resp.blob();
    const objUrl = URL.createObjectURL(blob);
    const img = new Image();
    img.decoding = 'async';
    img.src = objUrl;
    // decode() resolves once the bitmap is ready; it throws on a
    // corrupt response that the server happened to return as 200.
    // Treat that as transient (some providers can re-encode on retry).
    try {
      await img.decode();
    } catch (decodeErr) {
      URL.revokeObjectURL(objUrl);
      const err = new Error(`decode failed`);
      err.name = 'PageError';
      err.permanent = false;
      throw err;
    }
    return {
      page: pageNum,
      img,
      blobUrl: objUrl,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
      aspectRatio: img.naturalWidth / Math.max(1, img.naturalHeight),
    };
  } finally {
    clearTimeout(timer);
  }
}

function anySignal(signals) {
  const ctl = new AbortController();
  for (const s of signals) {
    if (!s) continue;
    if (s.aborted) { ctl.abort(); break; }
    s.addEventListener('abort', () => ctl.abort(), { once: true });
  }
  return ctl.signal;
}


// ── Page cache (LRU on decoded pages) ──────────────────────────

const pageCache = new Map();   // pageNum → loaded payload (insertion-ordered)

function cacheGet(pageNum) {
  const v = pageCache.get(pageNum);
  if (v) { pageCache.delete(pageNum); pageCache.set(pageNum, v); }
  return v;
}

function cacheSet(pageNum, payload) {
  if (pageCache.has(pageNum)) pageCache.delete(pageNum);
  pageCache.set(pageNum, payload);
  blobUrls.set(pageNum, payload.blobUrl);
  // Evict oldest until under bound. Don't evict pages we just used —
  // start from the head (oldest) of the insertion-ordered Map.
  while (pageCache.size > CACHE_MAX_PAGES) {
    const oldest = pageCache.keys().next().value;
    if (oldest == null) break;
    pageCache.delete(oldest);
    const url = blobUrls.get(oldest);
    if (url) { URL.revokeObjectURL(url); blobUrls.delete(oldest); }
  }
}


// ── Border crop ────────────────────────────────────────────────
// Same logic as the desktop reader — sample the page at a small size,
// walk inward from each edge while pixels match the median corner
// colour, then refuse the crop if any single side trims >30% (almost
// always a misread of a uniform-color splash panel).

function detectBorders(img) {
  if (!img?.naturalWidth || !img?.naturalHeight) return null;
  const scale = Math.min(128 / img.naturalWidth, 128 / img.naturalHeight, 1);
  const w = Math.max(2, Math.round(img.naturalWidth * scale));
  const h = Math.max(2, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  try { ctx.drawImage(img, 0, 0, w, h); } catch { return null; }
  let data;
  try { data = ctx.getImageData(0, 0, w, h).data; } catch { return null; }

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
      < CROP_THRESHOLD_PER_CHANNEL * 3;
  };
  const rowBorder = (y) => { for (let x = 0; x < w; x++) if (!isBorder(x, y)) return false; return true; };
  const colBorder = (x) => { for (let y = 0; y < h; y++) if (!isBorder(x, y)) return false; return true; };

  let top = 0, bot = h - 1, left = 0, right = w - 1;
  while (top < h && rowBorder(top)) top++;
  while (bot > top && rowBorder(bot)) bot--;
  while (left < w && colBorder(left)) left++;
  while (right > left && colBorder(right)) right--;

  const tF = top / h;
  const bF = (h - 1 - bot) / h;
  const lF = left / w;
  const rF = (w - 1 - right) / w;
  const total = tF + bF + lF + rF;
  if (total < CROP_MIN_TOTAL || total > CROP_MAX_TOTAL) return null;
  if (tF > CROP_PER_SIDE_CAP || bF > CROP_PER_SIDE_CAP
      || lF > CROP_PER_SIDE_CAP || rF > CROP_PER_SIDE_CAP) return null;
  return { top: tF, right: rF, bottom: bF, left: lF };
}

function cropImage(img, insets) {
  const srcW = img.naturalWidth, srcH = img.naturalHeight;
  const sx = Math.floor(insets.left * srcW);
  const sy = Math.floor(insets.top * srcH);
  const sw = Math.floor((1 - insets.left - insets.right) * srcW);
  const sh = Math.floor((1 - insets.top - insets.bottom) * srcH);
  if (sw < 64 || sh < 64) return null;
  const canvas = document.createElement('canvas');
  canvas.width = sw; canvas.height = sh;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
  try { return canvas.toDataURL('image/jpeg', 0.92); } catch { return null; }
}


// ── Reader state ───────────────────────────────────────────────

const state = {
  pageCount: 0,
  currentPage: 1,
  title: '',
  mode: 'single',         // single | dual | webtoon
  modeRequested: 'auto',  // user/patch override; 'auto' lets us pick
  fitMode: 'smart',       // smart | width | height | native
  readingDirection: 'ltr',
  borderCrop: true,
  autoplayMs: 0,            // paged / dual — ms between page flips
  autoplayPxPerSec: 0,      // webtoon — continuous scroll velocity
  autoplayTimer: null,
  autoplayProgressTimer: null,
  autoplayStartedAt: 0,
  autoplayPaused: false,
  autoplayRaf: 0,           // rAF handle for continuous scroll
  autoplayRafLast: 0,       // last-tick timestamp for px/ms math
  wakeLock: null,
  hudFadeTimer: null,
  spinnerTimer: null,
  loadToken: 0,           // monotonic; renders ignore stale loads
  failedPage: 0,          // current page is in error state if non-zero
  aspectSamples: [],      // sampled aspect ratios (for mode auto-pick)
  sliverPages: new Set(), // page indices that are sliver fragments
                          // (provider-split webtoon residue). Used to
                          // skip them in paged-mode navigation and to
                          // trigger late webtoon-mode promotion when
                          // they appear after the first few pages.
  // Chapter-grouping awareness — mirrors the web reader's siblings
  // array so cast can auto-advance at end-of-chapter when autoplay is
  // on. ``seriesId`` comes from /api/media/details; ``siblings`` is the
  // full ordered chapter list for that series. ``siblingIndex`` is the
  // current chapter's position in ``siblings``. Empty/null values mean
  // "no sibling awareness" — cast falls back to stopping at end-of-
  // chapter, same as before this feature landed.
  seriesId: '',
  siblings: null,         // array<{ id, ...FileEntry }> | null
  siblingIndex: -1,
  // Suppresses repeated transition attempts while one is in flight (the
  // autoplay tick can fire again before the new chapter's first page
  // has rendered). Cleared by ``transitionToChapter`` once the new
  // chapter has resumed display.
  chapterTransitionPending: false,
  // Cooldown clock for the autoplay end-of-chapter advance. Prevents a
  // race where the webtoon rAF tick fires before the new chapter's
  // strip has laid out (scrollHeight ≈ clientHeight on an empty
  // container reads as "end of chapter") and cascades a rapid-skip
  // through the whole series. Mirrors comic-reader's _SENTINEL_COOLDOWN
  // pattern but shorter — cast surfaces are observed less directly, so
  // a stuck transition is more disruptive but multi-chapter binging
  // should still feel responsive between intentional advances.
  lastChapterTransitionAt: 0,
};

// Minimum interval between autoplay-driven chapter transitions. Anything
// faster than this is a bug (the strip hasn't had time to render, or
// the previous transition hasn't fully settled). 4s is long enough to
// rule out layout races on a slow TV but short enough that a legitimate
// 1-page chapter doesn't feel stalled at the boundary.
const CHAPTER_TRANSITION_COOLDOWN_MS = 4000;

// Webtoon end-of-chapter detection requires the strip to be meaningfully
// scrollable BEYOND the viewport, not just present. An empty container
// has scrollHeight === clientHeight (or off by a few px) — treating that
// as end-of-chapter is the race that caused the cascading-skip bug.
// 64px is enough to disambiguate "strip rendered, scrolled to bottom"
// from "strip still loading / one-page chapter inside one viewport".
const WEBTOON_SCROLLABLE_MIN_PX = 64;


// ── Mode + fit application ─────────────────────────────────────

function applyStageClasses() {
  if (!stage) return;
  stage.classList.remove('mode-single', 'mode-dual', 'mode-webtoon');
  stage.classList.add(`mode-${state.mode}`);
  stage.classList.remove('fit-smart', 'fit-width', 'fit-height', 'fit-native');
  stage.classList.add(`fit-${state.fitMode}`);
  stage.classList.remove('dir-ltr', 'dir-rtl');
  stage.classList.add(`dir-${state.readingDirection}`);
  // Always show the mode chip — when Mode is set to Auto on the
  // controller, the TV is the only place that knows what was actually
  // picked (webtoon vs dual vs single). Hiding it on "default" left
  // the user blind to a mismatch between their expectation and the
  // resolved layout.
  if (modeChipEl) {
    const parts = [state.mode];
    if (state.readingDirection === 'rtl') parts.push('rtl');
    modeChipEl.textContent = parts.join(' · ').toUpperCase();
    modeChipEl.hidden = false;
  }
}


// ── Auto mode detection ────────────────────────────────────────
// After the first few pages load, classify the comic. Manga-style
// portrait pages on a 16:9 TV become dual-page. Vertical strips
// (webtoons) become webtoon mode. Mixed / single → single.

function maybePickAutoMode() {
  if (state.modeRequested !== 'auto') return;  // user override pins it
  if (state.aspectSamples.length < ASPECT_SAMPLE_COUNT) return;
  if (state.mode !== 'single') return;  // already promoted

  const samples = state.aspectSamples.slice(-ASPECT_SAMPLE_COUNT * 2);
  const avg = samples.reduce((a, b) => a + b, 0) / samples.length;
  const min = Math.min(...samples);
  const max = Math.max(...samples);
  const tvAspect = window.innerWidth / Math.max(1, window.innerHeight);

  // Strip / webtoon: page is much taller than wide.
  if (avg < STRIP_ASPECT_THRESHOLD) {
    setMode('webtoon');
    return;
  }

  // Sliver-content detection: provider-side webtoon splitting (common
  // on Suwayomi sources) chops one long chapter into fixed-height
  // chunks, leaving the trailing chunk as a tiny wide-short sliver.
  // Paged mode renders those slivers as a separate "page" with massive
  // letterboxing — useless to the reader. Webtoon mode stacks them
  // back into a seamless strip. We catch this two ways:
  //   1. Any single page has aspect > 2.2 (very wide-short, a sliver)
  //   2. High variance between min and max page aspects (mixed page
  //      shapes within one chapter is a smoking gun for split content)
  // Either condition alone is enough to promote — these patterns
  // don't appear in legitimate paged comics.
  if (max > 2.2 || (min > 0 && max / min > 2.5)) {
    setMode('webtoon');
    return;
  }

  // Dual-page candidate: portrait pages on a landscape TV.
  if (avg < PORTRAIT_ASPECT_THRESHOLD && tvAspect > 1.4) {
    setMode('dual');
    return;
  }
  // Otherwise stay single.
}

function setMode(newMode) {
  if (state.mode === newMode) return;
  const wasWebtoon = state.mode === 'webtoon';
  // Stop the current mode's autoplay engine before flipping — the new
  // mode will start its own engine (if configured) at the end.
  _stopAutoplayEngines();
  state.mode = newMode;
  // Switching mode invalidates the rendered DOM. Rebuild the stage and
  // re-render whatever we're currently on.
  applyStageClasses();
  if (wasWebtoon) {
    // Drop the webtoon-only column var so it doesn't bleed into other
    // modes (it's scoped via the CSS selector but explicit clear keeps
    // devtools clean and avoids surprises if the rule ever changes).
    stage.style.removeProperty('--webtoon-col-width');
  }
  if (newMode === 'webtoon') {
    // Seed the column from any samples we already have so the first
    // paint isn't a flash of the 65vh fallback.
    updateWebtoonColumnWidth();
    renderWebtoon();
  } else {
    stage.innerHTML = '';
    showPage(state.currentPage, { immediate: true });
  }
  // Re-evaluate the autoplay presentation + engine for the new mode.
  // E.g. switching paged→webtoon while autoplay was 15s/page should
  // kick the continuous scroll engine if pxPerSec is configured, OR
  // hide the countdown strip if not.
  _refreshAutoplayPresentation();
  if (!state.autoplayPaused) startAutoplayForMode();
}


// ── HUD ────────────────────────────────────────────────────────

function flashHud(ms = HUD_FADE_MS) {
  if (!hud) return;
  hud.classList.add('on');
  clearTimeout(state.hudFadeTimer);
  if (!hud.classList.contains('persistent')) {
    state.hudFadeTimer = setTimeout(() => hud.classList.remove('on'), ms);
  }
}

function setHudPersistent(on) {
  if (!hud) return;
  if (on) {
    hud.classList.add('persistent', 'on');
  } else {
    hud.classList.remove('persistent');
    flashHud();
  }
}

function updateProgressStrip() {
  if (!progressFill || !state.pageCount) return;
  const pct = clamp((state.currentPage / state.pageCount) * 100, 0, 100);
  progressFill.style.width = `${pct}%`;
}


// ── Spinner + error ────────────────────────────────────────────

function armSpinner() {
  clearTimeout(state.spinnerTimer);
  state.spinnerTimer = setTimeout(() => {
    if (spinner) spinner.classList.add('on');
  }, SPINNER_DELAY_MS);
}

function disarmSpinner() {
  clearTimeout(state.spinnerTimer);
  state.spinnerTimer = null;
  if (spinner) spinner.classList.remove('on');
}

function showError(text) {
  state.failedPage = state.currentPage;
  if (errorText) errorText.textContent = text;
  if (errorOverlay) errorOverlay.hidden = false;
  disarmSpinner();
}

function hideError() {
  state.failedPage = 0;
  if (errorOverlay) errorOverlay.hidden = true;
}


// ── Manifest + metadata ────────────────────────────────────────

// Seed the reading direction from the install default before the first render.
// The controller pushes a `reading_direction` patch when it has one, but a cast
// that starts before (or without) that patch used to hard-start left-to-right —
// so a manga chapter on the TV read backwards no matter what the reader was set
// to. Any later patch still wins; this only replaces the literal it fell back
// on. Failure is non-fatal: 'ltr' is also the server's own default.
async function fetchDefaultDirection() {
  try {
    const r = await fetch('/api/config/section/comic', { credentials: 'same-origin' });
    if (!r.ok) return;
    const d = (await r.json())?.comic_default_reading_direction;
    if (d === 'ltr' || d === 'rtl') state.readingDirection = d;
  } catch { /* offline / unauthenticated cast — keep the built-in default */ }
}

async function fetchManifest() {
  try {
    const r = await fetch(
      `/api/media/comic/manifest/${encodeURIComponent(FILE_ID)}`,
      { credentials: 'same-origin' },
    );
    if (!r.ok) {
      showError(`Manifest unavailable (HTTP ${r.status})`);
      return;
    }
    const body = await r.json();
    state.pageCount = Number(body.page_count || 0);
    if (body.current_page && body.current_page > 0
        && body.current_page <= state.pageCount) {
      // Resume — but if the saved page is at the very end, snap back
      // to 1 so the first "next" doesn't try to advance off-chapter.
      const isAtEnd = body.is_finished
        || body.current_page >= state.pageCount;
      state.currentPage = isAtEnd ? 1 : body.current_page;
    }
    if (totalEl) totalEl.textContent = state.pageCount || '—';
    if (pageEl) pageEl.textContent = String(state.currentPage);
    updateProgressStrip();
  } catch (err) {
    showError(`Manifest fetch failed: ${err.message || err}`);
  }
}

async function fetchMetadata() {
  try {
    const r = await fetch(
      `/api/media/details/${encodeURIComponent(FILE_ID)}`,
      { credentials: 'same-origin' },
    );
    if (!r.ok) return;
    const body = await r.json();
    const entry = body.entry || body || {};
    const meta = entry.source_metadata || entry.metadata || {};
    const title = entry.name || meta.series || meta.title || 'Comic';
    const subtitle = meta.series && meta.series !== title ? meta.series : '';
    state.title = subtitle ? `${title} — ${subtitle}` : title;
    if (titleEl) titleEl.textContent = state.title;
    // Capture series_id (added to the details response specifically for
    // sibling-chapter awareness). May be empty for one-off / non-series
    // comics — in that case fetchSiblings short-circuits and the cast
    // surface keeps its pre-feature behavior (stops at end-of-chapter).
    state.seriesId = String(body.series_id || entry.series_id || '').trim();
  } catch { /* best-effort */ }
}

// Fetch every chapter in the current series, ordered by source order
// (chapter 1 → N). Mirrors what the web reader's chapterCache holds —
// same backend endpoint, same ordering. Populates ``state.siblings``
// and finds the current chapter's index. No-op when seriesId is
// missing (one-off comic) or the fetch fails (network blip during
// boot just leaves cast in pre-feature behavior — at-end stops).
async function fetchSiblings() {
  if (!state.seriesId) return;
  try {
    const r = await fetch(
      `/api/files/comics/series/${encodeURIComponent(state.seriesId)}/chapters?limit=2000`,
      { credentials: 'same-origin' },
    );
    if (!r.ok) return;
    const body = await r.json();
    const files = Array.isArray(body.files) ? body.files : [];
    if (files.length === 0) return;
    state.siblings = files;
    state.siblingIndex = files.findIndex((c) => c && c.id === FILE_ID);
  } catch {
    // Best-effort — leave siblings null and the cast falls back to
    // stop-at-end-of-chapter.
  }
}

// ── Progress persistence ────────────────────────────────────────
//
// Throttled POST to /api/media/progress so the Continue rail on
// cast-home + cast-control orders this chapter by "most recently
// read on the TV" (via the last_played_at column), AND so reopening
// a chapter resumes on the saved page. The /progress endpoint has
// comic-specific handling that maps current_time_s → page number
// and duration_s → page count, mirroring those into source_metadata.
// extra.current_page where fetchManifest reads them back.
//
// Page-driven (not time-driven): we post on every showPage that
// actually moved the page index, throttled to one POST per 8s for
// the happy path (a slow auto-flip cadence is fine; a binge-skim
// shouldn't pelt the endpoint). Force-flushes happen on chapter
// transition (records is_finished for the OLD chapter before FILE_ID
// swaps) and on pagehide (sendBeacon).

const PROGRESS_POST_INTERVAL_MS = 8 * 1000;
let _lastProgressAt = 0;
let _lastProgressPage = 0;

function postProgress({ force = false, finished = false } = {}) {
  if (!FILE_ID || !state.pageCount) return;
  const page = state.currentPage | 0;
  if (page < 1) return;
  if (!force && page === _lastProgressPage) return;
  if (!force && Date.now() - _lastProgressAt < PROGRESS_POST_INTERVAL_MS) return;
  _lastProgressAt = Date.now();
  _lastProgressPage = page;
  const isFinished = finished || page >= state.pageCount;
  fetch(`/api/media/progress/${encodeURIComponent(FILE_ID)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({
      current_time_s: page,
      duration_s: state.pageCount,
      is_finished: isFinished,
    }),
  }).catch((err) => {
    console.warn('[cast-comic] progress post failed', err);
  });
}

// Final beat on teardown — sendBeacon survives the pagehide that a
// regular fetch can't. Skipped before the manifest has hydrated
// (page=0 would wipe the saved position). Safari (especially iOS in
// low-power mode) sometimes refuses the queue and returns false; the
// keepalive ``fetch`` fallback covers that path so reading progress
// isn't lost.
window.addEventListener('pagehide', () => {
  if (!FILE_ID || !state.pageCount || !state.currentPage) return;
  const url = `/api/media/progress/${encodeURIComponent(FILE_ID)}`;
  const payload = JSON.stringify({
    current_time_s: state.currentPage,
    duration_s: state.pageCount,
    is_finished: state.currentPage >= state.pageCount,
  });
  let sent = false;
  try {
    sent = navigator.sendBeacon(
      url,
      new Blob([payload], { type: 'application/json' }),
    );
  } catch { /* sendBeacon unavailable or refused */ }
  if (sent) return;
  try {
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload,
      keepalive: true,
    }).catch(() => {});
  } catch { /* nothing else to try */ }
});


// Transition the surface to the next chapter in the series. Resets
// per-chapter state (page cache, sliver tracking, aspect samples,
// current page), rewrites FILE_ID, refetches manifest, and resumes
// display from page 1. Autoplay state is preserved across the
// transition so a user reading on autoplay rolls straight into
// the next chapter without intervention. Idempotent while a
// transition is in flight (the autoplay tick can fire again before
// the new chapter's first page has rendered).
async function transitionToChapter(nextFileId) {
  if (!nextFileId || state.chapterTransitionPending) return false;
  state.chapterTransitionPending = true;
  // Force-record the outgoing chapter as finished BEFORE we swap
  // FILE_ID — otherwise the progress writer would credit the OLD
  // chapter's completion to the NEW chapter's id. Auto-advance only
  // fires at end-of-chapter so is_finished=true is correct here.
  postProgress({ force: true, finished: true });
  _lastProgressPage = 0;   // reset throttle gate for the new chapter
  // Stop the current autoplay engine BEFORE swapping FILE_ID so the
  // old tick can't accidentally re-fire against new state mid-swap.
  // We re-arm at the end if autoplay was active.
  const wasAutoplayOn = _isAutoplayActiveForMode();
  const savedAutoplayMs = state.autoplayMs;
  const savedAutoplayPxPerSec = state.autoplayPxPerSec;
  _stopAutoplayEngines();
  // Reset per-chapter state. Page cache + blob URLs would otherwise
  // leak across the chapter boundary (new chapter's page 1 would
  // serve stale bytes from old chapter's page 1 cache slot).
  for (const url of blobUrls.values()) {
    try { URL.revokeObjectURL(url); } catch { /* harmless */ }
  }
  blobUrls.clear();
  pageCache.clear();
  inflight.clear();
  state.sliverPages.clear();
  state.aspectSamples = [];
  state.currentPage = 1;
  state.pageCount = 0;
  state.failedPage = 0;
  state.loadToken += 1;       // invalidates any in-flight load callbacks
  if (stage) stage.innerHTML = '';
  FILE_ID = nextFileId;
  // Update local sibling index so the next end-of-chapter advance
  // knows where it is. Refresh from the existing list rather than
  // refetching — chapters list is stable within a session.
  if (Array.isArray(state.siblings)) {
    state.siblingIndex = state.siblings.findIndex((c) => c && c.id === nextFileId);
  }
  await Promise.all([fetchManifest(), fetchMetadata()]);
  await showPage(state.currentPage, { immediate: true });
  flashHud();
  // Re-arm autoplay with the user's saved cadence if it was running.
  if (wasAutoplayOn) {
    state.autoplayMs = savedAutoplayMs;
    state.autoplayPxPerSec = savedAutoplayPxPerSec;
    startAutoplayForMode();
    _refreshAutoplayPresentation();
  }
  state.chapterTransitionPending = false;
  return true;
}

// Called from the two end-of-chapter sites (discrete page-flip tick
// and continuous-scroll tick) when autoplay was the thing that
// reached the end. If a next chapter exists in the sibling list,
// transitions to it; otherwise stops autoplay (pre-feature behavior).
// Auto-advance is explicitly gated on autoplay being on per design —
// the user asked for "auto-advance only with autoplay" so a controller
// next-page tap at chapter end stays a stopping point.
//
// Cooldown guard prevents the cascading-skip bug: the webtoon rAF
// tick fires before a freshly-loaded chapter's strip has laid out,
// reads scrollHeight≈clientHeight as "end of chapter", and would
// otherwise trigger another transition immediately — repeatedly,
// through every chapter, in a fraction of a second. ``state.chapter
// TransitionPending`` is the in-flight guard; the timestamp cooldown
// catches the case where the transition resolved fast enough that
// pending cleared before the next bogus end detection.
function handleEndOfChapterAutoplay() {
  const now = Date.now();
  if (state.chapterTransitionPending) return;
  if (now - state.lastChapterTransitionAt < CHAPTER_TRANSITION_COOLDOWN_MS) {
    // Suspicious back-to-back end detection — most likely a layout race
    // on a freshly-transitioned chapter. Stop autoplay rather than
    // either advancing again (bug) or silently looping. The user can
    // re-engage autoplay if they want to continue.
    configureAutoplay(state.mode === 'webtoon' ? { pxPerSec: 0 } : { ms: 0 });
    return;
  }
  if (!Array.isArray(state.siblings) || state.siblingIndex < 0) {
    // No sibling list known — keep the pre-feature behavior.
    configureAutoplay(state.mode === 'webtoon' ? { pxPerSec: 0 } : { ms: 0 });
    return;
  }
  const nextSibling = state.siblings[state.siblingIndex + 1];
  if (!nextSibling || !nextSibling.id) {
    // Last chapter in series.
    configureAutoplay(state.mode === 'webtoon' ? { pxPerSec: 0 } : { ms: 0 });
    return;
  }
  // Stamp BEFORE firing — the async transitionToChapter sets
  // chapterTransitionPending synchronously, but the timestamp also
  // needs to be in place before any rAF tick could re-enter this
  // function.
  state.lastChapterTransitionAt = now;
  // Fire-and-forget. Errors inside the transition are swallowed by
  // the helper; if it fails the surface stays on the current
  // chapter's last page and autoplay is already stopped by the
  // transition's own _stopAutoplayEngines call.
  transitionToChapter(nextSibling.id);
}


// ── Page render (single / dual) ────────────────────────────────

async function showPage(pageNum, { immediate = false } = {}) {
  pageNum = clamp(pageNum, 1, state.pageCount || pageNum);
  state.currentPage = pageNum;
  if (pageEl) pageEl.textContent = String(pageNum);
  updateProgressStrip();
  flashHud();
  hideError();
  postProgress();

  if (state.mode === 'webtoon') {
    scrollWebtoonTo(pageNum);
    schedulePrefetch();
    return;
  }

  const token = ++state.loadToken;
  armSpinner();

  if (state.mode === 'dual') {
    // Compose two facing pages. Convention: requested pageNum is the
    // LEFT page in LTR, RIGHT in RTL. The cover (page 1) is solo, and
    // a pair containing a sliver collapses to solo so the screen isn't
    // half-wasted by provider-split residue.
    let left = pageNum, right = pageNum + 1;
    if (pageNum === 1) {
      await mountSinglePage(pageNum, token, immediate);
      schedulePrefetch();
      return;
    }
    if (pageNum % 2 === 1) left = pageNum - 1;
    right = left + 1;
    if (right > state.pageCount) {
      await mountSinglePage(left, token, immediate);
      schedulePrefetch();
      return;
    }
    // Sliver-aware pairing: if exactly one side is a sliver, show the
    // legit side solo and skip the sliver entirely. If both are
    // slivers, jump to the next pair-aligned non-sliver page.
    const leftSliver = state.sliverPages.has(left);
    const rightSliver = state.sliverPages.has(right);
    if (leftSliver && rightSliver) {
      const skipTo = skipSlivers(right + 1, 1);
      state.currentPage = skipTo;
      return showPage(skipTo, { immediate });
    }
    if (rightSliver) {
      // Clear the dual layout's leftover rightSlot — otherwise
      // mountSinglePage reuses leftSlot via ensureSingleSlot but the
      // stale rightSlot stays in the DOM showing the previous page,
      // producing "left changes, right stuck" on every navigation.
      stage.innerHTML = '';
      state.currentPage = left;
      if (pageEl) pageEl.textContent = String(left);
      await mountSinglePage(left, token, immediate);
      schedulePrefetch();
      return;
    }
    if (leftSliver) {
      stage.innerHTML = '';
      state.currentPage = right;
      if (pageEl) pageEl.textContent = String(right);
      await mountSinglePage(right, token, immediate);
      schedulePrefetch();
      return;
    }
    state.currentPage = left;
    if (pageEl) pageEl.textContent = `${left}–${right}`;
    await mountDualPages(left, right, token, immediate);
    schedulePrefetch();
    return;
  }

  await mountSinglePage(pageNum, token, immediate);
  schedulePrefetch();
}

async function mountSinglePage(pageNum, token, immediate) {
  let payload;
  try {
    payload = cacheGet(pageNum) || await fetchPage(pageNum);
    if (token !== state.loadToken) return;  // superseded
    cacheSet(pageNum, payload);
    recordAspectSample(payload.aspectRatio, pageNum);
  } catch (err) {
    if (token !== state.loadToken) return;
    if (err.name === 'AbortError') return;
    disarmSpinner();
    showError(`Page ${pageNum} couldn't load`);
    return;
  }
  disarmSpinner();
  mountSlot(stage, payload, immediate);
  maybePickAutoMode();
}

async function mountDualPages(leftN, rightN, token, immediate) {
  let leftPayload, rightPayload;
  try {
    [leftPayload, rightPayload] = await Promise.all([
      cacheGet(leftN) || fetchPage(leftN),
      cacheGet(rightN) || fetchPage(rightN),
    ]);
    if (token !== state.loadToken) return;
    cacheSet(leftN, leftPayload);
    cacheSet(rightN, rightPayload);
    recordAspectSample(leftPayload.aspectRatio, leftN);
    recordAspectSample(rightPayload.aspectRatio, rightN);
  } catch (err) {
    if (token !== state.loadToken) return;
    if (err.name === 'AbortError') return;
    disarmSpinner();
    showError(`Pages ${leftN}–${rightN} couldn't load`);
    return;
  }
  disarmSpinner();

  // Build two slots fresh. Reading direction is handled by CSS
  // (flex-direction: row-reverse) so we always append left-first.
  stage.innerHTML = '';
  const leftSlot = document.createElement('div');
  leftSlot.className = 'page-slot';
  leftSlot.dataset.page = String(leftN);
  const rightSlot = document.createElement('div');
  rightSlot.className = 'page-slot';
  rightSlot.dataset.page = String(rightN);
  stage.appendChild(leftSlot);
  stage.appendChild(rightSlot);
  mountSlot(leftSlot, leftPayload, immediate);
  mountSlot(rightSlot, rightPayload, immediate);
  maybePickAutoMode();
}

/** Append a fresh <img> from the payload to the slot, fade out any
 *  previous img, then remove it. Each navigation creates a new DOM
 *  node so there's no per-element state to corrupt across page turns —
 *  this is the structural fix for the every-other-page failure. */
function mountSlot(parentEl, payload, immediate) {
  const slot = parentEl.classList.contains('page-slot')
    ? parentEl
    : ensureSingleSlot(parentEl);

  // Border crop: cheap (downsampled detection), gives noticeably more
  // screen real estate on scanned manga. Memoized per-payload — the
  // detect+canvas pass is ~50–100 ms on TV-class CPUs and shouldn't
  // re-run on every prev/next cycle.
  let displaySrc = payload.blobUrl;
  if (state.borderCrop) {
    if (payload.cropChecked === undefined) {
      const insets = detectBorders(payload.img);
      payload.cropDataUrl = insets ? cropImage(payload.img, insets) : null;
      payload.cropChecked = true;
    }
    if (payload.cropDataUrl) displaySrc = payload.cropDataUrl;
  }

  const img = document.createElement('img');
  img.alt = `Page ${payload.page}`;
  img.decoding = 'async';
  img.src = displaySrc;

  // Fade out the existing img(s), then remove. The new img starts at
  // opacity 0 and is promoted to is-active on next frame so the
  // crossfade actually plays (without the rAF the transition skips on
  // some engines because the initial value matches the target).
  const old = slot.querySelectorAll('img.is-active');
  for (const o of old) {
    o.classList.add('is-leaving');
    o.classList.remove('is-active');
  }
  slot.appendChild(img);
  if (immediate) {
    img.classList.add('is-active');
    cleanupLeavers(slot, 0);
  } else {
    requestAnimationFrame(() => {
      img.classList.add('is-active');
      cleanupLeavers(slot, 380);
    });
  }
}

function ensureSingleSlot(parentEl) {
  let slot = parentEl.querySelector('.page-slot');
  if (!slot) {
    parentEl.innerHTML = '';
    slot = document.createElement('div');
    slot.className = 'page-slot';
    parentEl.appendChild(slot);
  }
  return slot;
}

function cleanupLeavers(slot, delay) {
  if (delay <= 0) {
    slot.querySelectorAll('img.is-leaving').forEach((o) => o.remove());
    return;
  }
  setTimeout(() => {
    slot.querySelectorAll('img.is-leaving').forEach((o) => o.remove());
  }, delay);
}


// ── Page render (webtoon) ──────────────────────────────────────
// Vertical strip. We render all pages into one column at full width,
// rely on the browser's native lazy decode (loading="lazy" via the
// underlying <img>), and scroll the stage to the requested page.

// Webtoon controller state — kept here (not in `state`) because all of
// it is mode-scoped and rebuilt on mode entry. The IntersectionObserver
// is used ONLY for mount triggering (when to start fetching a slot's
// image), not for currentPage tracking — the latter is computed
// deterministically from scrollTop because IO entries only contain
// changed-state slots, which produces wrong "most-visible" guesses
// during layout-shift cascades when many slots mount near-simultaneously.
let webtoonObserver = null;
let webtoonScrollLockUntil = 0;
let webtoonScrollRaf = 0;

function renderWebtoon() {
  stage.innerHTML = '';
  for (let p = 1; p <= state.pageCount; p++) {
    const slot = document.createElement('div');
    slot.className = 'page-slot webtoon-slot';
    slot.dataset.page = String(p);
    // Reserve vertical space so empty slots actually occupy layout
    // before their images load. Cleared on mount. 80vh is enough that
    // the observer sees nearby slots without overshooting.
    slot.style.minHeight = '80vh';
    stage.appendChild(slot);
  }
  setupWebtoonObserver();
  setupWebtoonScrollTracker();
  // Mount the target slot AND a couple of its neighbours synchronously
  // so the scroll target's true offsetTop is established before we
  // navigate to it. Without this, scrollTo lands at the placeholder y
  // and as nearby slots grow the user ends up anywhere — that's the
  // "skipped from page 7 to page 45" bug.
  const target = clamp(state.currentPage, 1, state.pageCount || 1);
  primeWebtoonSlots([target - 1, target, target + 1, target + 2])
    .finally(() => {
      scrollWebtoonTo(target, { instant: true });
      // Race ahead so the user has a few seconds of content buffered
      // before the IntersectionObserver even fires for further slots.
      _kickWebtoonPrefetch();
    });
}

function setupWebtoonObserver() {
  if (webtoonObserver) webtoonObserver.disconnect();
  // Observer fires solely to trigger lazy mounting for slots near the
  // current scroll position. currentPage updates are handled separately
  // by the scroll tracker.
  webtoonObserver = new IntersectionObserver((entries) => {
    for (const ent of entries) {
      if (!ent.isIntersecting) continue;
      const pageNum = Number(ent.target.dataset.page || 0);
      if (!pageNum) continue;
      mountWebtoonSlot(ent.target, pageNum);
    }
  // 2400 px ≈ 2 viewport-heights on 1080p — at Fast (400 px/s) the
  // pump has 6 s of buffered images by the time a slot enters frame.
  // 800 px gave only ~2 s and left visible white gaps on fast swipes.
  }, { root: stage, rootMargin: '2400px 0px', threshold: 0 });
  stage.querySelectorAll('.webtoon-slot').forEach((s) => webtoonObserver.observe(s));
}

function setupWebtoonScrollTracker() {
  stage.addEventListener('scroll', onWebtoonScroll, { passive: true });
}

function onWebtoonScroll() {
  if (webtoonScrollRaf) cancelAnimationFrame(webtoonScrollRaf);
  webtoonScrollRaf = requestAnimationFrame(() => {
    webtoonScrollRaf = 0;
    // Suppress updates during a programmatic scroll — otherwise the
    // smooth-scroll animation triggers rapid scrollTop changes that
    // bounce currentPage through every slot we pass over.
    if (performance.now() < webtoonScrollLockUntil) return;
    const page = pageAtViewportCenter();
    if (page && page !== state.currentPage) {
      state.currentPage = page;
      if (pageEl) pageEl.textContent = String(page);
      updateProgressStrip();
      postProgress();
      // Warm pages ahead so a fast swipe doesn't outrun the observer.
      // Cheap — fetchPage is idempotent via inflight + pageCache.
      _kickWebtoonPrefetch();
    }
  });
}

/** Fire-and-forget prefetch of the next WEBTOON_PREFETCH_AHEAD pages.
 *  Stacks with the IntersectionObserver's lazy mount, but reaches
 *  further: at 5 pages ahead we're priming content the user won't
 *  see for several seconds, which absorbs ~one viewport of network
 *  jitter without a visible gap. */
function _kickWebtoonPrefetch() {
  for (let i = 1; i <= WEBTOON_PREFETCH_AHEAD; i++) {
    const n = state.currentPage + i;
    if (n < 1) continue;
    if (state.pageCount && n > state.pageCount) continue;
    if (pageCache.has(n) || inflight.has(n)) continue;
    fetchPage(n).then((p) => {
      cacheSet(n, p);
      recordAspectSample(p.aspectRatio, n);
    }).catch(() => { /* prefetch failures are silent */ });
  }
}

/** Compute which slot's midpoint is closest to the viewport center.
 *  Walks slots once — O(n) but only on scroll-stable frames, and n is
 *  the chapter page count (typically <100). */
function pageAtViewportCenter() {
  const viewportTop = stage.scrollTop;
  const viewportCenter = viewportTop + stage.clientHeight / 2;
  let best = 0;
  let bestDist = Infinity;
  for (const slot of stage.querySelectorAll('.webtoon-slot')) {
    const top = slot.offsetTop;
    const mid = top + slot.offsetHeight / 2;
    const dist = Math.abs(mid - viewportCenter);
    if (dist < bestDist) {
      bestDist = dist;
      best = Number(slot.dataset.page || 0);
    }
    if (top > viewportCenter + stage.clientHeight) break;  // early exit
  }
  return best;
}

/** Force-mount the given pages synchronously (in parallel). Used to
 *  stabilize the layout around a scroll target before we navigate to
 *  it. Out-of-range page indices are silently dropped. */
async function primeWebtoonSlots(pageNums) {
  const valid = pageNums.filter(
    (n) => n >= 1 && (!state.pageCount || n <= state.pageCount),
  );
  const slots = valid.map(
    (n) => [n, stage.querySelector(`.webtoon-slot[data-page="${n}"]`)],
  ).filter(([, s]) => s);
  await Promise.all(slots.map(([n, s]) => mountWebtoonSlot(s, n)));
}

async function mountWebtoonSlot(slot, pageNum) {
  if (slot.dataset.mounted === '1') return;
  slot.dataset.mounted = '1';
  try {
    const payload = cacheGet(pageNum) || await fetchPage(pageNum);
    cacheSet(pageNum, payload);
    recordAspectSample(payload.aspectRatio, pageNum);
    const img = document.createElement('img');
    img.alt = `Page ${pageNum}`;
    img.decoding = 'async';
    img.src = payload.blobUrl;
    img.classList.add('is-active');
    slot.appendChild(img);
    slot.style.minHeight = '';  // image dictates height now
  } catch (err) {
    if (err.name === 'AbortError') return;
    slot.dataset.mounted = '0';  // allow retry on re-intersect
    slot.innerHTML = `<div class="webtoon-fail">Page ${pageNum} failed — scroll to retry</div>`;
  }
}

function scrollWebtoonTo(pageNum, { instant = false } = {}) {
  const slot = stage.querySelector(`.webtoon-slot[data-page="${pageNum}"]`);
  if (!slot) return;
  // Force a layout flush before reading offsetTop — defensive against
  // TV browsers that occasionally hand back a stale value mid-mount.
  void slot.offsetHeight;
  const targetTop = slot.offsetTop;
  // Lock currentPage tracking for the duration of the scroll — otherwise
  // every slot we pass through during the animation gets briefly marked
  // as current, ending with the wrong page when the scroll settles.
  webtoonScrollLockUntil = performance.now() + (instant ? 250 : 700);
  state.currentPage = pageNum;
  if (pageEl) pageEl.textContent = String(pageNum);
  updateProgressStrip();
  // Direct scrollTo on the stage. scrollIntoView is finickier on smart-
  // TV browsers (some don't honour {behavior: 'smooth'}, some pick a
  // surprising scroll ancestor when nested layouts are involved).
  stage.scrollTo({ top: targetTop, behavior: instant ? 'auto' : 'smooth' });
}


// ── Prefetch ───────────────────────────────────────────────────

function schedulePrefetch() {
  if (state.mode === 'webtoon') return;  // observer handles its own
  const targets = [];
  for (let i = 1; i <= PREFETCH_AHEAD; i++) targets.push(state.currentPage + i);
  for (let i = 1; i <= PREFETCH_BEHIND; i++) targets.push(state.currentPage - i);
  if (state.mode === 'dual') {
    // Dual mounts two at a time, so prefetch the next *pair* primarily.
    targets.push(state.currentPage + 2, state.currentPage + 3);
  }
  for (const n of targets) {
    if (n < 1) continue;
    if (state.pageCount && n > state.pageCount) continue;
    if (pageCache.has(n)) continue;
    if (inflight.has(n)) continue;
    // Prefetch + classify. The classify step is essential: the next()
    // path checks state.sliverPages to skip provider-split residue, and
    // we need that classification BEFORE the user navigates onto the
    // sliver — otherwise the skip kicks in one tap late.
    fetchPage(n).then((payload) => {
      cacheSet(n, payload);
      recordAspectSample(payload.aspectRatio, n);
    }).catch(() => { /* prefetch failures are silent */ });
  }
}


// ── Aspect sampling ────────────────────────────────────────────

function recordAspectSample(ar, pageNum) {
  if (!ar || !isFinite(ar)) return;
  state.aspectSamples.push(ar);
  if (state.aspectSamples.length > ASPECT_SAMPLE_COUNT * 4) {
    state.aspectSamples.shift();
  }
  // Sliver classification: aspect > 2.2 means very wide-short, which
  // for paged content is almost always provider-split webtoon residue
  // (the leftover fragment after a fixed-height chunk boundary).
  if (pageNum && ar > 2.2) {
    state.sliverPages.add(pageNum);
    // If a sliver appears after we already picked single-mode under
    // auto, re-evaluate. The original sample may have missed it
    // (slivers often appear at chapter ends, not the first few
    // pages we sample on mount).
    if (state.modeRequested === 'auto' && state.mode === 'single') {
      setMode('webtoon');
    }
  }
  // Webtoon mode: recompute the column width from non-sliver aspects
  // so slivers visually conform to the page column instead of
  // sprawling across the full viewport.
  if (state.mode === 'webtoon') updateWebtoonColumnWidth();
}

/** Set the webtoon column width as a CSS custom property on the
 *  stage. The column is sized so that a median-aspect page renders at
 *  viewport-height tall, which keeps the column at a comfortable size
 *  on any TV and forces slivers to stack at the same visual width.
 *  Only normal-aspect pages are considered — sliver aspects would
 *  pull the median wide and produce a tiny column.
 *
 *  Throttled + diff-gated: setting --webtoon-col-width re-lays out
 *  every slot in the strip, so calling it on every prefetched
 *  aspect sample was burning ~5 full-stage layouts per fetch burst
 *  for negligible visual change once the median stabilizes. We
 *  cap to one update per 250 ms AND skip writes that move the
 *  column by < 1 % of its current value (well below perceptual
 *  threshold). */
let _colWidthLastUpdateAt = 0;
let _colWidthLast = 0;
function updateWebtoonColumnWidth() {
  const now = performance.now();
  if (now - _colWidthLastUpdateAt < 250) return;
  const normal = state.aspectSamples.filter((a) => a > 0 && a < 1.5);
  if (normal.length === 0) return;
  const sorted = [...normal].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const viewportH = window.innerHeight || 1080;
  // width / height = aspect → width at viewport-height height.
  // Clamped to the viewport's actual width so a very-wide aspect
  // (panel-spread page) doesn't blow past the TV's horizontal extent.
  const colWidth = Math.min(
    Math.round(viewportH * median),
    window.innerWidth || 1920,
  );
  if (_colWidthLast && Math.abs(colWidth - _colWidthLast) / _colWidthLast < 0.01) {
    return;  // sub-1% change isn't worth a relayout
  }
  _colWidthLastUpdateAt = now;
  _colWidthLast = colWidth;
  stage.style.setProperty('--webtoon-col-width', `${colWidth}px`);
}

/** Walk forward/back from `from` skipping any sliver pages until we
 *  land on a real one. Bounded by [1, pageCount]; if no real page
 *  exists in the requested direction, returns `from` unchanged so the
 *  caller's clamp logic handles end-of-chapter. */
function skipSlivers(from, step) {
  let n = from;
  while (state.sliverPages.has(n)
         && n >= 1
         && (!state.pageCount || n <= state.pageCount)) {
    n += step;
  }
  return n;
}


// ── Navigation ─────────────────────────────────────────────────

function next() {
  const step = state.mode === 'dual' && state.currentPage > 1 ? 2 : 1;
  // Skip sliver pages — they're provider-split residue that just shows
  // an empty letterboxed band in paged mode. Webtoon mode doesn't go
  // through this path (scrolling handles everything continuously).
  const target = skipSlivers(state.currentPage + step, 1);
  showPage(target);
}
function prev() {
  const step = state.mode === 'dual' && state.currentPage > 2 ? 2 : 1;
  const target = skipSlivers(Math.max(1, state.currentPage - step), -1);
  showPage(target);
}


// ── Autoplay + countdown ───────────────────────────────────────

/** Mode-aware autoplay configuration. Accepts either form:
 *    configureAutoplay({ ms: 15000 })       — paged/dual page-flip
 *    configureAutoplay({ pxPerSec: 90 })    — webtoon continuous scroll
 *  Both values are remembered separately so a user who flips between
 *  modes keeps their preferred speed in each. The active engine is
 *  picked based on state.mode. Legacy callers pass a bare ms number
 *  for backward compat. */
function configureAutoplay(arg) {
  const params = typeof arg === 'number' ? { ms: arg } : (arg || {});
  if (typeof params.ms === 'number') state.autoplayMs = Math.max(0, params.ms);
  if (typeof params.pxPerSec === 'number') {
    state.autoplayPxPerSec = Math.max(0, params.pxPerSec);
  }
  _stopAutoplayEngines();
  _refreshAutoplayPresentation();
  startAutoplayForMode();
}

function startAutoplayForMode() {
  if (state.autoplayPaused) return;
  if (state.mode === 'webtoon') {
    if (state.autoplayPxPerSec > 0) startContinuousScroll();
  } else {
    if (state.autoplayMs > 0) startDiscreteCycle();
  }
}

function _stopAutoplayEngines() {
  clearTimeout(state.autoplayTimer);
  clearInterval(state.autoplayProgressTimer);
  state.autoplayTimer = null;
  state.autoplayProgressTimer = null;
  if (state.autoplayRaf) cancelAnimationFrame(state.autoplayRaf);
  state.autoplayRaf = 0;
}

/** Whether the current mode's autoplay engine has a non-zero speed. */
function _isAutoplayActiveForMode() {
  return state.mode === 'webtoon'
    ? state.autoplayPxPerSec > 0
    : state.autoplayMs > 0;
}

/** HUD + countdown strip + wake-lock — depends on whether the active
 *  mode's engine is configured (not whether ANY engine has a speed).
 *  The countdown strip only makes sense for discrete mode; continuous
 *  scroll provides its own moving-art feedback. */
function _refreshAutoplayPresentation() {
  const active = _isAutoplayActiveForMode();
  if (!active) {
    autoplayStrip.hidden = true;
    autoplayFill.style.width = '0%';
    setHudPersistent(false);
    releaseWakeLock();
    if (speedChipEl) speedChipEl.hidden = true;
    return;
  }
  setHudPersistent(true);
  requestWakeLock();
  if (state.mode === 'webtoon') {
    // Continuous mode — hide the per-page countdown strip; the comic
    // itself scrolling is the feedback. Show the speed chip instead
    // so users can see at a glance what px/sec they're at.
    autoplayStrip.hidden = true;
    autoplayFill.style.width = '0%';
    _updateSpeedChip();
  } else {
    autoplayStrip.hidden = false;
    if (speedChipEl) speedChipEl.hidden = true;
  }
}

/** Render the webtoon speed indicator. Maps the active px/sec value
 *  to its sheet-row label (Slow/Med/Fast) when it matches a preset,
 *  otherwise just shows the raw value. Custom speeds (set via a
 *  future +/- nudge) still get a readable readout. */
function _updateSpeedChip() {
  if (!speedChipEl) return;
  const px = state.autoplayPxPerSec;
  if (!px) {
    speedChipEl.hidden = true;
    return;
  }
  const labels = { 120: 'SLOW', 240: 'MED', 400: 'FAST' };
  const label = labels[px] || '';
  speedChipEl.textContent = label
    ? `↓ ${px} px/s · ${label}`
    : `↓ ${px} px/s`;
  speedChipEl.hidden = false;
}

/** Paged / dual engine — setTimeout-driven page flip + progress
 *  countdown bar. Same behavior as before, just renamed. */
function startDiscreteCycle() {
  if (!state.autoplayMs || state.autoplayPaused) return;
  state.autoplayStartedAt = performance.now();
  autoplayFill.style.width = '0%';
  state.autoplayTimer = setTimeout(() => {
    if (state.autoplayPaused) return;
    if (state.pageCount && state.currentPage >= state.pageCount) {
      // End-of-chapter: auto-advance to next sibling if available
      // (handleEndOfChapterAutoplay falls back to stopping autoplay
      // when there's no next chapter / no sibling list).
      handleEndOfChapterAutoplay();
      return;
    }
    next();
    startDiscreteCycle();
  }, state.autoplayMs);

  state.autoplayProgressTimer = setInterval(() => {
    const elapsed = performance.now() - state.autoplayStartedAt;
    const pct = clamp((elapsed / state.autoplayMs) * 100, 0, 100);
    autoplayFill.style.width = `${pct}%`;
  }, Math.round(1000 / AUTOPLAY_PROGRESS_HZ));
}

/** Webtoon engine — requestAnimationFrame loop that scrolls the stage
 *  by velocity × Δt every frame. Stops on pause, end-of-strip, or
 *  mode change. Uses behavior:'auto' on scrollBy because the
 *  per-frame deltas are small enough that the browser's smooth-scroll
 *  smoothing would lag behind the next tick. */
function startContinuousScroll() {
  if (!state.autoplayPxPerSec || state.autoplayPaused) return;
  if (state.autoplayRaf) return;  // already running
  state.autoplayRafLast = performance.now();
  const tick = (now) => {
    state.autoplayRaf = 0;
    if (state.autoplayPaused) return;
    if (state.mode !== 'webtoon') return;
    if (state.autoplayPxPerSec <= 0) return;
    const dt = Math.min(now - state.autoplayRafLast, 100);
    state.autoplayRafLast = now;
    const px = state.autoplayPxPerSec * (dt / 1000);
    if (px > 0.1) {
      stage.scrollBy({ top: px, behavior: 'auto' });
    }
    // End-of-chapter: reached the bottom (within a small slop). Also
    // require the strip to be meaningfully scrollable past the
    // viewport — an empty / still-loading strip has scrollHeight ≈
    // clientHeight and would otherwise read as "at end" on the first
    // rAF tick of a freshly-transitioned chapter, triggering a
    // cascading rapid-skip through the series. The threshold is large
    // enough to distinguish "we've actually reached the bottom" from
    // "the strip hasn't laid out yet."
    if (stage.scrollHeight - stage.clientHeight >= WEBTOON_SCROLLABLE_MIN_PX
        && stage.scrollTop + stage.clientHeight >= stage.scrollHeight - 4) {
      handleEndOfChapterAutoplay();
      return;
    }
    state.autoplayRaf = requestAnimationFrame(tick);
  };
  state.autoplayRaf = requestAnimationFrame(tick);
}

function pauseAutoplay() {
  state.autoplayPaused = true;
  _stopAutoplayEngines();
}

function resumeAutoplay() {
  state.autoplayPaused = false;
  startAutoplayForMode();
}


// ── Wake lock ──────────────────────────────────────────────────

async function requestWakeLock() {
  if (state.wakeLock) return;
  if (!('wakeLock' in navigator)) return;
  try {
    state.wakeLock = await navigator.wakeLock.request('screen');
    state.wakeLock.addEventListener('release', () => { state.wakeLock = null; });
  } catch { /* permission denied; harmless */ }
}

async function releaseWakeLock() {
  if (!state.wakeLock) return;
  try { await state.wakeLock.release(); } catch {}
  state.wakeLock = null;
}

document.addEventListener('visibilitychange', () => {
  // Wake locks are released when the page is hidden; re-acquire when
  // it comes back if either autoplay engine is configured.
  if (document.visibilityState === 'visible' && _isAutoplayActiveForMode()) {
    requestWakeLock();
  }
});

// Re-derive the webtoon column when the TV reflows — rare on direct-
// cast surfaces but happens routinely with Selkies-streamed Chromium
// when the host browser is resized.
window.addEventListener('resize', () => {
  if (state.mode === 'webtoon') updateWebtoonColumnWidth();
});


// ── Patch handler ──────────────────────────────────────────────

function handlePatch(patch) {
  if (!patch || typeof patch !== 'object') return;

  if (typeof patch.page_idx === 'number') {
    // Absolute page set is explicit user intent — don't auto-skip.
    showPage(Math.round(patch.page_idx));
  }
  if (typeof patch.page_delta === 'number') {
    const delta = Math.round(patch.page_delta);
    // Unit steps (±1) get sliver-aware nav — that's the case where a
    // stranded sliver page makes the tap feel like a no-op. Larger
    // jumps (±5, ±10) are explicit user intent, no skip.
    if (delta === 1) next();
    else if (delta === -1) prev();
    else showPage(state.currentPage + delta);
  }
  if (patch.jump === 'first') showPage(1);
  if (patch.jump === 'last' && state.pageCount) showPage(state.pageCount);

  if (typeof patch.autoplay_ms === 'number') {
    state.autoplayPaused = false;
    configureAutoplay({ ms: patch.autoplay_ms });
  }
  if (typeof patch.autoplay_px_per_sec === 'number') {
    state.autoplayPaused = false;
    configureAutoplay({ pxPerSec: patch.autoplay_px_per_sec });
  }
  if (typeof patch.paused === 'boolean') {
    patch.paused ? pauseAutoplay() : resumeAutoplay();
  }

  if (typeof patch.mode === 'string') {
    const want = patch.mode;
    state.modeRequested = want;
    if (want === 'auto') {
      // Reset samples so re-detection can pick a new mode.
      state.aspectSamples = [];
      setMode('single');
    } else if (['single', 'dual', 'webtoon'].includes(want)) {
      setMode(want);
    }
  }
  if (typeof patch.fit === 'string'
      && ['smart', 'width', 'height', 'native'].includes(patch.fit)) {
    state.fitMode = patch.fit;
    applyStageClasses();
  }
  if (typeof patch.reading_direction === 'string'
      && ['ltr', 'rtl'].includes(patch.reading_direction)) {
    state.readingDirection = patch.reading_direction;
    applyStageClasses();
  }
  if (typeof patch.border_crop === 'boolean') {
    state.borderCrop = patch.border_crop;
    // Re-render current page so crop change takes effect.
    if (state.mode !== 'webtoon') showPage(state.currentPage, { immediate: true });
  }
  if (patch.retry === true) {
    hideError();
    showPage(state.currentPage, { immediate: true });
  }
  if (typeof patch.narrate === 'boolean') {
    if (patch.narrate) startNarration();
    else stopNarration();
  }
  // ── Music-bed control (from the controller / phone) ──
  if (typeof patch.bed_picker === 'boolean') {
    patch.bed_picker ? openBedPicker() : closeBedPicker();
  }
  if (typeof patch.bed_search === 'string') {
    // Controller-driven search (phone keyboard). Opens the picker if needed.
    if (!picker?.open) openBedPicker();
    _loadPickerItems(patch.bed_search.trim());
  }
  if (typeof patch.bed_select === 'number' && picker) {
    picker.sel = clamp(Math.round(patch.bed_select), 0, picker.items.length - 1);
    _pickerActivate();
  }
  if (typeof patch.bed_source === 'object' && patch.bed_source) {
    // Controller resolved a descriptor itself (e.g. via music-source) and
    // hands it straight over — play it as the bed with no picker round-trip.
    startBed(patch.bed_source);
  }
  if (patch.bed_off === true) stopBed();
  if (typeof patch.bed_volume === 'number') setBedVolume(patch.bed_volume);
  if (typeof patch.scroll_delta_px === 'number') {
    handleScrollDelta(patch.scroll_delta_px);
  }
}

// Accumulated scroll-delta for paged / dual modes. Webtoon mode
// applies deltas directly to stage.scrollTop; paged/dual collect
// pixels until |total| ≥ half-viewport, then advance/retreat a page
// and reset. Sign convention matches the phone: positive = forward.
let _pagedScrollAccum = 0;
let _pagedScrollLockUntil = 0;  // brief refractory after a page-flip

// Webtoon scroll-smoothing buffer. The phone sends scroll_delta_px at
// ~30 Hz; applying each one directly to stage.scrollBy produces visibly
// stepped motion (the discrete jumps are ~16–32 ms apart, our compositor
// runs at 60 Hz, so every other frame is a no-op). The buffer + rAF
// pump acts as a one-frame-half-life low-pass: half the pending pixels
// drain per frame, so a burst is naturally distributed over ~6 frames
// (~100 ms) without bloating perceived input latency. Continues to use
// behavior:'auto' on the inner scrollBy because we ARE the smoothing.
let _smoothBufferPx = 0;
let _smoothRaf = 0;

function _smoothScrollPump(px) {
  _smoothBufferPx += px;
  if (!_smoothRaf) _smoothRaf = requestAnimationFrame(_smoothTick);
}

function _smoothTick() {
  _smoothRaf = 0;
  if (Math.abs(_smoothBufferPx) < 0.5) {
    _smoothBufferPx = 0;
    return;
  }
  const sign = Math.sign(_smoothBufferPx);
  // Half-life of one frame: consume 50 % of pending each tick. Floor
  // at 1 px so the tail doesn't get stuck rounding to zero.
  let step = Math.abs(_smoothBufferPx) * 0.5;
  step = Math.max(step, 1);
  step = Math.min(step, Math.abs(_smoothBufferPx));
  step *= sign;
  _smoothBufferPx -= step;
  // Suppress page-tracker thrash while we're actively pumping —
  // each frame's step would otherwise look like a user-driven scroll
  // and cycle currentPage through every slot we cross.
  webtoonScrollLockUntil = performance.now() + 150;
  stage.scrollBy({ top: step, behavior: 'auto' });
  _smoothRaf = requestAnimationFrame(_smoothTick);
}

function handleScrollDelta(px) {
  if (!isFinite(px) || px === 0) return;

  if (state.mode === 'webtoon') {
    // Hand off to the rAF pump — see _smoothScrollPump for rationale.
    _smoothScrollPump(px);
    flashHud(900);
    return;
  }

  // Paged / dual — accumulate, threshold-trigger page flips.
  const now = performance.now();
  if (now < _pagedScrollLockUntil) return;  // recovering from last flip
  _pagedScrollAccum += px;
  const threshold = (window.innerHeight || 800) * 0.5;
  if (_pagedScrollAccum >= threshold) {
    _pagedScrollAccum = 0;
    _pagedScrollLockUntil = now + 250;
    next();
  } else if (_pagedScrollAccum <= -threshold) {
    _pagedScrollAccum = 0;
    _pagedScrollLockUntil = now + 250;
    prev();
  }
}

window.addEventListener('message', (ev) => {
  const data = ev.data;
  if (!data || typeof data !== 'object') return;
  if (data.type === 'augmentum.surface_state') handlePatch(data.patch);
});


// ── Keyboard (HDMI-CEC remotes + dev) ─────────────────────────

window.addEventListener('keydown', (e) => {
  // The bed picker owns the D-pad while open — swallow every key so the
  // reader behind the overlay doesn't also navigate.
  if (picker?.open) { _handlePickerKey(e); return; }
  switch (e.key) {
    case 'b': case 'B':
      e.preventDefault();
      picker?.open ? closeBedPicker() : openBedPicker();
      break;
    case 'ArrowRight':
    case 'PageDown':
    case ' ':
      e.preventDefault();
      state.readingDirection === 'rtl' ? prev() : next();
      break;
    case 'ArrowLeft':
    case 'PageUp':
      e.preventDefault();
      state.readingDirection === 'rtl' ? next() : prev();
      break;
    case 'Home':
      e.preventDefault();
      showPage(1);
      break;
    case 'End':
      e.preventDefault();
      if (state.pageCount) showPage(state.pageCount);
      break;
    case 'r': case 'R':
      hideError();
      showPage(state.currentPage, { immediate: true });
      break;
    case 'n': case 'N':
      e.preventDefault();
      toggleNarration();
      break;
  }
});

if (retryBtn) {
  retryBtn.addEventListener('click', () => {
    hideError();
    showPage(state.currentPage, { immediate: true });
  });
}


// ── Narration (TTS read-along on the TV) ───────────────────────
// The audio-IS-the-clock engine (shared with the in-app reader via
// narration-clock.js) drives page advancement + the spoken-line caption.
// When narration is active it OWNS the page cursor — the discrete/webtoon
// autoplay engines are stood down so there's one clock, not two. Audio is
// routed through AudioBus at the speech tier (see music-bed below) so a
// background music bed ducks under the narration automatically.

// { payload, clock, active, prevMode } once a narration is found; null when
// the comic has no narration (409) and we stay image-only.
let narration = null;

// Spoken-line caption — a TV-legible subtitle for the currently-voiced
// bubble. Created lazily; styled in cast-comic.css.
let narrationCaption = null;
function _ensureCaption() {
  if (narrationCaption) return narrationCaption;
  narrationCaption = document.createElement('div');
  narrationCaption.className = 'cc-narration-caption';
  narrationCaption.hidden = true;
  document.body.appendChild(narrationCaption);
  return narrationCaption;
}

const NARRATION_CAST_URL = () =>
  `/api/comic-narration/${encodeURIComponent(FILE_ID)}/cast`;

/** Probe for a ready narration. 200 → available (kept until the user starts
 *  it, per never-auto-start); 409 → none, stay image-only. */
async function initNarration() {
  try {
    const resp = await fetch(NARRATION_CAST_URL(), {
      method: 'POST', credentials: 'same-origin',
    });
    if (!resp.ok) return;  // 409 not-ready / no narration → image-only
    const payload = await resp.json();
    if (!payload || !Array.isArray(payload.pages) || !payload.pages.length) return;
    narration = { payload, clock: null, active: false, prevMode: null };
    // Discoverability without auto-starting: a brief HUD hint. The controller
    // starts it via a {narrate:true} patch; locally 'N' toggles it.
    _flashNarrationHint();
  } catch { /* offline / unauthenticated cast → no narration */ }
}

function _flashNarrationHint() {
  const hint = _ensureCaption();
  hint.textContent = '🔊 Narration available — press play';
  hint.dataset.kind = 'hint';
  hint.hidden = false;
  flashHud();
  setTimeout(() => { if (narration && !narration.active) hint.hidden = true; }, 5000);
}

function startNarration() {
  if (!narration || narration.active) return;
  narration.active = true;
  // Narration is the clock now — stand down the page-flip / scroll engines.
  _stopAutoplayEngines();
  state.autoplayPaused = true;
  // Pin single-page mode for a clean one-page-at-a-time read-along; remember
  // the prior mode to restore when narration stops.
  narration.prevMode = state.modeRequested;
  state.modeRequested = 'single';
  if (state.mode !== 'single') setMode('single');

  const caption = _ensureCaption();
  narration.clock = createNarrationClock(narration.payload, {
    pollUrl: NARRATION_CAST_URL(),
    // Cinematic pacing on the TV: the art is the show, so give every page a
    // guaranteed beat on screen (no snap-past on a one-line page) and let a
    // splash panel hold as an intentional shot rather than a stall.
    minPageMs: 1400,
    pageCushionMs: 650,
    splashMs: 3600,
    onPage: ({ page }) => {
      // Server pages are 0-indexed; cast pages are 1-indexed.
      showPage((page || 0) + 1, { immediate: true });
    },
    onLine: ({ index, line }) => {
      if (index < 0 || !line || !line.text) { caption.hidden = true; return; }
      caption.textContent = line.text;
      caption.dataset.kind = line.kind || 'speech';
      caption.hidden = false;
    },
    onWaiting: ({ status }) => {
      caption.textContent = status === 'failed' ? 'Narration failed.' : 'Synthesizing…';
      caption.dataset.kind = 'hint';
      caption.hidden = false;
    },
    onFinish: () => {
      caption.hidden = true;
      stopNarration();
    },
  });
  // Route the narration audio through the shared bus at the speech tier so a
  // music bed (if any) ducks under it. No-op when the bed isn't loaded.
  _attachNarrationToBus(narration.clock.audio);
  narration.clock.play();
  requestWakeLock();
  // Bring up the user's chosen bed UNDER the narration. This is the explicit
  // "lofi while the comic plays" coupling — narration (user-triggered) is the
  // trigger, so it's not an unprompted auto-start. persist:false: we're
  // resuming an existing choice, not making a new one.
  if (_preferredBed && !bedDescriptor) startBed(_preferredBed, { persist: false });
}

function stopNarration() {
  if (!narration || !narration.active) return;
  _detachNarrationFromBus();
  narration.clock?.destroy();
  narration.clock = null;
  narration.active = false;
  if (narrationCaption) narrationCaption.hidden = true;
  // The bed rides with the read-along session. Stop it but KEEP the saved
  // choice so it resumes next time narration starts (persist:false).
  stopBed({ persist: false });
  // Restore the pre-narration mode.
  if (narration.prevMode != null) {
    state.modeRequested = narration.prevMode;
    if (narration.prevMode === 'auto') { state.aspectSamples = []; setMode('single'); }
    else if (['single', 'dual', 'webtoon'].includes(narration.prevMode)) setMode(narration.prevMode);
    narration.prevMode = null;
  }
}

function toggleNarration() {
  if (!narration) return;
  narration.active ? stopNarration() : startNarration();
}

// AudioBus hooks are filled in by the music-bed section below; declared here
// as no-ops so narration works even if the bed is never loaded.
let _attachNarrationToBus = () => {};
let _detachNarrationFromBus = () => {};


// ── Music bed (duckable background audio under the comic) ───────
// Any music-source descriptor with a directly-streamable URL (radio station
// or a file's stream URL) can play UNDER the narration. AudioBus does the
// ducking: narration claims the SPEECH tier, the bed the AMBIENT tier, so
// the bed drops while a line is spoken and swells back between lines/pages.
// Picking WHAT plays is the picker's job (Phase 5, over music-source.js);
// this section is the mechanism.

let bedAudio = null;
let bedBusHandle = null;
let bedDuckBaseline = null;   // set while ducked so we restore the right level
let bedVolume = 0.5;          // user-set baseline (0..1)
let bedDescriptor = null;     // the currently-loaded source descriptor

function _ensureBedAudio() {
  if (bedAudio) return bedAudio;
  bedAudio = new Audio();
  bedAudio.loop = true;       // a bed is continuous; loop short files/streams
  bedAudio.volume = bedVolume;
  // Do NOT set crossOrigin — internet-radio streams don't serve CORS and we
  // don't need to read the samples.
  bedBusHandle = AudioBus.register({
    id: 'cast-comic-bed',
    tier: 'ambient',
    kind: 'music',
    duck: (level) => {
      if (!bedAudio || bedDuckBaseline !== null) return;
      bedDuckBaseline = bedAudio.volume;
      bedAudio.volume = bedDuckBaseline * level;
    },
    unduck: () => {
      if (!bedAudio || bedDuckBaseline === null) return;
      bedAudio.volume = bedDuckBaseline;
      bedDuckBaseline = null;
    },
    stop: () => stopBed(),
  });
  bedAudio.addEventListener('play', () => bedBusHandle?.claim());
  bedAudio.addEventListener('pause', () => bedBusHandle?.release());
  return bedAudio;
}

// The user's saved bed choice (restored on boot, NOT auto-played). When the
// user starts narration, this bed comes up under it — narration is the
// explicit trigger, so we're not auto-starting an unprompted audio flow.
let _preferredBed = null;

/** Start (or switch) the music bed to `descriptor`. Streamable sources only
 *  on this surface today; YouTube beds need the iframe player (deferred).
 *  Persists the choice server-side unless {persist:false} (restore path). */
function startBed(descriptor, { persist = true } = {}) {
  if (!descriptor) return;
  if (!descriptor.url) { _bedUnsupported(descriptor); return; }
  bedDescriptor = descriptor;
  _preferredBed = descriptor;
  const a = _ensureBedAudio();
  a.src = descriptor.url;
  a.play().catch(() => { /* autoplay gate — remote can retry */ });
  if (persist) _saveBedChoice(descriptor);
}

function stopBed({ persist = true } = {}) {
  bedDescriptor = null;
  if (persist) { _preferredBed = null; _saveBedChoice(null); }
  if (!bedAudio) return;
  bedAudio.pause();
  bedAudio.removeAttribute('src');
  bedDuckBaseline = null;
}

// ── Bed choice persistence (server-side, per-user, shared w/ web) ──
function _compactDescriptor(d) {
  return {
    kind: d.kind, id: d.id, name: d.name, genre: d.genre || '',
    desc: d.desc || '', url: d.url || null, videoId: d.videoId || null,
    poster: d.poster || null, source: d.source || '',
  };
}

async function _saveBedChoice(descriptor) {
  try {
    await fetch('/api/config/ui', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        comicCastBed: descriptor ? JSON.stringify(_compactDescriptor(descriptor)) : '',
      }),
    });
  } catch { /* best-effort — the in-memory choice still holds for this session */ }
}

async function _loadBedChoice() {
  try {
    const r = await fetch('/api/config/ui', { credentials: 'same-origin' });
    if (!r.ok) return null;
    const data = await r.json();
    const raw = data?.comicCastBed;
    if (!raw) return null;
    const d = JSON.parse(raw);
    return (d && d.url) ? d : null;   // only auto-resumable (streamable) beds
  } catch { return null; }
}

function setBedVolume(v) {
  bedVolume = Math.max(0, Math.min(1, Number(v) || 0));
  if (bedAudio && bedDuckBaseline === null) bedAudio.volume = bedVolume;
}

function _bedUnsupported(descriptor) {
  const cap = _ensureCaption();
  cap.textContent = `Can't play "${descriptor?.name || 'that'}" as a bed here yet`;
  cap.dataset.kind = 'hint';
  cap.hidden = false;
  setTimeout(() => { if (!narration?.active) cap.hidden = true; }, 4000);
}

// Fill in the narration→bus hooks now that AudioBus is imported. Narration
// claims the SPEECH tier while its <audio> is playing so the bed ducks; the
// narration audio itself is never ducked.
let _narrationBusHandle = null;
let _narrationAudioEl = null;
const _narrationClaim = () => _narrationBusHandle?.claim();
const _narrationRelease = () => _narrationBusHandle?.release();

_attachNarrationToBus = (audioEl) => {
  if (!audioEl) return;
  if (!_narrationBusHandle) {
    _narrationBusHandle = AudioBus.register({
      id: 'cast-comic-narration',
      tier: 'speech',
      kind: 'speech',
      duck: () => {},
      unduck: () => {},
    });
  }
  _narrationAudioEl = audioEl;
  audioEl.addEventListener('play', _narrationClaim);
  audioEl.addEventListener('pause', _narrationRelease);
  if (!audioEl.paused) _narrationClaim();
};

_detachNarrationFromBus = () => {
  _narrationRelease();
  if (_narrationAudioEl) {
    _narrationAudioEl.removeEventListener('play', _narrationClaim);
    _narrationAudioEl.removeEventListener('pause', _narrationRelease);
    _narrationAudioEl = null;
  }
};


// ── Bed picker (favorites + combined search, remote-navigable) ─
// A TV overlay to choose WHAT plays under the comic, over the shared
// music-source layer — the same favorites + combined search Grove uses, so
// there's one music world across surfaces. Text entry on a TV is painful,
// so search is driven from the controller via a {bed_search} patch; the
// local D-pad (arrows/Enter/Back) navigates the list for HDMI-CEC remotes.
// Names render via textContent (no innerHTML) so untrusted station names
// can't inject. Never auto-selects; "Off" is always the first row.

let picker = null;   // { root, listEl, hintEl, items:[{label,sub,descriptor}], sel, open }

function _ensurePicker() {
  if (picker) return picker;
  const root = document.createElement('div');
  root.className = 'cc-bed-picker';
  root.hidden = true;
  const panel = document.createElement('div');
  panel.className = 'cc-bed-panel';
  const title = document.createElement('div');
  title.className = 'cc-bed-title';
  title.textContent = 'Music bed';
  const hintEl = document.createElement('div');
  hintEl.className = 'cc-bed-hint';
  hintEl.textContent = 'Search from your phone · ↑↓ choose · OK play · Back close';
  const listEl = document.createElement('div');
  listEl.className = 'cc-bed-list';
  panel.appendChild(title);
  panel.appendChild(hintEl);
  panel.appendChild(listEl);
  root.appendChild(panel);
  document.body.appendChild(root);
  picker = { root, listEl, hintEl, items: [], sel: 0, open: false };
  return picker;
}

async function openBedPicker() {
  const p = _ensurePicker();
  p.root.hidden = false;
  p.open = true;
  flashHud();
  await _loadPickerItems('');
}

function closeBedPicker() {
  if (!picker) return;
  picker.open = false;
  picker.root.hidden = true;
}

/** Load the picker rows: always an "Off" row first, then either favorites
 *  (no query) or combined search results (with a query). */
async function _loadPickerItems(query) {
  const p = _ensurePicker();
  const items = [{ label: 'Off — no music', sub: '', descriptor: null }];
  try {
    if (query) {
      const stations = await musicSource.searchStations({ q: query, limit: 20, source: 'all' });
      for (const s of stations) {
        if (!s || !s.url) continue;
        items.push({
          label: s.name || 'Station',
          sub: s.desc || s.genre || (s.source || ''),
          descriptor: musicSource.stationToSource(s),
        });
      }
    } else {
      const favs = await musicSource.loadFavorites();
      for (const f of favs) {
        if (!f || !f.url) continue;
        items.push({
          label: f.name || 'Station',
          sub: f.desc || f.genre || 'favorite',
          descriptor: musicSource.stationToSource(f, { source: 'favorites' }),
        });
      }
    }
  } catch { /* keep at least the Off row */ }
  p.items = items;
  // Keep the current bed highlighted if it's still in the list, else top.
  p.sel = Math.max(0, items.findIndex(
    it => it.descriptor && bedDescriptor && it.descriptor.id === bedDescriptor.id));
  _renderPicker();
}

function _renderPicker() {
  const p = picker;
  if (!p) return;
  p.listEl.innerHTML = '';
  p.items.forEach((it, i) => {
    const row = document.createElement('div');
    row.className = 'cc-bed-row' + (i === p.sel ? ' sel' : '')
      + (it.descriptor && bedDescriptor && it.descriptor.id === bedDescriptor.id ? ' active' : '')
      + (!it.descriptor && !bedDescriptor ? ' active' : '');
    const name = document.createElement('span');
    name.className = 'cc-bed-name';
    name.textContent = it.label;                 // textContent — no injection
    row.appendChild(name);
    if (it.sub) {
      const sub = document.createElement('span');
      sub.className = 'cc-bed-sub';
      sub.textContent = it.sub;
      row.appendChild(sub);
    }
    p.listEl.appendChild(row);
  });
}

function _pickerMove(delta) {
  if (!picker || !picker.items.length) return;
  picker.sel = (picker.sel + delta + picker.items.length) % picker.items.length;
  _renderPicker();
  picker.listEl.children[picker.sel]?.scrollIntoView({ block: 'nearest' });
}

function _pickerActivate() {
  if (!picker) return;
  const it = picker.items[picker.sel];
  if (!it) return;
  if (it.descriptor) startBed(it.descriptor); else stopBed();
  _renderPicker();
  closeBedPicker();
}

/** D-pad handling while the picker is open. Returns true if the key was
 *  consumed (so the reader's own nav doesn't also act on it). */
function _handlePickerKey(e) {
  switch (e.key) {
    case 'ArrowDown': e.preventDefault(); _pickerMove(1); return true;
    case 'ArrowUp':   e.preventDefault(); _pickerMove(-1); return true;
    case 'Enter': case ' ': e.preventDefault(); _pickerActivate(); return true;
    case 'Escape': case 'Backspace': case 'b': case 'B':
      e.preventDefault(); closeBedPicker(); return true;
    default: return false;
  }
}


// ── Boot ───────────────────────────────────────────────────────

(async () => {
  if (!FILE_ID) {
    showError('No comic id provided');
    return;
  }
  // Before applyStageClasses — the stage's dir- class is written from
  // state.readingDirection, so seeding after it would paint the wrong one.
  await fetchDefaultDirection();
  applyStageClasses();
  await Promise.all([fetchManifest(), fetchMetadata()]);
  // Sibling-chapter discovery rides on the series_id captured by
  // fetchMetadata. Fire-and-forget — siblings only gate the autoplay
  // end-of-chapter advance, so a missing/late sibling list just means
  // cast falls back to its pre-feature "stop at end" behavior for the
  // first chapter. Subsequent chapters pick it up once the fetch
  // resolves.
  fetchSiblings();
  await showPage(state.currentPage, { immediate: true });
  flashHud(HUD_FADE_MS * 1.6);  // longer first-show to orient the viewer
  // Probe for a voiced narration in the background — never blocks the reader,
  // never auto-plays (surfaces a hint; controller or 'N' starts it).
  initNarration();
  // Restore the user's saved music-bed choice (not played until narration
  // starts — see startNarration).
  _loadBedChoice().then((d) => { if (d) _preferredBed = d; });
})();


// ── Cleanup ────────────────────────────────────────────────────

window.addEventListener('beforeunload', () => {
  for (const url of blobUrls.values()) URL.revokeObjectURL(url);
  blobUrls.clear();
  releaseWakeLock();
  narration?.clock?.destroy();
  stopBed();
});
