/**
 * library/cover.js — per-item cover renderer.
 *
 * Produces a HTML string that fills its parent container (row / card /
 * tile / detail hero — every consumer styles the wrapper, the cover is
 * size-agnostic). Picks the strongest representation available:
 *
 *   1. metadata.cover_url   universal cover (EPUB extract, future PDF
 *                            first-page thumbs, enriched docs)
 *   2. format-specific      apps → /preview-image PNG; images → the
 *                            raw download; games → metadata.thumbnail_url
 *                            with libretro CDN guess for ROMs
 *   3. mini representation  PDFs/notes get a paper card with text
 *                            preview; PPTX gets a slide thumbnail;
 *                            XLSX gets a mini grid
 *   4. tinted icon          falls back to a hashed-hue tint with the
 *                            type icon centered
 *
 * `<img>` covers carry an inline onerror handler that swaps the broken
 * image for the matching tier-4 fallback so we never paint a
 * broken-image glyph on the surface.
 *
 * App-preview backfill: when an app artifact is rendered without a
 * captured preview, the cover carries `data-app-needs-capture` so the
 * main pane's IntersectionObserver can fire one POST /capture-preview
 * per artifact when it scrolls into view.
 */

import { escapeHtml } from '../app.js';
import { libretroThumbCandidates } from '../library-game-sources.js';


// ── Type icons (16×16 stroke, currentColor) ────────────────────────
//
// Used both as the visible glyph in tier-4 fallbacks AND as the badge
// stamp in the corner of mini representations. Kept simple line work
// so they read at every size from a 40px row cover to a 400px detail
// hero.
const ICONS = {
  app:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M10 9.2v5.6l5-2.8z" fill="currentColor" stroke="none"/></svg>',
  game:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 4h6v5h5v6h-5v5H9v-5H4V9h5z"/></svg>',
  doc:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/></svg>',
  pdf:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/><text x="7.5" y="17.5" font-size="5" font-weight="700" stroke="none" fill="currentColor" font-family="system-ui">PDF</text></svg>',
  epub:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 4h8a4 4 0 0 1 4 4v12a3 3 0 0 0-3-3H4z"/><path d="M20 4h-4a4 4 0 0 0-4 4v12a3 3 0 0 1 3-3h5z"/></svg>',
  note:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 4h10l4 4v12a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z"/><path d="M15 4v4h4M8 13h8M8 17h5M8 9h4"/></svg>',
  pptx:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="5" width="14" height="10" rx="1"/><rect x="7" y="9" width="14" height="10" rx="1"/></svg>',
  xlsx:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="1"/><path d="M3 10h18M3 16h18M9 4v16M15 4v16"/></svg>',
  chart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 20h18"/><path d="M7 20v-7M12 20v-12M17 20v-5"/></svg>',
  image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.5"/><path d="M21 16l-5-5-9 9"/></svg>',
};

// Per-format icon selection so PDFs / notes / books look distinct from
// each other in the doc bucket. Falls through to the type-level icon
// when the format isn't enumerated here.
const FORMAT_ICONS = {
  pdf:  ICONS.pdf,
  epub: ICONS.epub,
  md:   ICONS.note,
  txt:  ICONS.note,
  rst:  ICONS.note,
  log:  ICONS.note,
  json: ICONS.note,
  png:  ICONS.image,
  jpg:  ICONS.image,
  jpeg: ICONS.image,
  webp: ICONS.image,
  svg:  ICONS.image,
  gif:  ICONS.image,
};


// ── Public ─────────────────────────────────────────────────────────

/**
 * Return a HTML string representing the item's cover. Always renders
 * something visible — never returns the empty string. Falls back
 * through three tiers (image cover → mini representation → tinted icon)
 * so every artifact type lands somewhere intentional.
 *
 * `size` controls representation complexity. List rows render covers
 * at 40px where paper/slide/sheet mini layouts are unreadable, so the
 * 'row' size collapses those tiers onto the tinted icon variant and
 * keeps only real image covers / procedural game covers.
 */
export function renderCover(item, { size = 'card' } = {}) {
  if (!item) return _tintedIcon({ format: '', _type: 'doc' });

  const meta = item.metadata || {};

  // Publications (coder "Save to Library") carry a captured screenshot at
  // their assets route — the union blanks their metadata so none of the
  // metadata-driven tiers below fire. Use the screenshot with a graceful
  // fall-through to the tinted icon. Skipped at 'row' size (40px) where a
  // screenshot is unreadable and the tinted icon reads better.
  const isPub = item._isPublication
    || (typeof item.id === 'string' && item.id.startsWith('pub_'));
  if (isPub) {
    // Row size (40px): tinted icon reads better than a shrunk screenshot,
    // and it avoids the artifact-endpoint calls _appCover would make with a
    // pub_ id (those 404). Larger sizes get the real screenshot.
    return size === 'row' ? _tintedIcon(item) : _publicationCover(item);
  }

  // Tier 1 — universal cover_url (EPUB cover extract, future PDF
  // first-page renders). Skipped for apps/games which carry their own
  // dedicated preview path.
  if (meta.cover_url && item._type !== 'app' && item._type !== 'game') {
    return _imageCover(meta.cover_url, item);
  }

  // Tier 2 — format-specific dispatch.
  switch (item._type) {
    case 'app':   return _appCover(item);
    case 'game':  return _gameCover(item);
    case 'chart': return _imageArtifactCover(item);
    case 'pptx':  return size === 'row' ? _tintedIcon(item) : _slideMini(item);
    case 'xlsx':  return size === 'row' ? _tintedIcon(item) : _sheetMini(item);
    case 'doc':   return size === 'row' ? _tintedIcon(item) : _docCover(item);
    default:      return _tintedIcon(item);
  }
}


// ── Tier 1 / 2 image covers ────────────────────────────────────────

function _imageCover(url, item) {
  const alt = escapeHtml(item.display_name || item.filename || '');
  const fallback = _tintedIcon(item).replace(/'/g, "\\'");
  return `<img class="lib-cover-img" src="${escapeHtml(url)}" alt="${alt}"
               loading="lazy" decoding="async" referrerpolicy="no-referrer"
               onerror="this.outerHTML='${fallback}'">`;
}

function _publicationCover(item) {
  // The publication's saved screenshot. 404s (saved without one) fall back
  // to the tinted icon via onerror so we never paint a broken glyph.
  const alt = escapeHtml(item.display_name || item.filename || '');
  const fallback = _tintedIcon(item).replace(/'/g, "\\'");
  const src = `/api/library/publications/${encodeURIComponent(item.id)}`
    + `/assets/__screenshot.png`;
  return `<img class="lib-cover-img" src="${src}"
               alt="${alt}" loading="lazy" decoding="async"
               onerror="this.outerHTML='${fallback}'">`;
}

function _imageArtifactCover(item) {
  // chart/image type — the artifact IS the image. Use the download URL.
  const alt = escapeHtml(item.display_name || item.filename || '');
  const fallback = _tintedIcon(item).replace(/'/g, "\\'");
  return `<img class="lib-cover-img" src="/api/artifacts/${encodeURIComponent(item.id)}/download"
               alt="${alt}" loading="lazy" decoding="async"
               onerror="this.outerHTML='${fallback}'">`;
}

function _appCover(item) {
  const alt = escapeHtml(item.display_name || item.filename || '');
  if (item.metadata?.preview_image) {
    const fallback = _tintedIcon(item).replace(/'/g, "\\'");
    return `<img class="lib-cover-img" src="/api/artifacts/${encodeURIComponent(item.id)}/preview-image"
                 alt="${alt}" loading="lazy" decoding="async"
                 data-artifact-id="${escapeHtml(item.id)}"
                 onerror="this.outerHTML='${fallback}'">`;
  }
  // No screenshot yet — mark for backfill. The main pane's
  // IntersectionObserver looks for [data-app-needs-capture] and POSTs
  // /capture-preview when the cover scrolls into view. On success the
  // /preview-image endpoint starts returning a real PNG; the next
  // render picks it up.
  return _tintedIcon(item, {
    extraAttrs: ` data-artifact-id="${escapeHtml(item.id)}" data-app-needs-capture="1"`,
  });
}

// Shared onerror for game covers. Walks the remaining candidate URLs,
// then swaps in the procedural cover. One global per page, because the
// procedural HTML is far too big (and too quote-heavy) to live inline.
if (typeof window !== 'undefined' && !window.__augLibCoverFallback) {
  window.__augLibCoverFallback = function(img) {
    try {
      const remaining = JSON.parse(img.getAttribute('data-cover-fallbacks') || '[]');
      if (Array.isArray(remaining) && remaining.length > 0) {
        const next = remaining.shift();
        img.setAttribute('data-cover-fallbacks', JSON.stringify(remaining));
        img.src = next;
        return;
      }
    } catch { /* fall through to procedural */ }
    const procedural = img.getAttribute('data-procedural-fallback');
    if (procedural) {
      try { img.outerHTML = procedural; return; } catch { /* fall through */ }
    }
    img.remove();
  };
}

function _gameCover(item) {
  const meta = item.metadata || {};
  // A ROM has at most one explicit cover but SEVERAL libretro guesses
  // (filename stem, then title), so this is a chain, not one URL.
  const candidates = [
    meta.thumbnail_url,
    ...(meta.kind === 'emulator_rom' ? libretroThumbCandidates(item) : []),
  ].filter(Boolean);
  if (candidates.length) {
    const alt = escapeHtml(item.display_name || item.filename || '');
    // The old handler was `onerror="this.outerHTML='<procedural html>'"`.
    // The procedural markup contains double quotes, which CLOSED the
    // onerror attribute — so the handler was malformed and never ran,
    // and every libretro miss showed the browser's broken-image icon
    // with the alt text next to it. Hand both the chain and the
    // procedural HTML over as properly-escaped data attributes instead.
    return `<img class="lib-cover-img" src="${escapeHtml(candidates[0])}" alt="${alt}"
                 loading="lazy" decoding="async" referrerpolicy="no-referrer"
                 data-cover-fallbacks="${escapeHtml(JSON.stringify(candidates.slice(1)))}"
                 data-procedural-fallback="${escapeHtml(_proceduralGameCover(item))}"
                 onerror="window.__augLibCoverFallback(this)">`;
  }
  return _proceduralGameCover(item);
}


// ── Tier 2 mini representations ────────────────────────────────────

function _slideMini(item) {
  const meta = item.metadata || {};
  const title = String(meta.title || item.display_name || 'Presentation');
  const subtitle = String(meta.author || meta.organization || '');
  const slideCount = meta.slide_count || meta.slides || '';
  return `
    <div class="lib-cover-mini lib-cover-slide">
      <div class="lib-cover-slide-frame">
        <div class="lib-cover-slide-title">${escapeHtml(title.slice(0, 36))}</div>
        ${subtitle
          ? `<div class="lib-cover-slide-subtitle">${escapeHtml(subtitle.slice(0, 32))}</div>`
          : ''}
        <div class="lib-cover-slide-bullets">
          <span></span><span></span><span></span>
        </div>
      </div>
      <div class="lib-cover-mini-badge">
        ${slideCount ? `<span>${escapeHtml(String(slideCount))} slides</span>` : ''}
        <span class="lib-cover-mini-fmt">PPTX</span>
      </div>
    </div>`;
}

function _sheetMini(item) {
  const meta = item.metadata || {};
  const sheets = meta.sheet_count || meta.sheets || 1;
  const rows = meta.total_rows || meta.row_count || '';
  const fmt = (item.format || 'XLSX').toUpperCase();
  // 6×4 grid — visually-busy enough to read as a spreadsheet at small
  // sizes; deliberately sparse on cell content so the badge stays
  // legible under it.
  const cells = Array(24).fill(0).map((_, i) => {
    const filled = (i * 73) % 5 < 2;  // ~40% cells "filled"
    return `<span class="${filled ? 'on' : ''}"></span>`;
  }).join('');
  return `
    <div class="lib-cover-mini lib-cover-sheet">
      <div class="lib-cover-sheet-grid">${cells}</div>
      <div class="lib-cover-mini-badge">
        ${rows ? `<span>${escapeHtml(String(rows))} rows</span>` : ''}
        <span class="lib-cover-mini-fmt">${escapeHtml(fmt)}</span>
      </div>
    </div>`;
}

function _docCover(item) {
  // PDFs without a cover_url, plus EPUBs without one, plus all the
  // text-flavoured types (md/txt/rst/log/json/docx). Paper-card mini
  // with a snippet of content/description and the format stamp in the
  // corner.
  const meta = item.metadata || {};
  const desc = String(
    meta.description || meta.summary || meta.subject ||
    item.display_name || item.filename || ''
  );
  const fmt = (item.format || '').toUpperCase();
  const lines = desc.trim().slice(0, 160);
  return `
    <div class="lib-cover-mini lib-cover-paper" data-fmt="${escapeHtml(item.format || '')}">
      <div class="lib-cover-paper-page">
        <div class="lib-cover-paper-rule"></div>
        <div class="lib-cover-paper-rule"></div>
        <div class="lib-cover-paper-rule"></div>
        <div class="lib-cover-paper-rule"></div>
        ${lines
          ? `<div class="lib-cover-paper-text">${escapeHtml(lines)}</div>`
          : ''}
      </div>
      ${fmt ? `<div class="lib-cover-mini-badge"><span class="lib-cover-mini-fmt">${escapeHtml(fmt)}</span></div>` : ''}
    </div>`;
}


// ── Tier 4 tinted icon (uniform fallback) ──────────────────────────

function _tintedIcon(item, { extraAttrs = '' } = {}) {
  const fmt = (item?.format || item?._type || 'doc').toLowerCase();
  const icon = FORMAT_ICONS[fmt] || ICONS[item?._type] || ICONS.doc;
  const hue = _hashStr(fmt) % 360;
  return `<div class="lib-cover-tinted"
               style="--cover-hue:${hue}"${extraAttrs}>
    <span class="lib-cover-icon">${icon}</span>
  </div>`;
}


// ── Game procedural cover (gradient + initials + tag) ──────────────

function _proceduralGameCover(item) {
  const title = String(item.display_name || item.filename || 'Untitled');
  const meta = item.metadata || {};
  const sys = String(meta.system_label || meta.system_id || meta.system || '');
  const kind = String(meta.kind || '');
  const source = String(meta.source || '');
  const h = _hashStr(title + '|' + sys);
  const hue1 = h % 360;
  const hue2 = (hue1 + 38) % 360;
  const initials = title
    .replace(/^(The|A|An)\s+/i, '')
    .split(/\s+/).filter(Boolean).slice(0, 2)
    .map(w => w[0]).join('').toUpperCase().slice(0, 2) || '?';
  let tag = 'GAME';
  if (kind === 'emulator_rom' || source === 'emulator')      tag = 'RETRO';
  else if (kind === 'streamed_game' || source === 'streamed') tag = 'STREAM';
  else if (kind === 'js13k_game' || source === 'js13k')      tag = 'JS13K';
  else if (kind === 'web_app' || source === 'marketplace')   tag = 'CURATED';
  return `
    <svg class="lib-cover-procedural" viewBox="0 0 300 400"
         preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id="lpg-${h}" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="hsl(${hue1}, 55%, 32%)"/>
          <stop offset="100%" stop-color="hsl(${hue2}, 60%, 18%)"/>
        </linearGradient>
        <radialGradient id="lpgs-${h}" cx="0.3" cy="0.2" r="0.85">
          <stop offset="0%" stop-color="rgba(255,255,255,0.18)"/>
          <stop offset="100%" stop-color="rgba(255,255,255,0)"/>
        </radialGradient>
      </defs>
      <rect width="300" height="400" fill="url(#lpg-${h})"/>
      <rect width="300" height="400" fill="url(#lpgs-${h})"/>
      <text x="150" y="180" text-anchor="middle"
            font-family="system-ui, sans-serif" font-size="92"
            font-weight="700" fill="rgba(255,255,255,0.92)"
            letter-spacing="-2">${escapeHtml(initials)}</text>
      <text x="150" y="222" text-anchor="middle"
            font-family="system-ui, sans-serif" font-size="14"
            fill="rgba(255,255,255,0.55)" letter-spacing="2">${escapeHtml(tag)}</text>
    </svg>`;
}


// ── Helpers ────────────────────────────────────────────────────────

function _hashStr(s) {
  // Cheap deterministic hash — used only for procedural-cover hue
  // selection and the SVG gradient id. Doesn't need to be cryptographic.
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = (h * 33 + s.charCodeAt(i)) >>> 0;
  }
  return h;
}


// ── App preview backfill (POST /capture-preview on visibility) ─────

let _captureSentRecently = new Set();
let _captureTimer = null;

/**
 * Trigger a one-shot POST /capture-preview for any [data-app-needs-
 * capture] cover currently in `root`. Called by the main pane after
 * render. Per-artifact dedup so re-renders don't pile on duplicate
 * requests, and a 90-second cooldown so a refresh cycle doesn't
 * thrash chromium on a host that doesn't have it installed (the
 * endpoint will 200 quickly in that case, but it's cheaper to skip).
 */
export function backfillAppPreviews(root) {
  if (!root) return;
  const targets = root.querySelectorAll('[data-app-needs-capture]');
  if (!targets.length) return;
  if (!_captureTimer) {
    _captureTimer = setInterval(() => {
      _captureSentRecently.clear();
    }, 90_000);
  }
  // Lazy IntersectionObserver — created on first call so we don't
  // install one when the user isn't on a page with app covers.
  const io = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const id = entry.target.dataset.artifactId;
      if (!id) continue;
      io.unobserve(entry.target);
      if (_captureSentRecently.has(id)) continue;
      _captureSentRecently.add(id);
      fetch(`/api/artifacts/${encodeURIComponent(id)}/capture-preview`, {
        method: 'POST', credentials: 'same-origin',
      }).catch(() => { /* host without chromium — fine */ });
    }
  }, { root: null, rootMargin: '200px', threshold: 0.01 });
  for (const el of targets) io.observe(el);
}
