/**
 * Comics chip — series-first rendering with a dedicated series-detail
 * page on click-through.
 *
 * Two view modes owned by this module:
 *   - 'series'   → grid of cover cards (top-level Comics view)
 *   - 'chapters' → series detail page: cover hero + masthead + chapter list
 *
 * Design direction: editorial "printed manga volume" — Fraunces serif
 * display, vermillion accent, cream-on-charcoal, typographic hierarchy
 * inspired by Criterion Collection film pages. See
 * ``ui/comic-detail-preview.html`` for the aesthetic source of truth;
 * this module produces the same design wired to real data from
 * ``/api/files/comics/series`` + ``/series/{id}/chapters``.
 *
 * ``loadFiles`` in files/index.js delegates to ``renderComicsGrid`` when
 * the Comics chip is active; that function swaps the panel body between
 * the two modes. State is module-local and survives in-session nav.
 */

import { state, COMICS_SORTS } from './state.js';
import { formatCount } from './helpers.js';
import { escapeHtml, showToast, showConfirm } from '../app.js';
import { mediaCoverUrl } from './api.js';
import {
  renderComicDetail,
  renderChapterRows,
  trimChapterNumber,
} from '../consumption/comic-detail-view.js';


// --- Normalizers: Files raw data → shared comic-detail-view shape ----------
// The shared renderer (consumption/comic-detail-view.js) is the single
// source of truth for the csd-* design, used by both Files and the Media
// drill-in. Files' raw chapters carry their reading state under
// source_metadata.extra; these adapters flatten that into the normalized
// shape the renderer expects. Keep label/order/progress logic identical to
// what the old in-module renderers produced.

function _normSeries(s) {
  return {
    name: s.name,
    author: s.author,
    publisher: s.publisher,
    yearStarted: s.year_started,
    yearEnded: s.year_ended,
    status: s.status,
    genres: Array.isArray(s.genres) ? s.genres : [],
    description: s.description,
    coverUrl: s.cover_file_id ? mediaCoverUrl(s.cover_file_id, { size: 'full' }) : '',
  };
}

function _normChapter(c) {
  const meta = c.source_metadata || {};
  const extra = meta.extra || {};
  const cn = extra.chapter_number;
  const so = extra.chapter_source_order;
  const label = (cn !== null && cn !== undefined && cn !== '')
    ? `Ch. ${trimChapterNumber(cn)}`
    : (so != null ? `Ch. ${so}` : '');
  return {
    id: c.id,
    name: c.name,
    label,
    order: so ?? 0,
    currentS: Number(meta.current_time_s) || 0,
    totalS: Number(meta.duration_s) || Number(extra.page_count) || 0,
    isFinished: !!meta.is_finished,
    updatedAt: c.updated_at,
  };
}


// --- View state -----------------------------------------------------------
// Drill-down location (series grid vs a specific series' chapter list)
// persists to localStorage so the next Files open drops the user right
// back where they were — "I was reading Berserk, server restarted, let
// me click Files and pick back up". Only mode + activeSeriesId persist;
// caches, search query, and per-chapter-list filters reset because they
// lose meaning across a reload.

const _DRILL_STORAGE_KEY = 'augmentum.files.comicDrill';

function _loadDrill() {
  try {
    const raw = localStorage.getItem(_DRILL_STORAGE_KEY);
    if (!raw) return { mode: 'series', activeSeriesId: '' };
    const parsed = JSON.parse(raw);
    const mode = parsed?.mode === 'chapters' ? 'chapters' : 'series';
    const activeSeriesId = typeof parsed?.activeSeriesId === 'string'
      ? parsed.activeSeriesId : '';
    // A stored mode of 'chapters' without an id is nonsensical — fall
    // back to the grid so we don't render a broken detail page.
    if (mode === 'chapters' && !activeSeriesId) {
      return { mode: 'series', activeSeriesId: '' };
    }
    return { mode, activeSeriesId };
  } catch {
    return { mode: 'series', activeSeriesId: '' };
  }
}

function _saveDrill() {
  try {
    localStorage.setItem(_DRILL_STORAGE_KEY, JSON.stringify({
      mode: _view.mode,
      activeSeriesId: _view.activeSeriesId,
    }));
  } catch { /* ignore */ }
}

const _initialDrill = _loadDrill();

const _view = {
  mode: _initialDrill.mode,
  activeSeriesId: _initialDrill.activeSeriesId,
  activeSeriesMeta: null,
  seriesCache: [],
  chapterCache: [],
  hideRead: false,
  sortNewestFirst: true,
  // Chapter search query — client-side filter that matches chapter
  // number OR chapter name (case-insensitive substring). Reset when
  // the user drills out of a series; per-series state would be overkill.
  chapterQuery: '',
};


// --- Data fetching --------------------------------------------------------

async function _fetchSeries({
  q = '', sort = '',
  status = '', completion = '', genre = '',
} = {}) {
  const params = new URLSearchParams();
  if (q) params.set('q', q);
  // Backend sort whitelist: newest / oldest / name / updated / unread.
  // Anything else falls through to 'name' server-side, so a stray value
  // from a stale storage key can't break the grid.
  if (sort) params.set('sort', sort);
  // `status` is 'all' | 'reading' | 'caught-up' | 'unread'. Backend
  // treats 'all' (or missing) as no filter; frontend passes it through
  // literally so the response shape stays predictable.
  if (status && status !== 'all') params.set('status', status);
  if (completion) params.set('completion', completion);
  if (genre) params.set('genre', genre);
  params.set('limit', '500');
  try {
    const resp = await fetch(`/api/files/comics/series?${params}`);
    if (!resp.ok) return [];
    const data = await resp.json();
    return Array.isArray(data?.series) ? data.series : [];
  } catch (err) {
    console.warn('[comics] series fetch failed:', err);
    return [];
  }
}

async function _fetchChapters(seriesId) {
  try {
    const resp = await fetch(
      `/api/files/comics/series/${encodeURIComponent(seriesId)}/chapters?limit=2000`,
    );
    if (!resp.ok) return [];
    const data = await resp.json();
    return Array.isArray(data?.files) ? data.files : [];
  } catch (err) {
    console.warn('[comics] chapters fetch failed:', err);
    return [];
  }
}


// --- Utility: relative time formatting ------------------------------------
// Intl.RelativeTimeFormat handles the locale-aware formatting for
// "2 days ago" / "5 minutes ago" without a date library. Browsers since
// 2019 ship it; no feature-detect needed.

function _relativeTime(isoOrSql) {
  if (!isoOrSql) return '';
  try {
    // SQLite datetime('now') emits "YYYY-MM-DD HH:MM:SS" (UTC). JavaScript's
    // Date() needs a T separator and a timezone indicator to parse reliably
    // across engines — otherwise it gets interpreted as local time. Insert
    // both so the elapsed seconds math is accurate.
    const normalized = /Z$|[+-]\d{2}:?\d{2}$/.test(isoOrSql)
      ? isoOrSql
      : isoOrSql.replace(' ', 'T') + 'Z';
    const dt = new Date(normalized);
    if (Number.isNaN(dt.getTime())) return '';
    const diffSec = Math.round((Date.now() - dt.getTime()) / 1000);
    const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
    if (diffSec < 45)     return rtf.format(-diffSec, 'second');
    const minutes = Math.round(diffSec / 60);
    if (minutes < 45)     return rtf.format(-minutes, 'minute');
    const hours = Math.round(minutes / 60);
    if (hours < 22)       return rtf.format(-hours, 'hour');
    const days = Math.round(hours / 24);
    if (days < 26)        return rtf.format(-days, 'day');
    const months = Math.round(days / 30);
    if (months < 11)      return rtf.format(-months, 'month');
    return rtf.format(-Math.round(months / 12), 'year');
  } catch {
    return '';
  }
}


// --- Utility: reading-state analysis --------------------------------------
// Compute the "continue reading" target, last-read timestamp, and totals
// from the raw chapter list. Kept pure + synchronous — called on every
// render of the detail page. Cheap even for 2000-chapter series.

function _analyzeChapters(chapters) {
  let inProgressChapter = null;
  let inProgressUpdatedAt = 0;
  let firstUnread = null;
  let lastReadAt = '';
  let lastReadAtTs = 0;
  let finishedCount = 0;
  let inProgressCount = 0;
  const sorted = chapters.slice().sort(_bySourceOrderAsc);

  for (const ch of sorted) {
    const meta = ch.source_metadata || {};
    const current = Number(meta.current_time_s) || 0;
    const isFinished = !!meta.is_finished;
    const updatedAtTs = Date.parse(_normalizeTs(ch.updated_at)) || 0;

    if (isFinished) {
      finishedCount++;
      if (updatedAtTs > lastReadAtTs) {
        lastReadAtTs = updatedAtTs;
        lastReadAt = ch.updated_at;
      }
      continue;
    }
    if (current > 0) {
      inProgressCount++;
      // Most-recently-touched in-progress chapter is the "resume here" target
      if (updatedAtTs > inProgressUpdatedAt) {
        inProgressUpdatedAt = updatedAtTs;
        inProgressChapter = ch;
      }
      if (updatedAtTs > lastReadAtTs) {
        lastReadAtTs = updatedAtTs;
        lastReadAt = ch.updated_at;
      }
      continue;
    }
    if (firstUnread === null) firstUnread = ch;
  }

  // Continue-reading priority: in-progress wins, else first unread. If
  // everything is finished, user is caught up — return null so the CTA
  // renders as a re-read-from-start affordance.
  const continueChapter = inProgressChapter || firstUnread || null;
  return {
    continueChapter,
    lastReadAt,
    finishedCount,
    inProgressCount,
    unreadCount: Math.max(0, chapters.length - finishedCount - inProgressCount),
    firstChapter: sorted[0] || null,
  };
}

function _bySourceOrderAsc(a, b) {
  const ao = ((a.source_metadata || {}).extra || {}).chapter_source_order ?? 0;
  const bo = ((b.source_metadata || {}).extra || {}).chapter_source_order ?? 0;
  return Number(ao) - Number(bo);
}

function _normalizeTs(s) {
  if (!s) return '';
  return /Z$|[+-]\d{2}:?\d{2}$/.test(s) ? s : s.replace(' ', 'T') + 'Z';
}


// --- Rendering: series grid (top-level) -----------------------------------

function _renderSeriesCard(s) {
  const coverSrc = s.cover_file_id ? mediaCoverUrl(s.cover_file_id) : '';
  const chapterLabel = s.chapter_count === 1 ? 'chapter' : 'chapters';
  const progress = _seriesProgress(s);
  return `
    <button class="files-card files-card-comic-series"
            data-series-id="${escapeHtml(s.id)}"
            data-series-name="${escapeHtml(s.name)}"
            title="${escapeHtml(s.name)}">
      <div class="files-card-thumb files-card-thumb-comic">
        ${coverSrc
          ? `<img src="${escapeHtml(coverSrc)}" alt="" loading="lazy" decoding="async" onerror="this.style.display='none'">`
          : ''}
        ${progress
          ? `<div class="files-card-progress-bar"><div class="files-card-progress-fill" style="width:${progress.pct}%"></div></div>`
          : ''}
      </div>
      <div class="files-card-body">
        <div class="files-card-title">${escapeHtml(s.name)}</div>
        <div class="files-card-meta">
          ${s.author ? `<span class="files-card-author">${escapeHtml(s.author)}</span>` : ''}
          <span class="files-card-chapter-count">${formatCount(s.chapter_count)} ${chapterLabel}</span>
          ${s.unread_count > 0
            ? `<span class="files-card-unread-badge">${formatCount(s.unread_count)} new</span>`
            : ''}
        </div>
      </div>
    </button>
  `;
}

function _seriesProgress(s) {
  const total = s.chapter_count || 0;
  if (total <= 0) return null;
  const pct = Math.round((s.finished_count / total) * 100);
  if (pct <= 0 && s.in_progress_count === 0) return null;
  return { pct };
}


// --- Rendering: series detail page ---------------------------------------

function _renderSeriesDetail(series, chapters) {
  const analysis = _analyzeChapters(chapters);
  return `
    <div class="csd">
      ${_renderBreadcrumb()}
      ${_renderHero(series, analysis)}
      ${_renderChapterSection(series, chapters, analysis)}
    </div>
  `;
}

function _renderBreadcrumb() {
  return `
    <button type="button" class="csd-back" data-action="back-to-series">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>
      </svg>
      Comics
    </button>
  `;
}

function _renderHero(series, analysis) {
  const coverSrc = series.cover_file_id ? mediaCoverUrl(series.cover_file_id, { size: 'full' }) : '';
  const years = _formatYears(series.year_started, series.year_ended, series.status);
  const genreLine = Array.isArray(series.genres) && series.genres.length
    ? series.genres.slice(0, 6).join(' · ')
    : '';
  const statusLabel = (series.status || '').replace(/_/g, ' ').trim() || null;

  // CTA resolution — three states, picked in priority order:
  //   1. Has in-progress chapter → "Continue reading · Ch. N"
  //   2. Has unread chapter → "Start reading · Ch. 1"
  //   3. Caught up → "Re-read from Ch. 1"
  let ctaLabel = 'Start reading';
  let ctaChapterTarget = analysis.firstChapter;
  let ctaSubLabel = '';
  if (analysis.continueChapter && analysis.inProgressCount > 0) {
    ctaLabel = 'Continue reading';
    ctaChapterTarget = analysis.continueChapter;
    const meta = analysis.continueChapter.source_metadata || {};
    const current = Math.round(Number(meta.current_time_s) || 0);
    const total = Math.round(Number(meta.duration_s) || 0);
    if (total) ctaSubLabel = `Last read ${_relativeTime(analysis.lastReadAt)} · Page ${current} / ${total}`;
    else       ctaSubLabel = `Last read ${_relativeTime(analysis.lastReadAt)}`;
  } else if (analysis.continueChapter) {
    ctaLabel = 'Start reading';
    ctaChapterTarget = analysis.continueChapter;
  } else if (analysis.firstChapter) {
    ctaLabel = 'Re-read from Ch. 1';
    ctaChapterTarget = analysis.firstChapter;
    ctaSubLabel = 'All caught up';
  }
  const ctaChapterLabel = _chapterShortLabel(ctaChapterTarget);
  const secondaryVisible = ctaChapterTarget && ctaChapterTarget !== analysis.firstChapter;

  return `
    <section class="csd-hero">
      <figure class="csd-cover">
        <div class="csd-cover-spine" aria-hidden="true"></div>
        <div class="csd-cover-art">
          ${coverSrc
            ? `<img src="${escapeHtml(coverSrc)}" alt=""
                    onerror="this.closest('.csd-cover-art').classList.add('csd-cover-art-fallback')">`
            : ''}
          <div class="csd-cover-label" aria-hidden="true">
            <div class="csd-cover-label-top">
              ${series.publisher ? escapeHtml(series.publisher) : 'Volume'}
            </div>
            <div class="csd-cover-label-main">${escapeHtml(_shortForCover(series.name))}</div>
            <div class="csd-cover-label-author">
              ${escapeHtml(series.author || 'Unknown')}
            </div>
          </div>
        </div>
      </figure>

      <div class="csd-meta">
        <h1 class="csd-title">${escapeHtml(series.name)}</h1>
        ${series.author
          ? `<p class="csd-author">by <em>${escapeHtml(series.author)}</em></p>`
          : `<p class="csd-author">Author unknown</p>`
        }

        <div class="csd-divider"></div>

        <div class="csd-meta-row">
          ${statusLabel
            ? `<span class="csd-status">
                <span class="csd-status-dot"></span>${escapeHtml(_titleCase(statusLabel))}
              </span>`
            : ''
          }
          ${statusLabel && years ? `<span class="csd-sep">·</span>` : ''}
          ${years ? `<span>${escapeHtml(years)}</span>` : ''}
          ${(statusLabel || years) && genreLine ? `<span class="csd-sep">·</span>` : ''}
          ${genreLine ? `<span>${escapeHtml(genreLine)}</span>` : ''}
        </div>

        ${series.description
          ? `<p class="csd-description">${escapeHtml(series.description)}</p>`
          : ''}

        <div class="csd-cta-row">
          <button class="csd-btn csd-btn-primary" type="button"
                  data-action="open-cta-chapter"
                  ${ctaChapterTarget ? `data-chapter-file-id="${escapeHtml(ctaChapterTarget.id)}"` : ''}>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <polygon points="6 4 20 12 6 20"/>
            </svg>
            <span>${escapeHtml(ctaLabel)}</span>
            ${ctaChapterLabel
              ? `<span class="csd-btn-primary-chnum">${escapeHtml(ctaChapterLabel)}</span>`
              : ''}
          </button>
          ${secondaryVisible
            ? `<button class="csd-btn csd-btn-secondary" type="button"
                       data-action="open-first-chapter">
                Start from Ch. 1
              </button>`
            : ''}
        </div>
        ${ctaSubLabel
          ? `<div class="csd-last-read">${escapeHtml(ctaSubLabel)}</div>`
          : ''}
      </div>
    </section>
  `;
}

function _renderChapterSection(series, chapters, analysis) {
  return `
    <section class="csd-chapters">
      <header class="csd-chapters-header">
        <h2>Chapters</h2>
        <div class="csd-chapters-count">
          ${formatCount(chapters.length)} ${chapters.length === 1 ? 'chapter' : 'chapters'}
          ${analysis.unreadCount > 0
            ? `· <span class="csd-unread-count">${formatCount(analysis.unreadCount)} unread</span>`
            : ''}
          ${analysis.inProgressCount > 0
            ? `· ${formatCount(analysis.inProgressCount)} in progress`
            : ''}
        </div>
        <div class="csd-chapters-controls">
          <label class="csd-chapter-search" aria-label="Search chapters">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="11" cy="11" r="8"/>
              <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <input type="search" placeholder="Find chapter…"
                   data-action="chapter-search"
                   value="${escapeHtml(_view.chapterQuery)}"
                   spellcheck="false" autocomplete="off">
          </label>
          <button class="csd-toggle-btn${_view.hideRead ? ' on' : ''}"
                  type="button" data-action="toggle-hide-read"
                  aria-pressed="${_view.hideRead ? 'true' : 'false'}">
            <span class="csd-toggle-check">
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
            </span>
            Hide read
          </button>
          <button class="csd-sort${_view.sortNewestFirst ? '' : ' reversed'}"
                  type="button" data-action="toggle-sort"
                  aria-pressed="${_view.sortNewestFirst ? 'false' : 'true'}">
            <span class="csd-sort-arrow">↓</span>${_view.sortNewestFirst ? 'Newest' : 'Oldest'}
          </button>
          <button class="csd-series-more" type="button"
                  data-action="series-more"
                  aria-label="Series actions"
                  aria-haspopup="menu"
                  title="Series actions">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"
                 aria-hidden="true">
              <circle cx="12" cy="5"  r="1.7"/>
              <circle cx="12" cy="12" r="1.7"/>
              <circle cx="12" cy="19" r="1.7"/>
            </svg>
          </button>
        </div>
      </header>
      <ul class="csd-chapter-list" data-chapter-list>
        ${_renderChapterRows(chapters)}
      </ul>
    </section>
  `;
}

function _renderChapterRows(chapters) {
  let ordered = chapters.slice().sort(_bySourceOrderAsc);
  if (_view.sortNewestFirst) ordered = ordered.reverse();
  let visible = _view.hideRead
    ? ordered.filter(c => !((c.source_metadata || {}).is_finished))
    : ordered;
  // Chapter search — case-insensitive substring match against name +
  // chapter number + source_order. Cheap client-side filter; for a
  // 2000-chapter series this is well under a frame.
  const q = (_view.chapterQuery || '').trim().toLowerCase();
  if (q) {
    visible = visible.filter(c => {
      const name = (c.name || '').toLowerCase();
      if (name.includes(q)) return true;
      const extra = (c.source_metadata || {}).extra || {};
      const cn = extra.chapter_number;
      const so = extra.chapter_source_order;
      if (cn != null && String(cn).toLowerCase().includes(q)) return true;
      if (so != null && String(so).toLowerCase().includes(q)) return true;
      return false;
    });
  }
  if (!visible.length) {
    return `<li class="csd-chapter-empty">No chapters match${q ? ` "${escapeHtml(q)}"` : ' the current filter'}.</li>`;
  }
  return visible.map(_renderChapterRow).join('');
}

function _renderChapterRow(chapter) {
  const meta = chapter.source_metadata || {};
  const extra = meta.extra || {};
  const chapterNumber = extra.chapter_number;
  const sourceOrder = extra.chapter_source_order;
  const label = (chapterNumber !== null && chapterNumber !== undefined && chapterNumber !== '')
    ? `Ch. ${_trimNumber(chapterNumber)}`
    : (sourceOrder != null ? `Ch. ${sourceOrder}` : '');

  const current = Number(meta.current_time_s) || 0;
  const total = Number(meta.duration_s) || Number(extra.page_count) || 0;
  const finished = !!meta.is_finished;
  const inProgress = !finished && current > 0;
  const stateClass = finished ? 'read' : (inProgress ? 'in-progress' : 'unread');

  let metaRight = '';
  if (finished)            metaRight = 'Read';
  else if (inProgress && total) metaRight = `${Math.round(current)} / ${Math.round(total)}`;
  else if (chapter.updated_at)  metaRight = _relativeTime(chapter.updated_at);

  const progressPct = inProgress && total
    ? Math.min(100, Math.round((current / total) * 100))
    : 0;

  // Two trailing actions, both wrapped with the meta in one flex cell so
  // the grid template stays at the original column count. Buttons are
  // always visible (per direct user feedback that auto-hiding chrome
  // makes intent harder); they're styled subtly so they don't fight the
  // text. data-action values route through the panel-level click
  // listener via _handleChapterAction in the comics module.
  const toggleLabel = finished ? 'Mark as unread' : 'Mark as read';
  return `
    <li class="csd-chapter state-${stateClass}"
        data-chapter-file-id="${escapeHtml(chapter.id)}"
        title="${escapeHtml(chapter.name)}"
        role="button" tabindex="0">
      <div class="csd-chapter-state"></div>
      <div class="csd-chapter-num">${escapeHtml(label)}</div>
      <div class="csd-chapter-title">${escapeHtml(chapter.name)}</div>
      <div class="csd-chapter-progress-bar">
        ${inProgress
          ? `<div class="csd-chapter-progress-fill" style="width:${progressPct}%"></div>`
          : ''}
      </div>
      <div class="csd-chapter-actions">
        <span class="csd-chapter-meta">${escapeHtml(metaRight)}</span>
        <button type="button"
                class="csd-chapter-toggle${finished ? ' is-read' : ''}"
                data-action="toggle-read"
                aria-label="${escapeHtml(toggleLabel)}"
                aria-pressed="${finished ? 'true' : 'false'}"
                title="${escapeHtml(toggleLabel)}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.5"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
        </button>
        <button type="button"
                class="csd-chapter-more"
                data-action="chapter-more"
                aria-label="More actions"
                aria-haspopup="menu"
                title="More actions">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"
               aria-hidden="true">
            <circle cx="5" cy="12" r="1.7"/>
            <circle cx="12" cy="12" r="1.7"/>
            <circle cx="19" cy="12" r="1.7"/>
          </svg>
        </button>
      </div>
    </li>
  `;
}


// --- Small helpers --------------------------------------------------------

function _chapterShortLabel(chapter) {
  if (!chapter) return '';
  const extra = ((chapter.source_metadata || {}).extra) || {};
  const n = extra.chapter_number;
  if (n != null && n !== '') return `Ch. ${_trimNumber(n)}`;
  const so = extra.chapter_source_order;
  if (so != null) return `Ch. ${so}`;
  return '';
}

function _trimNumber(n) {
  // Drop trailing .0 on whole-number chapter numbers but keep .5 etc.
  const num = Number(n);
  if (Number.isNaN(num)) return String(n);
  return Number.isInteger(num) ? String(num) : num.toFixed(1).replace(/\.0$/, '');
}

function _formatYears(start, end, status) {
  if (!start && !end) return '';
  if (start && !end) {
    // Ongoing series render as "1989—" (em-dash); completed with year
    // render as "1989–1994" (en-dash, shorter).
    return (status || '').toLowerCase() === 'ongoing' ? `${start}—` : `${start}`;
  }
  if (!start && end) return `${end}`;
  if (start === end) return `${start}`;
  return `${start}–${end}`;
}

function _titleCase(s) {
  return String(s || '').replace(/\b\w/g, c => c.toUpperCase());
}

function _shortForCover(name) {
  // The fallback cover label shows a short version of the title when the
  // real cover image fails to load. Cut aggressively so it fits a 280px
  // card without wrapping.
  if (!name) return 'Untitled';
  const trimmed = name.trim();
  return trimmed.length > 14 ? trimmed.slice(0, 14).trim() + '…' : trimmed;
}


// --- Public API -----------------------------------------------------------

export function isComicsChipActive() {
  return state.currentScope === 'cloud' && state.currentSource === 'comics';
}

export function resetComicsView() {
  _view.mode = 'series';
  _view.activeSeriesId = '';
  _view.activeSeriesMeta = null;
  _view.chapterCache = [];
  _view.hideRead = false;
  _view.sortNewestFirst = true;
  _view.chapterQuery = '';
  _saveDrill();
}

/** Entry point called by loadFiles when the Comics chip is active. */
export async function renderComicsGrid() {
  const grid = state.el.grid;
  if (!grid) return;

  // Reset any class that the detail page layout added, so the grid can
  // render its standard layout when we switch back to series mode.
  grid.className = 'files-grid';

  // Loading shim while we fetch.
  grid.innerHTML = `<div class="files-comics-loading">Loading…</div>`;
  _clearDetailContainer();

  if (_view.mode === 'chapters' && _view.activeSeriesId) {
    // Cold-start restore: persisted drill-down but no in-memory series
    // cache yet. Populate the cache from the list endpoint so we can
    // find the active series' metadata before fetching chapters.
    let series = _view.activeSeriesMeta
      || (_view.seriesCache || []).find(s => s.id === _view.activeSeriesId);
    if (!series) {
      const allSeries = await _fetchSeries({});
      _view.seriesCache = allSeries;
      series = allSeries.find(s => s.id === _view.activeSeriesId) || null;
    }
    if (!series) {
      // The stored series no longer exists (deleted upstream, user
      // switched accounts, etc.) — drop back to the grid and clear the
      // stale drill-down so the next reload starts clean.
      _view.mode = 'series';
      _view.activeSeriesId = '';
      _saveDrill();
      return renderComicsGrid();
    }
    _view.activeSeriesMeta = series;
    const chapters = await _fetchChapters(_view.activeSeriesId);
    _view.chapterCache = chapters;
    // Render into a dedicated container so the detail layout doesn't have
    // to fight the file-grid's CSS grid rules. The container lives inside
    // state.el.grid to keep scroll + empty-state handling consistent.
    grid.innerHTML = '';
    grid.className = 'files-grid files-grid-csd';
    const container = document.createElement('div');
    container.className = 'csd-root';
    container.innerHTML = renderComicDetail({
      series: _normSeries(series),
      chapters: chapters.map(_normChapter),
      view: _view,
      back: { label: 'Comics' },
    });
    grid.appendChild(container);
    return;
  }

  // Series grid. Sort comes from the shared Files sort dropdown —
  // values outside the comics-sort whitelist (e.g. 'size', 'author')
  // are sent as-is; the backend falls back to 'name' for anything it
  // doesn't recognize, which keeps the grid sensible even if the user
  // picked a sort that was meaningful for a different chip.
  const q = state.el.search?.value?.trim() || '';
  const sort = state.currentSort || '';
  const status = state.currentComicStatus || 'all';
  const completion = state.currentComicCompletion || '';
  const genre = state.currentComicGenre || '';
  const series = await _fetchSeries({ q, sort, status, completion, genre });
  _view.seriesCache = series;

  if (!series.length) {
    grid.innerHTML = `
      <div class="files-empty">
        <span class="files-empty-icon">\u{1F4DA}</span>
        <span class="files-empty-text">No comics yet</span>
        <span class="files-empty-hint">
          Connect Suwayomi, Komga, or Kavita under Cloud, then sync.
          Your library will group into series here.
        </span>
      </div>
    `;
    return;
  }

  grid.innerHTML = series.map(_renderSeriesCard).join('');
}

/** Partial re-render — swap just the chapter list body when filter/sort
 *  state changes, without rebuilding the hero (which is static for the
 *  current series). Much cheaper than re-rendering the whole detail. */
function _rerenderChapterList() {
  const listEl = document.querySelector('[data-chapter-list]');
  if (!listEl) return;
  listEl.innerHTML = renderChapterRows(_view.chapterCache.map(_normChapter), _view);
}

function _clearDetailContainer() {
  // Legacy chapter-list class from earlier iteration; sweep it so the
  // grid doesn't render with stale layout rules.
  const g = state.el.grid;
  if (!g) return;
  g.classList.remove('files-grid-chapter-list', 'files-grid-csd');
}


/** Wire click + keyboard listeners. Call once from initFiles. Uses event
 *  delegation on the Files panel root so we don't need to re-wire after
 *  every re-render. */
export function initComicsListeners() {
  const panel = state.el.panel;
  if (!panel) return;

  panel.addEventListener('click', async (e) => {
    if (!isComicsChipActive()) return;

    // Series card → drill into chapter list
    const seriesBtn = e.target.closest('.files-card-comic-series');
    if (seriesBtn) {
      e.preventDefault();
      const seriesId = seriesBtn.dataset.seriesId;
      if (!seriesId) return;
      _view.mode = 'chapters';
      _view.activeSeriesId = seriesId;
      _view.activeSeriesMeta = (_view.seriesCache || [])
        .find(s => s.id === seriesId) || null;
      _view.hideRead = false;                     // fresh filter per series
      _view.sortNewestFirst = true;
      _saveDrill();
      await renderComicsGrid();
      return;
    }

    // Continue-reading CTA
    if (e.target.closest('[data-action="open-cta-chapter"]')) {
      e.preventDefault();
      const btn = e.target.closest('[data-action="open-cta-chapter"]');
      const fileId = btn.dataset.chapterFileId;
      if (!fileId) return;
      await _openChapter(fileId);
      return;
    }

    // "Start from Ch. 1" secondary CTA
    if (e.target.closest('[data-action="open-first-chapter"]')) {
      e.preventDefault();
      const first = _view.chapterCache.slice().sort(_bySourceOrderAsc)[0];
      if (first) await _openChapter(first.id);
      return;
    }

    // Per-chapter toggle: ✓ button. Routed BEFORE the row-open path so
    // a click on the button doesn't fall through to opening the chapter.
    // The button's own data-action attribute is what disambiguates.
    const toggleBtn = e.target.closest('[data-action="toggle-read"]');
    if (toggleBtn) {
      e.preventDefault();
      e.stopPropagation();
      const row = toggleBtn.closest('.csd-chapter');
      const fileId = row?.dataset.chapterFileId;
      if (fileId) await _toggleChapterRead(fileId);
      return;
    }

    // Per-chapter "more" button: opens the per-row menu (Mark up to here).
    const moreBtn = e.target.closest('[data-action="chapter-more"]');
    if (moreBtn) {
      e.preventDefault();
      e.stopPropagation();
      const row = moreBtn.closest('.csd-chapter');
      const fileId = row?.dataset.chapterFileId;
      if (fileId) _openChapterMoreMenu(moreBtn, fileId);
      return;
    }

    // Series-wide "more" button: opens the bulk-actions popover (Mark
    // all read / Reset progress). Same popover pattern as the per-row
    // menu — reused via _openSeriesMoreMenu so the menu chrome stays
    // consistent across both surfaces.
    const seriesMoreBtn = e.target.closest('[data-action="series-more"]');
    if (seriesMoreBtn) {
      e.preventDefault();
      e.stopPropagation();
      _openSeriesMoreMenu(seriesMoreBtn);
      return;
    }

    // Chapter row
    const chapterRow = e.target.closest('.csd-chapter');
    if (chapterRow) {
      e.preventDefault();
      const fileId = chapterRow.dataset.chapterFileId;
      if (fileId) await _openChapter(fileId);
      return;
    }

    // Hide-read toggle
    if (e.target.closest('[data-action="toggle-hide-read"]')) {
      e.preventDefault();
      _view.hideRead = !_view.hideRead;
      const btn = e.target.closest('[data-action="toggle-hide-read"]');
      btn.classList.toggle('on', _view.hideRead);
      btn.setAttribute('aria-pressed', _view.hideRead ? 'true' : 'false');
      _rerenderChapterList();
      return;
    }

    // Sort toggle
    if (e.target.closest('[data-action="toggle-sort"]')) {
      e.preventDefault();
      _view.sortNewestFirst = !_view.sortNewestFirst;
      const btn = e.target.closest('[data-action="toggle-sort"]');
      btn.classList.toggle('reversed', !_view.sortNewestFirst);
      btn.setAttribute('aria-pressed', !_view.sortNewestFirst ? 'true' : 'false');
      // Swap label text without touching the arrow span
      const arrow = btn.querySelector('.csd-sort-arrow');
      btn.textContent = _view.sortNewestFirst ? 'Newest' : 'Oldest';
      if (arrow) btn.prepend(arrow);
      _rerenderChapterList();
      return;
    }

    // Back-to-series
    if (e.target.closest('[data-action="back-to-series"]')) {
      e.preventDefault();
      _view.mode = 'series';
      _view.activeSeriesId = '';
      _view.activeSeriesMeta = null;
      _view.chapterCache = [];
      _saveDrill();
      await renderComicsGrid();
      return;
    }
  });

  // Keyboard enter / space on chapter rows (tabindex=0, role=button).
  // Skip when focus is on one of the inline action buttons — those have
  // their own implicit Enter/Space activation as <button> elements, and
  // we don't want the row-open to also fire and steal them.
  panel.addEventListener('keydown', (e) => {
    if (!isComicsChipActive()) return;
    if (e.target.closest?.('.csd-chapter-actions button')) return;
    const row = e.target.closest?.('.csd-chapter');
    if (!row) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      const fileId = row.dataset.chapterFileId;
      if (fileId) _openChapter(fileId);
    }
  });

  // Chapter search — live filter as the user types. Small debounce keeps
  // the re-render off the typing hot path without feeling laggy.
  let searchDebounce = null;
  panel.addEventListener('input', (e) => {
    if (!isComicsChipActive()) return;
    const input = e.target.closest?.('[data-action="chapter-search"]');
    if (!input) return;
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => {
      _view.chapterQuery = input.value || '';
      _rerenderChapterList();
    }, 120);
  });
}

async function _openChapter(fileId) {
  const chapter = (_view.chapterCache || []).find(c => c.id === fileId);
  if (!chapter) return;
  const { openComicReader } = await import('../comic-reader/index.js?v=surface-handoff-20260512a');
  openComicReader(chapter, { siblings: _view.chapterCache });
}


// --- Read-state writes ----------------------------------------------------
// Three actions all flow through the same POST /api/media/progress
// endpoint, just with different payloads:
//   toggle          → flip is_finished on one chapter
//   markUpToHere    → set is_finished on every chapter with source_order
//                     ≤ the clicked one (cumulative catch-up)
//   bulk (series)   → "mark all read" / "reset progress" over the whole
//                     chapter list
//
// All three share `_pushChapterProgress` for the wire write + cache-mutation
// pattern, and `_runBulk` for the parallel-capped iteration. Optimistic
// updates: cache is mutated first, view re-renders, then the network
// requests fly. On any failure we rollback the affected rows from a
// snapshot taken at the start.
//
// We refresh the chapter list with `_rerenderChapterList()` after each
// state change so the visual indicator (green dot, progress bar, ✓
// button state) stays in sync without a full re-fetch.

const BULK_CONCURRENCY = 5;

/**
 * Build the progress-update payload for a chapter. is_finished=true sets
 * progress to 100% (current_time_s = page_count); =false zeroes it.
 * Falls back to a duration of 1 when page count is unknown so the
 * Suwayomi push_progress branch has a sensible last_page value.
 */
function _progressPayload(chapter, isFinished) {
  const meta = chapter.source_metadata || {};
  const extra = meta.extra || {};
  const total = Number(meta.duration_s) || Number(extra.page_count) || 1;
  return {
    current_time_s: isFinished ? total : 0,
    duration_s:     total,
    is_finished:    !!isFinished,
  };
}

/**
 * Mutate a chapter's local cache entry to reflect the new read state.
 * Mirrors what the backend writes to source_metadata in update_progress
 * so the UI's analyze + render functions see the same shape.
 */
function _applyLocalProgress(chapter, isFinished) {
  if (!chapter.source_metadata) chapter.source_metadata = {};
  const meta = chapter.source_metadata;
  const extra = meta.extra || {};
  const total = Number(meta.duration_s) || Number(extra.page_count) || 1;
  meta.duration_s    = total;
  meta.current_time_s = isFinished ? total : 0;
  meta.is_finished   = !!isFinished;
  meta.progress_pct  = isFinished ? 100 : 0;
  meta.last_read_at  = new Date().toISOString();
}

/**
 * Write progress for a single chapter to the server. Returns true on
 * success. Doesn't touch the local cache — caller is responsible for
 * the optimistic update + rollback strategy.
 */
async function _pushChapterProgress(chapter, isFinished) {
  try {
    const resp = await fetch(
      `/api/media/progress/${encodeURIComponent(chapter.id)}`,
      {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(_progressPayload(chapter, isFinished)),
      },
    );
    return resp.ok;
  } catch {
    return false;
  }
}

/**
 * Run an array of jobs with bounded concurrency. Each job is a function
 * returning a Promise. Resolves to the count of successful jobs (jobs
 * that resolved truthy). Used by the bulk and "mark up to here" paths
 * — keeping concurrency capped means we don't hammer Suwayomi with 500
 * parallel mutations on a long-running series.
 */
async function _runBulk(jobs) {
  let cursor = 0;
  let successes = 0;
  const workers = Array.from({ length: Math.min(BULK_CONCURRENCY, jobs.length) }, async () => {
    while (cursor < jobs.length) {
      const idx = cursor++;
      const ok = await jobs[idx]();
      if (ok) successes++;
    }
  });
  await Promise.all(workers);
  return successes;
}

/**
 * Toggle one chapter's read state. Optimistic — flips the local cache
 * first, re-renders, then writes upstream. On failure, rolls back and
 * surfaces a toast.
 */
async function _toggleChapterRead(fileId) {
  const chapter = (_view.chapterCache || []).find(c => c.id === fileId);
  if (!chapter) return;
  const wasFinished = !!(chapter.source_metadata || {}).is_finished;
  const next = !wasFinished;

  // Optimistic update.
  _applyLocalProgress(chapter, next);
  _rerenderChapterList();

  const ok = await _pushChapterProgress(chapter, next);
  if (!ok) {
    // Rollback to previous state on failure. Suwayomi's source extension
    // sometimes returns 5xx during heavy library scans; better to surface
    // the failure than silently lie about state.
    _applyLocalProgress(chapter, wasFinished);
    _rerenderChapterList();
    showToast(
      `Couldn't update read state. The provider may be busy — try again in a moment.`,
      'error', 3500,
    );
  }
}

/**
 * Mark every chapter with source_order ≤ the clicked chapter's source
 * order as read. Useful for "I read all of these elsewhere, catch me
 * up." Skips chapters that are already marked read so we don't re-write
 * for no reason.
 */
async function _markUpToHere(fileId) {
  const target = (_view.chapterCache || []).find(c => c.id === fileId);
  if (!target) return;
  const targetOrder = ((target.source_metadata || {}).extra || {}).chapter_source_order;
  if (targetOrder == null) {
    showToast("Couldn't determine chapter order — try refreshing the series.", 'error', 3500);
    return;
  }

  const queue = (_view.chapterCache || []).filter(c => {
    const order = ((c.source_metadata || {}).extra || {}).chapter_source_order;
    if (order == null) return false;
    if (order > targetOrder) return false;
    return !(c.source_metadata || {}).is_finished;
  });

  if (!queue.length) {
    showToast('Already read up to here.', 'info', 1800);
    return;
  }

  const total = queue.length;
  // 'loading' toast type renders a spinner and stays put until we
  // dismiss it explicitly — perfect for bulk operations whose duration
  // depends on Suwayomi's response time.
  const toastId = showToast(`Marking ${total} chapter${total === 1 ? '' : 's'}…`, 'loading');

  // Snapshot prior state for rollback in case nothing succeeds.
  const snapshot = queue.map(c => ({
    chapter:     c,
    wasFinished: !!(c.source_metadata || {}).is_finished,
  }));

  // Optimistic UI: flip everything in the queue, render, then network.
  for (const c of queue) _applyLocalProgress(c, true);
  _rerenderChapterList();

  const jobs = queue.map(c => () => _pushChapterProgress(c, true));
  const successes = await _runBulk(jobs);

  // Dismiss the in-progress toast and report.
  try { (await import('../app.js')).dismissToast(toastId); } catch { /* ignore */ }

  if (successes === total) {
    showToast(`Marked ${total} chapter${total === 1 ? '' : 's'} as read.`, 'success', 2400);
  } else if (successes === 0) {
    // Total failure — roll everything back.
    for (const { chapter, wasFinished } of snapshot) {
      _applyLocalProgress(chapter, wasFinished);
    }
    _rerenderChapterList();
    showToast(`Couldn't mark chapters — provider unavailable.`, 'error', 3500);
  } else {
    // Partial — leave successes flipped, roll back failures by re-checking
    // each pushed chapter against its snapshot. We can't tell which
    // specific chapters failed without per-job results, so we accept the
    // partial state and tell the user. A subsequent catalog sync will
    // reconcile.
    showToast(
      `Marked ${successes} of ${total} chapters. Some failed — try again or wait for the next sync.`,
      'warning', 4000,
    );
  }
}

/**
 * Series-wide bulk: flip every chapter in the series to the same state.
 * `mode` is 'all-read' (mark all finished) or 'reset' (mark all unread).
 * Reset goes through showConfirm because it's destructive on long-running
 * series — irreversible without re-reading or a manual flip back.
 */
async function _bulkSeriesProgress(mode) {
  const chapters = _view.chapterCache || [];
  if (!chapters.length) return;

  const wantsRead = mode === 'all-read';
  if (mode === 'reset') {
    const ok = await showConfirm({
      title:        'Reset reading progress?',
      message:      `This marks all ${chapters.length} chapter${chapters.length === 1 ? '' : 's'} as unread, both in Augmentum and on the source server.`,
      confirmLabel: 'Reset progress',
      variant:      'danger',
    });
    if (!ok) return;
  }

  // Skip chapters that already match the target state — saves wire calls.
  const queue = chapters.filter(c => {
    const finished = !!(c.source_metadata || {}).is_finished;
    return wantsRead ? !finished : finished;
  });
  if (!queue.length) {
    showToast(wantsRead ? 'All chapters already read.' : 'Nothing to reset.', 'info', 1800);
    return;
  }

  const total = queue.length;
  const verbing = wantsRead ? 'Marking' : 'Resetting';
  const toastId = showToast(`${verbing} ${total} chapter${total === 1 ? '' : 's'}…`, 'loading');

  const snapshot = queue.map(c => ({
    chapter:     c,
    wasFinished: !!(c.source_metadata || {}).is_finished,
  }));
  for (const c of queue) _applyLocalProgress(c, wantsRead);
  _rerenderChapterList();

  const jobs = queue.map(c => () => _pushChapterProgress(c, wantsRead));
  const successes = await _runBulk(jobs);

  try { (await import('../app.js')).dismissToast(toastId); } catch { /* ignore */ }

  if (successes === total) {
    const verb = wantsRead ? 'marked' : 'reset';
    showToast(`${total} chapter${total === 1 ? '' : 's'} ${verb}.`, 'success', 2400);
  } else if (successes === 0) {
    for (const { chapter, wasFinished } of snapshot) {
      _applyLocalProgress(chapter, wasFinished);
    }
    _rerenderChapterList();
    showToast(`Couldn't reach the provider — try again later.`, 'error', 3500);
  } else {
    showToast(
      `Updated ${successes} of ${total} chapters. Some failed — a sync will reconcile.`,
      'warning', 4000,
    );
  }
}

// --- "More" popovers -----------------------------------------------------
// Two popovers share the same chrome + dismissal logic via _openMoreMenu:
//   - per-chapter (anchored to a row's ⋯ button) — Mark up to here
//   - series-wide (anchored to the chapters-header ⋯ button) — Mark all
//     read / Reset progress
// One popover at a time across both. Reusing the helper keeps the menu
// styling, viewport-clamping, and outside-click dismissal in one spot.

let _moreMenu = null;
let _moreMenuDocClick = null;

function _closeMoreMenu() {
  if (_moreMenu) {
    _moreMenu.remove();
    _moreMenu = null;
  }
  if (_moreMenuDocClick) {
    document.removeEventListener('click', _moreMenuDocClick, true);
    document.removeEventListener('keydown', _moreMenuDocClick, true);
    _moreMenuDocClick = null;
  }
}

/**
 * Open a popover anchored to `anchorBtn`. `items` is an array of
 * `{ label, action, danger }` objects rendered as menu items. When an
 * item is clicked, the menu closes and the corresponding action key is
 * passed to `onSelect(action)`.
 */
function _openMoreMenu(anchorBtn, items, onSelect) {
  _closeMoreMenu();
  const menu = document.createElement('div');
  menu.className = 'csd-chapter-more-menu';
  menu.setAttribute('role', 'menu');
  menu.innerHTML = items.map(it => `
    <button type="button"
            class="csd-more-item${it.danger ? ' is-danger' : ''}"
            data-action="${escapeHtml(it.action)}" role="menuitem">
      ${escapeHtml(it.label)}
    </button>
  `).join('');
  document.body.appendChild(menu);
  _moreMenu = menu;

  // Position the menu under the anchor, right-aligned. Clamp to
  // viewport so it doesn't run off the edge on the rightmost button.
  const r = anchorBtn.getBoundingClientRect();
  const mw = menu.offsetWidth;
  const left = Math.max(8, Math.min(window.innerWidth - mw - 8, r.right - mw));
  const top = Math.min(window.innerHeight - menu.offsetHeight - 8, r.bottom + 6);
  menu.style.left = `${left}px`;
  menu.style.top  = `${top}px`;

  // Close on outside click or Escape. Capture-phase so a click on the
  // menu items themselves still gets handled by their own listener
  // before this dismisses the menu.
  _moreMenuDocClick = (e) => {
    if (e.type === 'keydown' && e.key !== 'Escape') return;
    if (e.type === 'click' && menu.contains(e.target)) return;
    _closeMoreMenu();
  };
  // Defer attaching the document listeners until after the click that
  // opened the menu has fully unwound — otherwise the same click event
  // (capturing) would close it immediately.
  setTimeout(() => {
    document.addEventListener('click',   _moreMenuDocClick, true);
    document.addEventListener('keydown', _moreMenuDocClick, true);
  }, 0);

  // Wire menu items via delegation.
  menu.addEventListener('click', async (e) => {
    const item = e.target.closest('[data-action]');
    if (!item) return;
    e.preventDefault();
    e.stopPropagation();
    const action = item.dataset.action;
    _closeMoreMenu();
    if (onSelect) await onSelect(action);
  });
}

function _openChapterMoreMenu(anchorBtn, fileId) {
  _openMoreMenu(
    anchorBtn,
    [{ label: 'Mark up to here as read', action: 'mark-up-to-here' }],
    async (action) => {
      if (action === 'mark-up-to-here') await _markUpToHere(fileId);
    },
  );
}

function _openSeriesMoreMenu(anchorBtn) {
  _openMoreMenu(
    anchorBtn,
    [
      { label: 'Mark all as read',  action: 'mark-all-read' },
      { label: 'Reset progress',    action: 'reset-progress', danger: true },
    ],
    async (action) => {
      if (action === 'mark-all-read')   await _bulkSeriesProgress('all-read');
      else if (action === 'reset-progress') await _bulkSeriesProgress('reset');
    },
  );
}
