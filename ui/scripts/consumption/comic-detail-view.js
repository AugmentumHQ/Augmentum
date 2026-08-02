/**
 * consumption/comic-detail-view.js — the shared "printed manga volume"
 * comic series-detail renderer (csd-* design).
 *
 * One source of truth for the tailored comic detail page (cover hero with
 * spine + fallback label, status/years/genres meta, continue-reading CTA,
 * chapter masthead + list). Both the Files comics surface
 * (ui/scripts/files/comics.js) and the Media drill-in
 * (ui/scripts/consumption/comic-series.js) render from this, so the two
 * never drift. The csd-* CSS lives in ui/styles/files.css (global).
 *
 * It operates on a NORMALIZED shape so callers with different raw data
 * (Files' source_metadata.extra blobs vs Media's flat chapter tiles) both
 * map into the same view. Callers own their own data fetching, cover-URL
 * resolution, and click wiring — the rendered HTML carries the same
 * `data-action` contract for both, so each surface delegates clicks to its
 * own handlers (Files: its panel-level listener; Media: its callbacks).
 *
 *   NormSeries  = { name, author, publisher, yearStarted, yearEnded,
 *                   status, genres: string[], description, coverUrl }
 *   NormChapter = { id, label, name, order, currentS, totalS,
 *                   isFinished, updatedAt }
 *   view        = { chapterQuery, hideRead, sortNewestFirst }
 */

import { escapeHtml } from '../app.js';
import { formatCount } from '../files/helpers.js';

// ── Reading-state analysis (pure, synchronous) ───────────────────────
// Continue-reading target, last-read timestamp, totals. Mirrors the
// original files/comics.js _analyzeChapters, but on the normalized shape.

export function analyzeChapters(chapters) {
  let inProgressChapter = null;
  let inProgressUpdatedAt = 0;
  let firstUnread = null;
  let lastReadAt = '';
  let lastReadAtTs = 0;
  let finishedCount = 0;
  let inProgressCount = 0;
  const sorted = chapters.slice().sort(_byOrderAsc);

  for (const ch of sorted) {
    const current = Number(ch.currentS) || 0;
    const isFinished = !!ch.isFinished;
    const updatedAtTs = Date.parse(_normalizeTs(ch.updatedAt)) || 0;

    if (isFinished) {
      finishedCount++;
      if (updatedAtTs > lastReadAtTs) { lastReadAtTs = updatedAtTs; lastReadAt = ch.updatedAt; }
      continue;
    }
    if (current > 0) {
      inProgressCount++;
      if (updatedAtTs > inProgressUpdatedAt) { inProgressUpdatedAt = updatedAtTs; inProgressChapter = ch; }
      if (updatedAtTs > lastReadAtTs) { lastReadAtTs = updatedAtTs; lastReadAt = ch.updatedAt; }
      continue;
    }
    if (firstUnread === null) firstUnread = ch;
  }

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

// ── Top-level render ─────────────────────────────────────────────────

/**
 * Render the full detail page HTML.
 *
 * @param {object}      opts
 * @param {NormSeries}  opts.series
 * @param {NormChapter[]} opts.chapters   reading order (ascending)
 * @param {object}      opts.view         {chapterQuery, hideRead, sortNewestFirst}
 * @param {{label:string}|null} [opts.back]  breadcrumb button; omit for none
 */
export function renderComicDetail({ series, chapters, view, back = null }) {
  const analysis = analyzeChapters(chapters);
  return `
    <div class="csd">
      ${back ? _renderBreadcrumb(back.label) : ''}
      ${_renderHero(series, analysis)}
      ${_renderChapterSection(chapters, view, analysis)}
    </div>
  `;
}

/** Just the chapter `<li>` rows — for cheap partial re-render on
 *  filter/sort/search changes (the hero is static for the series). */
export function renderChapterRows(chapters, view) {
  let ordered = chapters.slice().sort(_byOrderAsc);
  if (view.sortNewestFirst) ordered = ordered.reverse();
  let visible = view.hideRead ? ordered.filter(c => !c.isFinished) : ordered;
  const q = (view.chapterQuery || '').trim().toLowerCase();
  if (q) {
    visible = visible.filter(c =>
      (c.name || '').toLowerCase().includes(q)
      || (c.label || '').toLowerCase().includes(q));
  }
  if (!visible.length) {
    return `<li class="csd-chapter-empty">No chapters match${q ? ` "${escapeHtml(q)}"` : ' the current filter'}.</li>`;
  }
  return visible.map(_renderChapterRow).join('');
}

// ── Sections ─────────────────────────────────────────────────────────

function _renderBreadcrumb(label) {
  return `
    <button type="button" class="csd-back" data-action="back-to-series">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>
      </svg>
      ${escapeHtml(label || 'Comics')}
    </button>
  `;
}

function _renderHero(series, analysis) {
  const coverSrc = series.coverUrl || '';
  const years = _formatYears(series.yearStarted, series.yearEnded, series.status);
  const genreLine = Array.isArray(series.genres) && series.genres.length
    ? series.genres.slice(0, 6).join(' · ')
    : '';
  const statusLabel = (series.status || '').replace(/_/g, ' ').trim() || null;

  let ctaLabel = 'Start reading';
  let ctaChapterTarget = analysis.firstChapter;
  let ctaSubLabel = '';
  if (analysis.continueChapter && analysis.inProgressCount > 0) {
    ctaLabel = 'Continue reading';
    ctaChapterTarget = analysis.continueChapter;
    const current = Math.round(Number(analysis.continueChapter.currentS) || 0);
    const total = Math.round(Number(analysis.continueChapter.totalS) || 0);
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
  const ctaChapterLabel = ctaChapterTarget ? (ctaChapterTarget.label || '') : '';
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

function _renderChapterSection(chapters, view, analysis) {
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
                   value="${escapeHtml(view.chapterQuery || '')}"
                   spellcheck="false" autocomplete="off">
          </label>
          <button class="csd-toggle-btn${view.hideRead ? ' on' : ''}"
                  type="button" data-action="toggle-hide-read"
                  aria-pressed="${view.hideRead ? 'true' : 'false'}">
            <span class="csd-toggle-check">
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
            </span>
            Hide read
          </button>
          <button class="csd-sort${view.sortNewestFirst ? '' : ' reversed'}"
                  type="button" data-action="toggle-sort"
                  aria-pressed="${view.sortNewestFirst ? 'false' : 'true'}">
            <span class="csd-sort-arrow">↓</span>${view.sortNewestFirst ? 'Newest' : 'Oldest'}
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
        ${renderChapterRows(chapters, view)}
      </ul>
    </section>
  `;
}

function _renderChapterRow(chapter) {
  const current = Number(chapter.currentS) || 0;
  const total = Number(chapter.totalS) || 0;
  const finished = !!chapter.isFinished;
  const inProgress = !finished && current > 0;
  const stateClass = finished ? 'read' : (inProgress ? 'in-progress' : 'unread');

  let metaRight = '';
  if (finished)                  metaRight = 'Read';
  else if (inProgress && total)  metaRight = `${Math.round(current)} / ${Math.round(total)}`;
  else if (chapter.updatedAt)    metaRight = _relativeTime(chapter.updatedAt);

  const progressPct = inProgress && total
    ? Math.min(100, Math.round((current / total) * 100))
    : 0;

  const toggleLabel = finished ? 'Mark as unread' : 'Mark as read';
  return `
    <li class="csd-chapter state-${stateClass}"
        data-chapter-file-id="${escapeHtml(chapter.id)}"
        title="${escapeHtml(chapter.name)}"
        role="button" tabindex="0">
      <div class="csd-chapter-state"></div>
      <div class="csd-chapter-num">${escapeHtml(chapter.label || '')}</div>
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

// ── Pure helpers ─────────────────────────────────────────────────────

function _byOrderAsc(a, b) {
  return (Number(a.order) || 0) - (Number(b.order) || 0);
}

function _normalizeTs(s) {
  if (!s) return '';
  return /Z$|[+-]\d{2}:?\d{2}$/.test(s) ? s : s.replace(' ', 'T') + 'Z';
}

/** Drop trailing .0 on whole-number chapter numbers but keep .5 etc. */
export function trimChapterNumber(n) {
  const num = Number(n);
  if (Number.isNaN(num)) return String(n);
  return Number.isInteger(num) ? String(num) : num.toFixed(1).replace(/\.0$/, '');
}

function _formatYears(start, end, status) {
  if (!start && !end) return '';
  if (start && !end) {
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
  if (!name) return 'Untitled';
  const trimmed = name.trim();
  return trimmed.length > 14 ? trimmed.slice(0, 14).trim() + '…' : trimmed;
}

function _relativeTime(isoOrSql) {
  if (!isoOrSql) return '';
  try {
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
