/**
 * consumption/comic-series.js — series drill-in view for Media (and,
 * later, the cast surfaces).
 *
 * Renders the SAME tailored "printed manga volume" detail page as the
 * Files comics surface, via the shared renderer in
 * ./comic-detail-view.js (csd-* design; CSS in ui/styles/files.css). This
 * module owns the Media-specific data adapter (the /api/cast/library
 * chapter tiles), the read-state writes (POST /api/media/progress), and
 * the click wiring — the rendered HTML carries the same data-action
 * contract Files uses, so the two surfaces stay visually identical without
 * duplicating the markup.
 *
 * Input is the /api/cast/library/chapters payload ({series, chapters}) —
 * chapter tiles carry progress_pct / is_finished / duration_s; the series
 * block carries series_id so the rich comic_series record (publisher,
 * years, genres, description) can be fetched from
 * /api/files/comics/series/{id}.
 */

import { escapeHtml, showToast, showConfirm } from '../app.js';
import {
  renderComicDetail,
  renderChapterRows,
  trimChapterNumber,
} from './comic-detail-view.js';

const BULK_CONCURRENCY = 5;

export async function renderComicSeries(container, payload, {
  onOpenChapter,
  onSecondary,
} = {}) {
  const seriesTile = payload?.series || {};
  const rawChapters = Array.isArray(payload?.chapters) ? payload.chapters.slice() : [];

  // Rich series record — publisher/years/genres/description live on
  // comic_series, not the chapter tiles. Best-effort: the thin tile
  // header still renders if this fails or series_id is missing.
  let rich = null;
  if (seriesTile.series_id) {
    try {
      const resp = await fetch(
        `/api/files/comics/series/${encodeURIComponent(seriesTile.series_id)}`,
        { credentials: 'same-origin' },
      );
      if (resp.ok) rich = await resp.json();
    } catch (err) {
      console.warn('[comic-series] rich series fetch failed:', err);
    }
  }
  if (!container.isConnected) return;

  const series = _normSeries(seriesTile, rich);
  // Reading order: the payload arrives server-sorted ascending, so the
  // array index IS the canonical order. Keep the raw tiles addressable by
  // file_id so open/toggle actions can hand the caller the real object.
  const chapters = rawChapters.map((c, i) => _normChapter(c, i));
  const rawById = new Map(rawChapters.map((c) => [c.file_id, c]));
  const view = { chapterQuery: '', hideRead: false, sortNewestFirst: false };

  const renderAll = () => {
    container.innerHTML =
      `<div class="csd-root">${renderComicDetail({ series, chapters, view, back: null })}</div>`;
  };
  const rerenderRows = () => {
    const listEl = container.querySelector('[data-chapter-list]');
    if (listEl) listEl.innerHTML = renderChapterRows(chapters, view);
  };

  const openById = (fileId) => {
    const raw = rawById.get(fileId);
    if (raw && onOpenChapter) onOpenChapter(raw);
  };

  // Sync a normalized chapter's read state in place after a write so the
  // next re-render reflects it without re-fetching the payload.
  const applyReadState = (ids, finished) => {
    const set = new Set(ids);
    for (const ch of chapters) {
      if (!set.has(ch.id)) continue;
      ch.isFinished = finished;
      ch.currentS = finished ? ch.totalS : 0;
    }
  };

  renderAll();

  // ── Click wiring (shared data-action contract) ──────────────────────
  container.addEventListener('click', async (e) => {
    // Continue / start CTA + secondary "Start from Ch. 1".
    const cta = e.target.closest('[data-action="open-cta-chapter"]');
    if (cta) {
      e.preventDefault();
      openById(cta.dataset.chapterFileId);
      return;
    }
    if (e.target.closest('[data-action="open-first-chapter"]')) {
      e.preventDefault();
      const first = chapters.slice().sort((a, b) => a.order - b.order)[0];
      if (first) openById(first.id);
      return;
    }

    // Per-chapter read toggle (routed before the row-open path).
    const toggleBtn = e.target.closest('[data-action="toggle-read"]');
    if (toggleBtn) {
      e.preventDefault();
      e.stopPropagation();
      const row = toggleBtn.closest('.csd-chapter');
      const id = row?.dataset.chapterFileId;
      const raw = id && rawById.get(id);
      if (raw) {
        await _writeReadState([raw], !raw.is_finished);
        applyReadState([id], !!raw.is_finished);
        rerenderRows();
      }
      return;
    }

    // Per-chapter "more" → mark read up to here (the Media equivalent of
    // the Files per-row menu). onSecondary lets the host add its own.
    const moreBtn = e.target.closest('[data-action="chapter-more"]');
    if (moreBtn) {
      e.preventDefault();
      e.stopPropagation();
      const row = moreBtn.closest('.csd-chapter');
      const id = row?.dataset.chapterFileId;
      const choice = await _rowMenu(moreBtn);
      if (choice === 'up-to-here' && id) {
        const idx = chapters.findIndex((c) => c.id === id);
        if (idx >= 0) {
          const targets = chapters.slice(0, idx + 1)
            .filter((c) => !c.isFinished)
            .map((c) => rawById.get(c.id))
            .filter(Boolean);
          if (!targets.length) {
            showToast('Already read up to here.', 'info', 1800);
          } else {
            showToast(`Marking ${targets.length} chapter${targets.length === 1 ? '' : 's'} read…`, 'info', 2000);
            await _writeReadState(targets, true);
            applyReadState(targets.map((t) => t.file_id), true);
            rerenderRows();
          }
        }
      } else if (choice === 'secondary' && onSecondary) {
        const raw = id && rawById.get(id);
        if (raw) onSecondary(raw, e);
      }
      return;
    }

    // Series-wide "more" → bulk menu (mark all read / reset).
    const seriesMore = e.target.closest('[data-action="series-more"]');
    if (seriesMore) {
      e.preventDefault();
      e.stopPropagation();
      const choice = await _bulkMenu(seriesMore);
      if (choice === 'all-read') {
        const targets = chapters.filter((c) => !c.isFinished)
          .map((c) => rawById.get(c.id)).filter(Boolean);
        if (!targets.length) {
          showToast('Every chapter is already read.', 'info', 1800);
        } else {
          showToast(`Marking ${targets.length} chapter${targets.length === 1 ? '' : 's'} read…`, 'info', 2000);
          await _writeReadState(targets, true);
          applyReadState(targets.map((t) => t.file_id), true);
          rerenderRows();
        }
      } else if (choice === 'reset') {
        const ok = await showConfirm({
          title: 'Reset read state?',
          message: 'Every chapter in this series goes back to unread. Reading positions are cleared.',
          confirmLabel: 'Reset',
          variant: 'danger',
        });
        if (!ok) return;
        const targets = chapters
          .filter((c) => c.isFinished || c.currentS > 0)
          .map((c) => rawById.get(c.id)).filter(Boolean);
        showToast('Resetting series…', 'info', 2000);
        await _writeReadState(targets, false);
        applyReadState(targets.map((t) => t.file_id), false);
        rerenderRows();
      }
      return;
    }

    // Hide-read toggle.
    const hideBtn = e.target.closest('[data-action="toggle-hide-read"]');
    if (hideBtn) {
      e.preventDefault();
      view.hideRead = !view.hideRead;
      hideBtn.classList.toggle('on', view.hideRead);
      hideBtn.setAttribute('aria-pressed', String(view.hideRead));
      rerenderRows();
      return;
    }

    // Sort toggle.
    const sortBtn = e.target.closest('[data-action="toggle-sort"]');
    if (sortBtn) {
      e.preventDefault();
      view.sortNewestFirst = !view.sortNewestFirst;
      sortBtn.classList.toggle('reversed', !view.sortNewestFirst);
      sortBtn.setAttribute('aria-pressed', String(!view.sortNewestFirst));
      const arrow = sortBtn.querySelector('.csd-sort-arrow');
      sortBtn.textContent = view.sortNewestFirst ? 'Newest' : 'Oldest';
      if (arrow) sortBtn.prepend(arrow);
      rerenderRows();
      return;
    }

    // Chapter row → open.
    const row = e.target.closest('.csd-chapter');
    if (row) {
      e.preventDefault();
      openById(row.dataset.chapterFileId);
    }
  });

  // Chapter search (debounced).
  let timer = 0;
  container.addEventListener('input', (e) => {
    const input = e.target.closest('[data-action="chapter-search"]');
    if (!input) return;
    clearTimeout(timer);
    timer = setTimeout(() => {
      view.chapterQuery = input.value.trim();
      rerenderRows();
    }, 200);
  });
}

/* ── Normalizers: Media tiles → shared comic-detail-view shape ─────── */

function _normSeries(seriesTile, rich) {
  return {
    name: rich?.name || seriesTile.title || 'Series',
    author: rich?.author || '',
    publisher: rich?.publisher || '',
    yearStarted: rich?.year_started,
    yearEnded: rich?.year_ended,
    status: rich?.status || seriesTile.series_status || '',
    genres: Array.isArray(rich?.genres) ? rich.genres : [],
    description: rich?.description || '',
    coverUrl: seriesTile.cover_url || '',
  };
}

function _normChapter(c, index) {
  const cn = c.chapter_number;
  const label = (cn !== null && cn !== undefined && cn !== '')
    ? `Ch. ${trimChapterNumber(cn)}`
    : '';
  const total = Number(c.duration_s) || 0;
  const pct = Math.max(0, Math.min(100, Number(c.progress_pct) || 0));
  return {
    id: c.file_id,
    name: c.title || 'Untitled',
    label,
    order: index,                 // payload is already reading order
    currentS: c.is_finished ? total : Math.round((pct / 100) * total),
    totalS: total,
    isFinished: !!c.is_finished,
    updatedAt: c.last_played_at || c.updated_at || '',
  };
}

/* ── Read-state writes (Media: POST /api/media/progress) ──────────── */

async function _writeOne(rawChapter, finished) {
  const pages = Number(rawChapter.duration_s) || 0;
  try {
    const resp = await fetch(`/api/media/progress/${encodeURIComponent(rawChapter.file_id)}`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        current_time_s: finished ? pages : 0,
        duration_s: pages,
        is_finished: finished,
      }),
    });
    if (!resp.ok) return false;
    rawChapter.is_finished = finished;
    rawChapter.progress_pct = finished ? 100 : 0;
    return true;
  } catch {
    return false;
  }
}

async function _writeReadState(targets, finished) {
  let failures = 0;
  const queue = targets.slice();
  const workers = Array.from({ length: Math.min(BULK_CONCURRENCY, queue.length) }, async () => {
    while (queue.length) {
      const ch = queue.shift();
      const ok = await _writeOne(ch, finished);
      if (!ok) failures += 1;
    }
  });
  await Promise.all(workers);
  if (failures) {
    showToast(`Couldn't update ${failures} chapter${failures === 1 ? '' : 's'} — try again.`, 'error', 3200);
  }
}

/* ── Menus (anchored popovers) ───────────────────────────────────── */

function _rowMenu(anchor) {
  return _popover(anchor, [
    { value: 'up-to-here', label: 'Mark read up to here' },
  ]);
}

function _bulkMenu(anchor) {
  return _popover(anchor, [
    { value: 'all-read', label: 'Mark all read' },
    { value: 'reset', label: 'Reset read state…' },
  ]);
}

function _popover(anchor, items) {
  return new Promise((resolve) => {
    document.querySelector('.media-comic-bulk-menu')?.remove();
    const menu = document.createElement('div');
    menu.className = 'media-comic-bulk-menu';
    menu.setAttribute('role', 'menu');
    menu.innerHTML = items
      .map((it) => `<button type="button" role="menuitem" data-bulk="${escapeHtml(it.value)}">${escapeHtml(it.label)}</button>`)
      .join('');
    const rect = anchor.getBoundingClientRect();
    menu.style.top = `${rect.bottom + 6}px`;
    menu.style.right = `${Math.max(8, window.innerWidth - rect.right)}px`;
    document.body.appendChild(menu);
    const done = (val) => {
      menu.remove();
      document.removeEventListener('pointerdown', onAway, true);
      resolve(val);
    };
    const onAway = (ev) => { if (!menu.contains(ev.target)) done(null); };
    document.addEventListener('pointerdown', onAway, true);
    menu.querySelectorAll('[data-bulk]').forEach((btn) => {
      btn.addEventListener('click', () => done(btn.dataset.bulk));
    });
  });
}
