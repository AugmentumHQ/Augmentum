/**
 * consumption/tile.js — uniform tile for Media + (eventually) cast-control + cast-app.
 *
 * Comfort canon: title UNDER the cover (never overlaid), subtitle UNDER
 * the title, 2px progress footer ON the cover. No autoplay, no overlaid
 * synopsis, quiet at idle.
 *
 * Two variants:
 *   portrait (default) — 3:4 cover, library browse tiles
 *   wide               — 16:9 backdrop-first, used by the Continue rail
 *                        (and any rail whose art is episodic/landscape)
 *
 * State badges (all data already on the tile payload — no extra fetch):
 *   watched check   — is_finished, no progress bar shown alongside
 *   unplayed count  — series tiles with unwatched episodes
 *   duration        — leaf video/audio runtime, bottom-right, low key
 *
 * Tile shape comes from cast_routes._entry_to_tile:
 *   { file_id, title, subtitle, kind, entity_kind, cover_url, backdrop_url,
 *     progress_pct, duration_s, year, is_finished, unplayed_count,
 *     play: { action, surface_kind, surface_url } }
 */

import { escapeHtml } from '../app.js';

export function formatDuration(seconds) {
  const s = Number(seconds) || 0;
  if (s < 60) return '';
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  if (h > 0) return m > 0 ? `${h}h ${m}m` : `${h}h`;
  return `${m}m`;
}

export function renderTile(item, { onActivate, onSecondary, variant = 'portrait' } = {}) {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = variant === 'wide' ? 'media-tile media-tile-wide' : 'media-tile';
  el.dataset.fileId = item.file_id || '';
  el.dataset.kind = item.kind || '';
  el.dataset.action = item.play?.action || 'cast';

  // Wide tiles prefer the landscape backdrop; the poster cover is the
  // fallback (onerror swap below covers items without a real backdrop —
  // the backdrop route 404s for them).
  const cover = item.cover_url || '';
  const wideArt = (variant === 'wide' && item.backdrop_url) ? item.backdrop_url : cover;
  const art = variant === 'wide' ? wideArt : cover;
  const fallbackAttr = (variant === 'wide' && item.backdrop_url && cover)
    ? ` data-fallback-src="${escapeHtml(cover)}"`
    : '';

  const progress = Math.max(0, Math.min(100, Number(item.progress_pct) || 0));
  const isFinished = !!item.is_finished;
  const showProgress = !isFinished && progress > 0.5;

  const isSeries = (item.entity_kind || '') === 'series'
    || (item.play?.action === 'browse_series' && item.kind === 'video');
  const unplayed = Number(item.unplayed_count) || 0;
  const durationLabel = (!isSeries && (item.kind === 'video' || item.kind === 'audio'))
    ? formatDuration(item.duration_s)
    : '';

  const badges = [];
  if (isFinished) {
    badges.push(`<span class="media-tile-badge media-tile-badge-watched" title="Watched" aria-label="Watched">
      <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>
    </span>`);
  } else if (isSeries && unplayed > 0) {
    badges.push(`<span class="media-tile-badge media-tile-badge-count" title="${unplayed} unwatched">${unplayed > 99 ? '99+' : unplayed}</span>`);
  }
  if (durationLabel) {
    badges.push(`<span class="media-tile-duration">${escapeHtml(durationLabel)}</span>`);
  }

  el.innerHTML = `
    <div class="media-tile-cover">
      ${art
        ? `<img loading="lazy" decoding="async" src="${escapeHtml(art)}" alt=""${fallbackAttr}>`
        : `<div class="media-tile-cover-fallback">${escapeHtml((item.title || '?').slice(0, 1))}</div>`}
      ${badges.join('')}
      ${showProgress
        ? `<div class="media-tile-progress" style="width:${progress.toFixed(1)}%"></div>`
        : ''}
    </div>
    <div class="media-tile-title">${escapeHtml(item.title || 'Untitled')}</div>
    ${item.subtitle
      ? `<div class="media-tile-sub">${escapeHtml(item.subtitle)}</div>`
      : ''}
  `;

  // Backdrop 404 → swap to the poster cover instead of a dead image.
  const img = el.querySelector('img[data-fallback-src]');
  if (img) {
    img.addEventListener('error', () => {
      const fallback = img.dataset.fallbackSrc;
      if (fallback && img.src !== fallback) {
        delete img.dataset.fallbackSrc;
        img.src = fallback;
      }
    }, { once: true });
  }

  if (onActivate) {
    el.addEventListener('click', (ev) => {
      ev.preventDefault();
      onActivate(item, ev);
    });
  }
  if (onSecondary) {
    el.addEventListener('contextmenu', (ev) => {
      ev.preventDefault();
      onSecondary(item, ev);
    });
  }
  return el;
}
