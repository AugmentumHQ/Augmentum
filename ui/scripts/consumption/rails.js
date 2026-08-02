/**
 * consumption/rails.js — rail renderer for Media + cast surfaces.
 *
 * Each rail = header (title + optional explainable subtitle for AI rails
 * + optional "See all" jump) + horizontal strip of tiles with hover
 * paging chevrons. Scrollbars are hidden (the chevrons + drag/touch are
 * the affordance) — see .media-rail-strip in media.css. No hero, no
 * autoplay, no parallax.
 */

import { escapeHtml } from '../app.js';
import { renderTile } from './tile.js';

// Rails whose art is episodic/landscape render wide (backdrop-first)
// tiles. The Continue rail is the canonical case — resumed video looks
// like a freeze-frame, not a poster.
const DEFAULT_WIDE_SLUGS = ['resume'];

// Rails that don't support a paginated See-all drill-in. Resume is
// capped server-side by design (a "page 4 of half-watched items" view
// helps nobody).
const DEFAULT_NO_SEE_ALL = ['resume'];

export function renderRails(container, sections, {
  onTileActivate,
  onTileSecondary,
  onSeeAll = null,
  hiddenSlugs = [],
  wideSlugs = DEFAULT_WIDE_SLUGS,
  noSeeAllSlugs = DEFAULT_NO_SEE_ALL,
} = {}) {
  container.innerHTML = '';
  const skip = new Set(hiddenSlugs);
  const wide = new Set(wideSlugs);
  const noSeeAll = new Set(noSeeAllSlugs);
  for (const section of sections) {
    if (!section || skip.has(section.id) || !(section.items?.length)) continue;
    container.appendChild(renderRail(section, {
      onTileActivate,
      onTileSecondary,
      onSeeAll: (onSeeAll && !noSeeAll.has(section.id)) ? onSeeAll : null,
      variant: wide.has(section.id) ? 'wide' : 'portrait',
    }));
  }
}

function renderRail(section, { onTileActivate, onTileSecondary, onSeeAll, variant }) {
  const rail = document.createElement('section');
  rail.className = 'media-rail';
  rail.dataset.slug = section.id || '';

  const header = document.createElement('header');
  header.className = 'media-rail-header';
  header.innerHTML = `
    <div class="media-rail-heading">
      <h3 class="media-rail-title">${escapeHtml(section.title || section.id || '')}</h3>
      ${section.reason
        ? `<p class="media-rail-reason">${escapeHtml(section.reason)}</p>`
        : ''}
    </div>
    ${onSeeAll
      ? `<button class="media-rail-seeall" type="button" aria-label="See all ${escapeHtml(section.title || section.id || '')}">
           <span>See all</span>
           <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="9 18 15 12 9 6"/></svg>
         </button>`
      : ''}
  `;
  if (onSeeAll) {
    header.querySelector('.media-rail-seeall').addEventListener('click', () => {
      onSeeAll(section);
    });
  }
  rail.appendChild(header);

  const scroller = document.createElement('div');
  scroller.className = 'media-rail-scroller';

  const strip = document.createElement('div');
  strip.className = 'media-rail-strip';
  for (const item of section.items) {
    strip.appendChild(renderTile(item, {
      onActivate: onTileActivate,
      onSecondary: onTileSecondary,
      variant,
    }));
  }
  scroller.appendChild(strip);
  _wireChevrons(scroller, strip);
  rail.appendChild(scroller);
  return rail;
}

/* Hover paging chevrons. Hidden at the scroll edges; page by ~90% of
 * the visible width so the last partially-visible tile becomes the
 * first fully-visible one (the Plex/Jellyfin paging feel without the
 * permanent scrollbar noise). Exported so custom rails (e.g. the Media
 * Live TV rail, whose channel tiles aren't file tiles) get identical
 * paging chrome without duplicating this. */
export function wireChevrons(scroller, strip) {
  return _wireChevrons(scroller, strip);
}

function _wireChevrons(scroller, strip) {
  const mk = (dir) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `media-rail-chevron media-rail-chevron-${dir}`;
    btn.setAttribute('aria-label', dir === 'left' ? 'Scroll back' : 'Scroll forward');
    btn.innerHTML = dir === 'left'
      ? '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="15 18 9 12 15 6"/></svg>'
      : '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="9 18 15 12 9 6"/></svg>';
    btn.addEventListener('click', () => {
      const delta = Math.max(160, strip.clientWidth * 0.9);
      strip.scrollBy({ left: dir === 'left' ? -delta : delta, behavior: 'smooth' });
    });
    return btn;
  };
  const left = mk('left');
  const right = mk('right');
  scroller.appendChild(left);
  scroller.appendChild(right);

  let raf = 0;
  const update = () => {
    raf = 0;
    const max = strip.scrollWidth - strip.clientWidth;
    const x = strip.scrollLeft;
    left.classList.toggle('is-hidden', x <= 4);
    right.classList.toggle('is-hidden', max <= 8 || x >= max - 4);
  };
  const schedule = () => {
    if (!raf) raf = requestAnimationFrame(update);
  };
  strip.addEventListener('scroll', schedule, { passive: true });
  // Initial state after layout (images may still be loading, but tile
  // widths are fixed so scrollWidth is already correct).
  requestAnimationFrame(update);
}
