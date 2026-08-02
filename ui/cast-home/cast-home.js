/**
 * cast-home.js — TV idle landing.
 *
 * Pure consumer: fetches /api/cast/library/home, renders rails, ticks
 * the clock, optionally cycles backdrops. No input handling — TVs
 * today are non-interactive in our flow (phone is the navigator).
 *
 * Refresh strategy: re-fetch on focus + every 5 minutes. Backdrop
 * cycle is independent (30s) and only runs if there's art to cycle.
 */

const REFRESH_MS = 5 * 60 * 1000;
const BACKDROP_CYCLE_MS = 30 * 1000;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// Per-receiver prefs bag, posted in by cast-receiver after it fetches
// /api/cast/receiver-self/prefs. Until that lands we render every
// rail and run the backdrop cycle (i.e. defaults-on). The first
// ``augmentum.prefs`` message triggers a re-render with whatever the
// user has hidden / disabled.
let _prefs = null;
// Trusted-receiver id, captured alongside _prefs from the parent
// cast-receiver shell. Passed on /api/cast/library/home so the
// server pre-filters rails by this receiver's rails_visible bag —
// previously cast-home filtered locally AFTER the server returned
// every rail, wasting a file_index fetch per hidden rail.
let _trustedId = '';

// Receiver id arrives in two messages: ``augmentum.surface_init`` on
// iframe load AND a follow-up ``augmentum.identity`` when the server's
// CMD_IDENTITY handshake completes (the two race). Tile-tap-to-cast
// requires this id; the click handler shows a hint if it hasn't
// landed yet (rare — only on the very first interaction after a
// fresh receiver pair).
let _receiverId = '';

// Items map indexed by file_id so tile click handlers can resolve the
// full item shape (cover, kind, play descriptor) without re-fetching
// or stashing every field as data-attributes on the article element.
const _itemById = new Map();

function _railVisible(slug) {
  // Missing keys = visible. Only explicit ``false`` hides a rail.
  if (!_prefs) return true;
  const rv = _prefs.rails_visible;
  if (!rv || typeof rv !== 'object') return true;
  return rv[slug] !== false;
}


/* ── Clock ──────────────────────────────────────────────────────── */

function tickClock() {
  const now = new Date();
  const time = now.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  const date = now.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' });
  const timeEl = $('[data-cs-time]');
  const dateEl = $('[data-cs-date]');
  if (timeEl) timeEl.textContent = time;
  if (dateEl) dateEl.textContent = date;
}


/* ── Brand glyph orbits ─────────────────────────────────────────
 *
 * Three orbs trace ellipses around the central core. SVG SMIL's
 * ``animateMotion`` is unreliable inside transformed groups on some
 * Chromium WebView builds (the cast TV's Brave 148 in particular),
 * so we drive cx/cy via requestAnimationFrame instead. The parent
 * <g transform="rotate(...)"> still handles the orbit plane tilt;
 * we only compute the position on the un-rotated ellipse.
 *
 * RAF naturally pauses when the tab/iframe is hidden, so this is
 * cheap. Bails entirely under prefers-reduced-motion. */

const ORB_SPEC = [
  // [orb data-key, ellipse rx, ellipse ry, period ms, direction]
  ['narr', 10, 3.4, 14000,  1],
  ['anal',  8, 2.6, 10000, -1],  // opposite direction matches the
                                  // SMIL sweep-flag pair on this ring
  ['pass',  6, 2,    7000,  1],
];

function startBrandGlyphOrbit() {
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return;
  const orbs = ORB_SPEC.map(([key, rx, ry, dur, dir]) => {
    const el = document.querySelector(`.brand-glyph [data-cs-orb="${key}"]`);
    return el ? { el, rx, ry, dur, dir } : null;
  }).filter(Boolean);
  if (!orbs.length) return;

  function frame(tNow) {
    for (const o of orbs) {
      const phase = ((tNow % o.dur) / o.dur) * Math.PI * 2 * o.dir;
      // Center of the SVG viewBox is (12,12). The parent <g> rotates
      // around that same point, so we compute the orb's position on
      // the un-rotated ellipse and let the parent's transform tilt
      // it into place.
      const cx = 12 + Math.cos(phase) * o.rx;
      const cy = 12 + Math.sin(phase) * o.ry;
      o.el.setAttribute('cx', cx.toFixed(3));
      o.el.setAttribute('cy', cy.toFixed(3));
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}


/* ── Rails ──────────────────────────────────────────────────────── */

function escapeAttr(s) {
  return String(s ?? '').replace(/[<>&"']/g, (c) => ({
    '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function renderTile(item) {
  // Match cast-control's per-entity-kind aspect logic. Movies +
  // series posters are 2:3 — only episodes + music videos use
  // landscape 16:9 thumbs.
  const ek = (item.entity_kind || '').toLowerCase();
  const landscape = item.kind === 'video'
    && (ek === 'episode' || ek === 'music_video');
  const cover = item.cover_url || '';
  const sub = item.subtitle || (item.source ? item.source : '');
  const pct = Math.max(0, Math.min(100, item.progress_pct || 0));
  // Stash the full item by file_id so the delegated click handler
  // can resolve play descriptor + kind without rehydrating.
  if (item.file_id) _itemById.set(item.file_id, item);
  return `
    <article class="tile" data-cs-tile-id="${escapeAttr(item.file_id || '')}">
      <div class="tile-art ${landscape ? 'landscape' : ''}">
        ${cover
          ? `<img src="${escapeAttr(cover)}" alt="" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('div'), {className:'tile-art-placeholder', textContent: '${escapeAttr(item.kind || 'media')}'}))">`
          : `<div class="tile-art-placeholder">${escapeAttr(item.kind || 'media')}</div>`
        }
        ${pct > 0 ? `<div class="tile-progress"><div class="tile-progress-fill" style="width:${pct}%"></div></div>` : ''}
      </div>
      <div class="tile-title">${escapeAttr(item.title)}</div>
      ${sub ? `<div class="tile-sub">${escapeAttr(sub)}</div>` : ''}
    </article>
  `;
}

// Roman numerals for rail headers. 8 rails ship today (rail_catalog.py);
// if more land we fall through to integers rather than crash, but the
// fact lookup in rail_catalog will catch that case at audit time.
const _ROMAN_NUMERALS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'];

function _renderRailsHost(sections) {
  const host = document.querySelector('[data-cs-rails-host]');
  if (!host) return;
  // Server filters by trusted_id once prefs land, so this list is
  // already user-shaped on every fetch after the first. _railVisible
  // is the local belt-and-brace for the brief window before the
  // augmentum.prefs postMessage arrives — same defense the old
  // renderRail had. Empty rails are dropped here too so a section
  // the server kept (no prefs yet) but has no items doesn't show
  // an empty header on the TV.
  const visible = (sections || []).filter(
    (s) => _railVisible(s.id) && Array.isArray(s.items) && s.items.length > 0,
  );
  host.innerHTML = visible.map((section, i) => {
    const numeral = _ROMAN_NUMERALS[i] || String(i + 1);
    const tiles = section.items.map(renderTile).join('');
    return `
      <section class="rail" data-cs-rail="${escapeAttr(section.id)}">
        <header class="rail-head">
          <span class="rail-numeral">${escapeAttr(numeral)}.</span>
          <span class="rail-title">${escapeAttr(section.title || section.id)}</span>
        </header>
        <div class="rail-strip" data-cs-strip="${escapeAttr(section.id)}">${tiles}</div>
      </section>
    `;
  }).join('');
}


/* ── Backdrop cycle ─────────────────────────────────────────────── */

let _backdropEl = null;
let _backdropUrls = [];
let _backdropIdx = 0;
let _backdropTimer = null;

function collectBackdrops(sections) {
  const out = [];
  for (const section of sections || []) {
    for (const item of section.items || []) {
      if (item.backdrop_url) out.push(item.backdrop_url);
    }
  }
  // Dedupe while preserving order.
  return Array.from(new Set(out));
}

function cycleBackdrop() {
  if (!_backdropEl || _backdropUrls.length === 0) return;
  const url = _backdropUrls[_backdropIdx % _backdropUrls.length];
  _backdropIdx += 1;
  // Probe load first — if 404, skip this URL.
  const probe = new Image();
  probe.onload = () => {
    _backdropEl.style.backgroundImage = `url("${url}")`;
    _backdropEl.classList.add('on');
  };
  probe.onerror = () => {
    // Drop bad URLs so future cycles don't keep retrying them.
    _backdropUrls = _backdropUrls.filter((u) => u !== url);
  };
  probe.src = url;
}

function startBackdropCycle() {
  _backdropEl = document.querySelector('[data-cs-backdrop]');
  clearInterval(_backdropTimer);
  // Pref-gated. When the user disables the cycle we also reset the
  // backdrop element to a neutral state so a stale image doesn't
  // freeze on screen until the next page load.
  if (_prefs && _prefs.backdrop_cycle === false) {
    if (_backdropEl) {
      _backdropEl.classList.remove('on');
      _backdropEl.style.backgroundImage = '';
    }
    return;
  }
  if (_backdropUrls.length === 0 || !_backdropEl) return;
  cycleBackdrop();
  _backdropTimer = setInterval(cycleBackdrop, BACKDROP_CYCLE_MS);
}


/* ── Data fetch ─────────────────────────────────────────────────── */

async function refresh() {
  let body;
  try {
    const qs = _trustedId
      ? `?trusted_id=${encodeURIComponent(_trustedId)}`
      : '';
    const r = await fetch(`/api/cast/library/home${qs}`, {
      credentials: 'same-origin',
      cache: 'no-store',
    });
    if (!r.ok) {
      console.warn('[cast-home] library fetch returned', r.status);
      return;
    }
    body = await r.json();
  } catch (err) {
    console.warn('[cast-home] library fetch threw', err);
    return;
  }

  // Render every rail the server returned (in catalog display order).
  // The host approach replaces an older 2-slug hardcoded loop that
  // showed only Continue + Recently Added on the TV regardless of
  // what the user enabled in cast-control's prefs sheet.
  _renderRailsHost(body.sections || []);

  _backdropUrls = collectBackdrops(body.sections);
  _backdropIdx = 0;
  startBackdropCycle();

  const status = document.querySelector('[data-cs-status]');
  if (status) {
    const total = (body.sections || []).reduce((n, s) => n + (s.items || []).length, 0);
    status.textContent = total > 0
      ? `${total} item${total === 1 ? '' : 's'} ready to cast.`
      : 'Listening for casts.';
  }
}


/* ── Follow mode ────────────────────────────────────────────────
 *
 * cast-receiver forwards ``nav`` cmds from the controller as
 * ``augmentum.nav`` postMessages. We listen, fetch the matching
 * payload, and swap the main content area between the default
 * home rails and a focused "you're browsing X" view.
 *
 * Stays calm — no animations, just an editorial swap. The TV is
 * presence furniture, not a noisy mirror. */

const followEl = (() => {
  // Lazy-create the follow container so home-mode HTML stays as-is.
  // Lives between the masthead and the colophon. We toggle .on/off
  // along with the original rails' display to swap views.
  const main = document.querySelector('main.home');
  if (!main) return null;
  const el = document.createElement('div');
  el.className = 'follow';
  el.hidden = true;
  // Insert before the colophon (last child).
  const colophon = main.querySelector('.colophon');
  if (colophon) {
    main.insertBefore(el, colophon);
  } else {
    main.appendChild(el);
  }
  return el;
})();

function showHomeMode() {
  if (followEl) followEl.hidden = true;
  document.querySelectorAll('[data-cs-rails-host] > .rail').forEach((r) => {
    // Re-show whatever rails were visible before follow mode took over.
    r.hidden = !r.dataset.csHadItems;
  });
}

function showFollowMode(innerHtml) {
  // Hide the default rails — only one mode visible at a time.
  document.querySelectorAll('[data-cs-rails-host] > .rail').forEach((r) => {
    r.dataset.csHadItems = !r.hidden ? '1' : '';
    r.hidden = true;
  });
  if (!followEl) return;
  followEl.innerHTML = innerHtml;
  followEl.hidden = false;
}

async function renderFollowSection(slug) {
  console.log('[cast-home] renderFollowSection start', { slug, followElExists: !!followEl });
  showFollowMode(`<div class="follow-loading">Loading…</div>`);
  try {
    const r = await fetch(
      `/api/cast/library/section/${encodeURIComponent(slug)}?limit=60`,
      { credentials: 'same-origin', cache: 'no-store' },
    );
    console.log('[cast-home] renderFollowSection fetch', { slug, status: r.status });
    if (!r.ok) {
      showFollowMode(`<div class="follow-loading">Couldn't load (HTTP ${r.status}).</div>`);
      return;
    }
    const body = await r.json();
    const items = body.items || [];
    console.log('[cast-home] renderFollowSection items', { slug, count: items.length });
    showFollowMode(`
      <header class="follow-head">
        <span class="follow-overline">browsing</span>
        <h2 class="follow-title">${escapeText(body.title || slug)}</h2>
      </header>
      <div class="follow-grid">
        ${items.map((it) => renderTile(it)).join('')}
      </div>
    `);
  } catch (err) {
    console.warn('[cast-home] renderFollowSection threw', err);
    showFollowMode(`<div class="follow-loading">Network error.</div>`);
  }
}

async function renderFollowSeries(fileId, kindHint) {
  // Comic series and video series have separate drill-in endpoints
  // (chapters/ vs episodes/) and different response shapes. We prefer
  // the explicit kind hint from the nav payload when the controller
  // provides one; otherwise fall back to trying chapters first
  // (cheap if 404) so a comic series doesn't render as "no episodes".
  const isComic = kindHint === 'comic'
    || kindHint === 'comic_series';
  showFollowMode(`<div class="follow-loading">${isComic ? 'Loading chapters' : 'Loading episodes'}…</div>`);

  const endpoints = isComic
    ? [`/api/cast/library/chapters/${encodeURIComponent(fileId)}`]
    : [`/api/cast/library/episodes/${encodeURIComponent(fileId)}`,
       `/api/cast/library/chapters/${encodeURIComponent(fileId)}`];

  let body = null;
  for (const ep of endpoints) {
    try {
      const r = await fetch(ep, { credentials: 'same-origin', cache: 'no-store' });
      if (r.ok) { body = await r.json(); break; }
    } catch { /* try next */ }
  }
  if (!body) {
    showFollowMode(`<div class="follow-loading">Couldn't load series.</div>`);
    return;
  }

  const seriesTile = body.series || {};
  // Comic response: {series, chapters: [...]}. Video response: {series, seasons: [...]}.
  if (Array.isArray(body.chapters)) {
    const chapters = body.chapters;
    showFollowMode(`
      <header class="follow-head">
        <span class="follow-overline">series</span>
        <h2 class="follow-title">${escapeText(seriesTile.title || 'Series')}</h2>
        <div class="follow-sub">${chapters.length} chapter${chapters.length === 1 ? '' : 's'}</div>
      </header>
      <div class="follow-grid follow-grid-episodes">
        ${chapters.map(renderTile).join('')}
      </div>
    `);
    return;
  }
  const seasons = body.seasons || [];
  const epCount = seasons.reduce((n, s) => n + (s.episodes || []).length, 0);
  showFollowMode(`
    <header class="follow-head">
      <span class="follow-overline">series</span>
      <h2 class="follow-title">${escapeText(seriesTile.title || 'Series')}</h2>
      <div class="follow-sub">${epCount} episode${epCount === 1 ? '' : 's'}</div>
    </header>
    ${seasons.map((season) => `
      <section class="follow-season">
        <h3 class="follow-season-head">${escapeText(season.label || `Season ${season.season_number}`)}</h3>
        <div class="follow-grid follow-grid-episodes">
          ${(season.episodes || []).map(renderTile).join('')}
        </div>
      </section>
    `).join('')}
  `);
}

function escapeText(s) {
  return String(s ?? '').replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]));
}

window.addEventListener('message', (ev) => {
  const data = ev.data;
  if (!data || typeof data !== 'object') return;
  // Per-receiver prefs arrive from the parent cast-receiver shortly
  // after this iframe mounts. Adopt the bag + re-run the data path
  // so the rail filter takes effect immediately (rather than waiting
  // for the 5-minute periodic refresh).
  if (data.type === 'augmentum.prefs') {
    _prefs = (data.payload && typeof data.payload === 'object')
      ? data.payload : null;
    if (typeof data.trusted_id === 'string') {
      _trustedId = data.trusted_id;
    }
    // Re-apply visibility to currently-rendered rails + restart the
    // backdrop cycle (which respects backdrop_cycle pref now).
    refresh();
    return;
  }
  // Server-side library_invalidate fanout (debounced ~30s) — forwarded
  // by cast-receiver after a /api/media/progress write lands. Lets the
  // Continue rail reorder within ~1s of the user playing something via
  // the controller, instead of waiting up to 5 minutes for the next
  // polling tick.
  if (data.type === 'augmentum.library_invalidate') {
    refresh();
    return;
  }
  // Receiver id arrives via surface_init (iframe load) AND a dedicated
  // identity message (when the server's identity handshake races
  // against iframe load). Either path hydrates _receiverId.
  if (data.type === 'augmentum.surface_init' && data.receiver_id) {
    _receiverId = String(data.receiver_id);
    return;
  }
  if (data.type === 'augmentum.identity' && data.receiver_id) {
    _receiverId = String(data.receiver_id);
    return;
  }
  if (data.type !== 'augmentum.nav') return;
  const payload = data.payload || {};
  const view = String(payload.view || 'home');
  console.log('[cast-home] nav received', payload);
  // ── Trackpad-style remote events ──────────────────────────────
  // Phone-side scroll pad sends these when the user drags or taps
  // it. We translate proportionally to the TV viewport. Scroll moves
  // the page; tap fires a synthetic click at the equivalent position
  // so tiles can self-action (cast on tap — wired separately).
  if (view === 'scroll' && typeof payload.dy === 'number') {
    window.scrollBy({ top: payload.dy, left: payload.dx || 0, behavior: 'auto' });
    return;
  }
  if (view === 'tap'
      && typeof payload.x_pct === 'number'
      && typeof payload.y_pct === 'number') {
    const x = Math.round(window.innerWidth * payload.x_pct);
    const y = Math.round(window.innerHeight * payload.y_pct);
    const el = document.elementFromPoint(x, y);
    if (el) {
      // Climb to the nearest actionable element (a tile or button).
      // elementFromPoint returns the deepest hit, which is often a
      // span/img inside the actually-clickable tile.
      const actionable = el.closest('[data-cs-tile-id], .tile, a, button')
        || el;
      try { actionable.click(); }
      catch (err) { console.warn('[cast-home] tap click threw', err); }
    }
    return;
  }
  if (view === 'home') {
    showHomeMode();
    return;
  }
  if (view === 'section' && payload.slug) {
    renderFollowSection(String(payload.slug));
    return;
  }
  if (view === 'series' && payload.file_id) {
    renderFollowSeries(String(payload.file_id), payload.kind || payload.entity_kind);
    return;
  }
  console.warn('[cast-home] nav payload unrecognised', payload);
});


/* ── Tile click handler ────────────────────────────────────────
 * Wired via event delegation so it works for both home rails AND
 * the follow-mode grid (rebuilt on every nav). Click → cast that
 * item to this receiver. For series tiles (browse_series action),
 * render the drill-in inline instead (consistent with the existing
 * follow path). */

async function _handleTileClick(tile) {
  const fileId = tile.dataset.csTileId;
  if (!fileId) return;
  const item = _itemById.get(fileId);
  if (!item) {
    console.warn('[cast-home] tile click: unknown file_id', fileId);
    return;
  }
  const play = item.play || {};
  if (play.action === 'browse_series') {
    // Local drill-in — same as follow nav, just triggered from a tap.
    renderFollowSeries(fileId, item.kind || item.entity_kind);
    return;
  }
  if (!play.surface_kind || !play.surface_url) {
    console.warn('[cast-home] tile click: no play descriptor', item);
    return;
  }
  if (!_receiverId) {
    console.warn('[cast-home] tile click: no receiver id yet');
    return;
  }
  try {
    const r = await fetch('/api/cast/send', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        receiver_id: _receiverId,
        surface_kind: play.surface_kind,
        surface_url: play.surface_url,
        slot: 'main',
      }),
    });
    if (!r.ok) {
      console.warn('[cast-home] cast send rejected', r.status);
    }
  } catch (err) {
    console.warn('[cast-home] cast send failed', err);
  }
}

document.addEventListener('click', (ev) => {
  const tile = ev.target.closest('[data-cs-tile-id]');
  if (!tile) return;
  _handleTileClick(tile);
});


/* ── Boot ───────────────────────────────────────────────────────── */

tickClock();
setInterval(tickClock, 1000);
startBrandGlyphOrbit();

refresh();
setInterval(refresh, REFRESH_MS);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refresh();
});
