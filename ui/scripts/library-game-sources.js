// ui/scripts/library-game-sources.js
//
// Source registry for the Library's Games tab. Each source (js13k,
// AGSP-streamed games, future emulators, future GitHub builds)
// self-describes here. The library renders only sources whose
// `isEnabled(settings)` returns true, so the visibility wiring stays
// in one place per source.
//
// Adding a new source = one entry. No surgical library.js edits.
//
// Contract:
//   id            string, stable identifier ('js13k', 'streamed', ...)
//   label         button text
//   hint          sub-label under the button ("Plays in-app", etc.)
//   isRecommended boolean — adds the existing 'is-recommended' style
//   isEnabled     (settings) => boolean — gates the source button
//   subtitle      string shown above the discover grid for this source
//   sortable      boolean — when true, render the sort buttons
//   sortOptions   string[] — sort keys (default ['newest', 'popular'])
//   fetch         async (params) => { items, hasMore, nextPage } | { items }
//                  params = { sort, page }
//   renderCard    (item) => HTML string for one card
//   onLaunch      (item) => void — fired when the user clicks the
//                  primary action on a card (Pin, Launch, Open, etc.)
//                  May be omitted; defaults to "open details overlay".
//
// Sources don't import from library.js -- they're consumed by it.

import { escapeHtml, showToast } from './app.js';


// ── Procedural cover ──────────────────────────────────────────────
//
// Every library item gets an image. When a real cover isn't
// available (no thumbnail_url, libretro miss, etc.), we generate a
// deterministic gradient card based on a hash of the title +
// system. Same title always renders the same colours, so the
// fallback feels intentional rather than blank.
//
// References for the visual style:
//   * Steam library grid (3:4 aspect, dim title overlay)
//   * Plex placeholder cards (gradient + glyph)
//   * Apple TV "missing artwork" treatment


function _hashStr(s) {
  // Simple djb2 -- enough entropy to spread hue/saturation evenly
  // across a few hundred items without collisions visible to the eye.
  let h = 5381;
  const str = String(s || '');
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) + h) ^ str.charCodeAt(i);
  }
  return h >>> 0;
}

function _proceduralCoverHTML(opts) {
  // opts: { title, systemLabel?, glyph?, kind?, source? }
  // Returns an SVG-backed cover element that fits the same 3:4 slot
  // as a real <img>. Inline SVG so it picks up the parent's CSS
  // dimensions exactly and scales crisply on hi-DPI.
  const title = String(opts.title || 'Untitled');
  const sys = String(opts.systemLabel || opts.kind || opts.source || '');
  const h = _hashStr(title + '|' + sys);
  const hue = h % 360;
  const hue2 = (hue + 38) % 360;
  // Pick a glyph from the shared library based on kind/source so
  // an emulator card looks distinct from a streamed-game card from
  // a web-app card -- still no real cover, but the silhouette
  // gives the user a hint at what they're looking at.
  const glyph = opts.glyph || _glyphForKind(opts.kind, opts.source);
  // Initials from title (max 2 letters) -- big and readable.
  const initials = title
    .replace(/^(The|A|An)\s+/i, '')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map(w => w[0])
    .join('')
    .toUpperCase()
    .slice(0, 2) || '?';

  // Two-stop gradient + a faint diagonal sheen + the system badge.
  // System label rides at the bottom-left so it isn't covered by the
  // overlay title pill we render in the card body.
  return `
    <svg class="library-procedural-cover" viewBox="0 0 300 400" preserveAspectRatio="xMidYMid slice"
         xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id="pg-${h}" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="hsl(${hue}, 55%, 32%)"/>
          <stop offset="100%" stop-color="hsl(${hue2}, 60%, 18%)"/>
        </linearGradient>
        <radialGradient id="pgs-${h}" cx="0.3" cy="0.2" r="0.85">
          <stop offset="0%" stop-color="rgba(255,255,255,0.18)"/>
          <stop offset="100%" stop-color="rgba(255,255,255,0)"/>
        </radialGradient>
      </defs>
      <rect width="300" height="400" fill="url(#pg-${h})"/>
      <rect width="300" height="400" fill="url(#pgs-${h})"/>
      <text x="150" y="180" text-anchor="middle"
            font-family="system-ui, sans-serif" font-size="92"
            font-weight="700" fill="rgba(255,255,255,0.92)"
            letter-spacing="-2">${escapeHtml(initials)}</text>
      <text x="150" y="222" text-anchor="middle"
            font-family="system-ui, sans-serif" font-size="14"
            fill="rgba(255,255,255,0.55)" letter-spacing="2">${escapeHtml(glyph)}</text>
    </svg>`;
}


function _glyphForKind(kind, source) {
  // Word marks instead of emoji -- emoji rendering varies across
  // platforms, single-color text with the procedural background
  // reads cleanly everywhere. Falls back to "GAME" if unknown.
  if (kind === 'emulator_rom' || source === 'emulator') return 'RETRO';
  if (kind === 'streamed_game' || source === 'streamed') return 'STREAM';
  if (kind === 'js13k_game' || source === 'js13k') return 'JS13K';
  if (kind === 'web_app' || source === 'marketplace') return 'CURATED';
  return 'GAME';
}


// Single shared onerror handler — walks an embedded list of fallback
// thumbnail URLs (some sources, like js13k, ship multiple candidates
// because filename conventions vary by year), then swaps the <img> for
// the procedural cover when every candidate has 404'd. Never leaves a
// broken-image icon visible. One global per page so each card stays
// closure-free and the innerHTML payload stays small.
if (typeof window !== 'undefined' && !window.__augLibPreviewFallback) {
  window.__augLibPreviewFallback = function(img) {
    try {
      const remaining = JSON.parse(img.getAttribute('data-thumb-fallbacks') || '[]');
      if (Array.isArray(remaining) && remaining.length > 0) {
        const next = remaining.shift();
        img.setAttribute('data-thumb-fallbacks', JSON.stringify(remaining));
        img.src = next;
        return;
      }
    } catch { /* fall through */ }
    const procedural = img.getAttribute('data-procedural-fallback');
    if (procedural) {
      try { img.outerHTML = procedural; return; } catch { /* fall through */ }
    }
    // Last resort if procedural HTML is unavailable: a typed placeholder
    // styled like the rest of the surface, never the browser default.
    const ph = document.createElement('span');
    ph.className = 'library-preview-icon';
    ph.textContent = '🎮';
    img.replaceWith(ph);
  };
}

function _ensurePreviewHTML(thumbUrl, title, opts) {
  // Single source of truth: every card body's preview slot goes
  // through this. Accepts either a single URL or an array of candidate
  // URLs (walked in order). If every URL fails (or none was given),
  // we drop into the procedural cover so we NEVER end up with a broken
  // image icon.
  const candidates = Array.isArray(thumbUrl)
    ? thumbUrl.filter(Boolean)
    : (thumbUrl ? [thumbUrl] : []);
  if (candidates.length === 0) {
    return _proceduralCoverHTML({ title, ...opts });
  }
  const first = candidates[0];
  const rest = candidates.slice(1);
  const proceduralHTML = _proceduralCoverHTML({ title, ...opts });
  // Both attributes are escaped for HTML-attribute context; the
  // procedural SVG carries its own internal escaping for title/glyph.
  const restAttr = escapeHtml(JSON.stringify(rest));
  const proceduralAttr = escapeHtml(proceduralHTML);
  return `<img class="library-img-preview" src="${escapeHtml(first)}" alt=""
               loading="lazy" decoding="async" referrerpolicy="no-referrer"
               data-thumb-fallbacks="${restAttr}"
               data-procedural-fallback="${proceduralAttr}"
               onerror="window.__augLibPreviewFallback(this)">`;
}

export { _ensurePreviewHTML as ensurePreviewHTML };

// ── Defaults shared by sources that just hit /api/games/browse ─────

function _browseGamesFetch(source) {
  // Wraps /api/games/browse for any public-catalog source. The backend
  // returns ``{results, source, sort, page, has_more}`` with results
  // shaped like ``GameBrowseResult.to_dict()`` (source, source_id,
  // name, author, tagline, thumbnail_url, embed_url, ...).
  //
  // Pagination contract: this function does NOT advance
  // ``_state.gamesBrowsePage`` -- ``_loadMoreGames`` in library.js
  // owns the page counter. Returning nextPage here would
  // double-increment.
  return async ({ sort, page }) => {
    const url = `/api/games/browse?source=${encodeURIComponent(source)}`
      + `&sort=${encodeURIComponent(sort || 'newest')}`
      + `&page=${encodeURIComponent(page || 1)}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch failed: ${r.status}`);
    const body = await r.json();
    const items = Array.isArray(body.results) ? body.results : [];
    // Heuristic preserved from the original implementation: only offer
    // Load More if the page came back full AND the server signals more
    // pages exist. Short pages mean the next page is likely empty too,
    // so hide the CTA to avoid dangling clicks that return nothing.
    const hasMore = items.length >= 10 && !!body.has_more;
    return { items, hasMore };
  };
}

// ── Public-catalog sources (js13k today; future GitHub builds, etc.)
//    share the existing browse-card renderer; library.js re-exports
//    it via the renderCard hook below.

let _externalCardRenderer = null;
export function setSharedBrowseCardRenderer(fn) {
  // library.js owns _renderBrowseCard; we wire it in once on init so
  // the registry doesn't need to know its internals.
  _externalCardRenderer = fn;
}
function _sharedRenderBrowseCard(item) {
  return _externalCardRenderer ? _externalCardRenderer(item) : '';
}

// ── Curated games (the "marketplace" source) ──────────────────────
//
// Hand-picked free browser games shipped with Augmentum. Each entry
// pins via the same /api/games/pin path js13k uses (source = 'marketplace'
// is allowed in games_routes.py::_KNOWN_SOURCES) — so a curated game
// becomes a normal pinned bookmark in the user's library, plays in
// the in-app sandbox via embed_url, and unpins like any other.
//
// Selection criteria:
//   * free, in-browser playable (no install)
//   * stable canonical URL (github.io or the author's own domain)
//   * ideally MIT/CC-licensed source so this list stays stable
//   * famous enough that "I've heard of this" carries water
//
// Adding a new entry: append below. source_id must be unique within
// this list and start with `curated-` so it never collides with a
// real backend listing id from a future marketplace API.
const _CURATED_WEB_GAMES = [
  // ── Puzzle / brain ──
  {
    source: 'marketplace',
    source_id: 'curated-2048',
    name: '2048',
    author: 'Gabriele Cirulli',
    tagline: 'Slide tiles, combine numbers, reach 2048.',
    embed_url: 'https://gabrielecirulli.github.io/2048/',
    source_url: 'https://github.com/gabrielecirulli/2048',
    thumbnail_url: 'https://cloud.githubusercontent.com/assets/1175750/8614312/280e5dc2-26f1-11e5-9f1f-5891c3ca8b26.png',
  },
  {
    source: 'marketplace',
    source_id: 'curated-hextris',
    name: 'Hextris',
    author: 'Logan Engstrom et al.',
    tagline: 'Tetris in a hexagon — reflex puzzle with sharp curves.',
    embed_url: 'https://hextris.io/',
    source_url: 'https://github.com/Hextris/hextris',
    thumbnail_url: 'https://raw.githubusercontent.com/Hextris/hextris/master/images/twitter-opengraph.png',
  },
  {
    source: 'marketplace',
    source_id: 'curated-puzzlescript',
    name: 'PuzzleScript',
    author: 'Stephen Lavelle (increpare)',
    tagline: 'A puzzle DSL hub — one card unlocks dozens of community-made puzzle games.',
    embed_url: 'https://www.puzzlescript.net/',
    source_url: 'https://github.com/increpare/PuzzleScript',
  },
  {
    source: 'marketplace',
    source_id: 'curated-cubecomposer',
    name: 'Cube Composer',
    author: 'David Peter (sharkdp)',
    tagline: 'Compose pure functions to transform colored stacks. Meditative functional puzzles.',
    embed_url: 'https://david-peter.de/cube-composer/',
    source_url: 'https://github.com/sharkdp/cube-composer',
  },
  // ── Word / trivia ──
  {
    source: 'marketplace',
    source_id: 'curated-hellowordl',
    name: 'hello wordl',
    author: 'Lynn',
    tagline: 'Wordle, but configurable — any word length, unlimited rounds. No paywall.',
    embed_url: 'https://hellowordl.net/',
    source_url: 'https://github.com/lynn/hello-wordl',
  },
  {
    source: 'marketplace',
    source_id: 'curated-wikitrivia',
    name: 'Wikitrivia',
    author: 'Tom J Watson',
    tagline: 'Order historical events on a timeline. Trivia for nerds.',
    embed_url: 'https://wikitrivia.tomjwatson.com/',
    source_url: 'https://github.com/tom-james-watson/wikitrivia',
  },
  // ── Idle / narrative ──
  {
    source: 'marketplace',
    source_id: 'curated-adarkroom',
    name: 'A Dark Room',
    author: 'Doublespeak Games',
    tagline: 'A minimalist text-based survival RPG. Light the fire.',
    embed_url: 'https://adarkroom.doublespeakgames.com/',
    source_url: 'https://github.com/doublespeakgames/adarkroom',
    thumbnail_url: 'https://raw.githubusercontent.com/doublespeakgames/adarkroom/main/img/Logo1.jpg',
  },
  {
    source: 'marketplace',
    source_id: 'curated-paperclips',
    name: 'Universal Paperclips',
    author: 'Frank Lantz',
    tagline: 'An idle game about an AI that makes paperclips. Then everything else.',
    embed_url: 'https://www.decisionproblem.com/paperclips/',
    source_url: 'https://www.decisionproblem.com/paperclips/',
    thumbnail_url: 'https://www.decisionproblem.com/paperclips/title.png',
  },
  // ── Action / 3D ──
  {
    source: 'marketplace',
    source_id: 'curated-hexgl',
    name: 'HexGL',
    author: 'Thibaut Despoulain (BKcore)',
    tagline: 'Futuristic anti-grav racer in WebGL. The twitchy-action pick.',
    embed_url: 'https://hexgl.bkcore.com/play/',
    source_url: 'https://github.com/BKcore/HexGL',
    thumbnail_url: 'https://hexgl.bkcore.com/image.png',
  },
  {
    source: 'marketplace',
    source_id: 'curated-spacehuggers',
    name: 'Space Huggers',
    author: 'Frank Force (KilledByAPixel)',
    tagline: 'js13k-winning roguelike platformer with destructible terrain and 4-player co-op.',
    embed_url: 'https://killedbyapixel.github.io/SpaceHuggers/',
    source_url: 'https://github.com/KilledByAPixel/SpaceHuggers',
    thumbnail_url: 'https://raw.githubusercontent.com/KilledByAPixel/SpaceHuggers/master/screenshot.png',
  },
  // ── Sandbox / experimental ──
  {
    source: 'marketplace',
    source_id: 'curated-sandspiel',
    name: 'Sandspiel',
    author: 'Max Bittker',
    tagline: 'Rust+WASM cellular automata sandbox. Plant seeds, watch chemistry happen.',
    embed_url: 'https://sandspiel.club/',
    source_url: 'https://github.com/MaxBittker/sandspiel',
    thumbnail_url: 'https://raw.githubusercontent.com/MaxBittker/sandspiel/master/Screenshot.png',
  },
  {
    source: 'marketplace',
    source_id: 'curated-patatap',
    name: 'Patatap',
    author: 'Jono Brandel + Lullatone',
    tagline: 'Press any key, get a synesthetic shape and sound. Audiovisual toy.',
    embed_url: 'https://patatap.com/',
    source_url: 'https://github.com/jonobr1/Patatap',
    thumbnail_url: 'https://raw.githubusercontent.com/jonobr1/Patatap/main/images/thumbnail.png',
  },
  // ── Music / rhythm ──
  {
    source: 'marketplace',
    source_id: 'curated-bemuse',
    name: 'Bemuse',
    author: 'Bemusic Project',
    tagline: 'Full BMS-format rhythm game in your browser. WebAudio-tight.',
    embed_url: 'https://bemuse.ninja/',
    source_url: 'https://github.com/bemusic/bemuse',
    thumbnail_url: 'https://raw.githubusercontent.com/bemusic/bemuse/master/website/static/img/screenshots/music-selection.jpg',
  },
];

async function _marketplaceFetch(/* {sort, page} */) {
  // Frontend curated list is the floor. We still try the backend
  // marketplace endpoint so a future server-managed catalog can
  // supplement (or fully replace) the curated set without code
  // changes here. Backend listings are reshaped to the same browse-
  // item schema so they share the renderer + pin path.
  let backendItems = [];
  try {
    const r = await fetch('/api/titles/marketplace/');
    if (r.ok) {
      const body = await r.json();
      const listings = Array.isArray(body.listings) ? body.listings : [];
      backendItems = listings.map((l) => ({
        source: 'marketplace',
        source_id: `backend-${l.id}`,
        name: l.title || 'Untitled',
        author: l.publisher || '',
        tagline: l.tagline || '',
        thumbnail_url: l.thumbnail_url || '',
        embed_url: l.metadata?.embed_url || '',
        source_url: l.metadata?.source_url || '',
      }));
    }
    // 503 (master toggle / backend offline) → just curated, no error
  } catch {
    // Network / parse failure → fall back to curated only. The
    // surface always renders something playable.
  }
  return { items: [..._CURATED_WEB_GAMES, ...backendItems], hasMore: false };
}

// ── Streamed games (AGSP) renderer + launch flow ───────────────────

function _renderStreamedCard(profile) {
  // profile = { id, display_name, description, multiplayer, scriptable, ... }
  const name = escapeHtml(profile.display_name || profile.id);
  const desc = escapeHtml(profile.description || '');
  const tags = [];
  if (profile.multiplayer) tags.push('multiplayer');
  if (profile.scriptable)  tags.push('moddable');
  const tagBlock = tags.length
    ? `<div class="library-card-tags">${tags.map(t => `<span class="library-card-tag">${escapeHtml(t)}</span>`).join('')}</div>`
    : '';
  const preview = _ensurePreviewHTML(
    profile.thumbnail_url,
    profile.display_name || profile.id,
    { source: 'streamed', kind: 'streamed_game' },
  );
  return `
    <div class="library-card is-browse is-streamed"
         data-streamed-id="${escapeHtml(profile.id)}">
      <div class="library-card-preview">${preview}</div>
      <div class="library-card-body">
        <div class="library-card-title">${name}</div>
        <div class="library-card-tagline">${desc}</div>
        ${tagBlock}
      </div>
      <div class="library-card-actions">
        <button class="library-discover-action" data-launch-streamed="${escapeHtml(profile.id)}">
          Launch
        </button>
        <button class="library-discover-action" data-streamed-worlds="${escapeHtml(profile.id)}"
                title="Manage persistent worlds for this profile (saves + whitelist)">
          Worlds
        </button>
      </div>
    </div>`;
}

// ── Worlds management modal ───────────────────────────────────────
// Persistent per-user worlds for game-stream profiles. Each world is a
// named save state (settings + whitelist) that can be launched in-place
// of an ephemeral session. CRUD only — the launch flow that mounts a
// world into a session lives in stream-stage.js when the user picks
// one from the worlds list.

async function _openWorldsModal(profileId) {
  // Single-instance modal — clicking Worlds on another card replaces it.
  let modal = document.getElementById('streamed-worlds-modal');
  if (modal) modal.remove();

  modal = document.createElement('div');
  modal.id = 'streamed-worlds-modal';
  modal.className = 'streamed-worlds-overlay';
  modal.innerHTML = `
    <div class="streamed-worlds-card" role="dialog" aria-labelledby="streamed-worlds-title">
      <header class="streamed-worlds-head">
        <h3 id="streamed-worlds-title">Worlds — ${escapeHtml(profileId)}</h3>
        <button class="streamed-worlds-close" aria-label="Close">✕</button>
      </header>
      <div class="streamed-worlds-body">
        <div id="streamed-worlds-list" class="streamed-worlds-list"></div>
        <div class="streamed-worlds-add">
          <input id="streamed-worlds-name" type="text" placeholder="New world name" />
          <button id="streamed-worlds-create" class="btn btn-sm">Create</button>
        </div>
      </div>
    </div>`;
  document.body.appendChild(modal);

  const close = () => modal.remove();
  modal.querySelector('.streamed-worlds-close').addEventListener('click', close);
  modal.addEventListener('click', (e) => { if (e.target === modal) close(); });

  const list = modal.querySelector('#streamed-worlds-list');
  const refresh = async () => {
    list.innerHTML = '<div style="color:var(--text-muted)">Loading…</div>';
    try {
      const r = await fetch(`/api/game-stream/worlds?profile_id=${encodeURIComponent(profileId)}`, { credentials: 'same-origin' });
      if (!r.ok) { list.innerHTML = `<div>Failed (status ${r.status})</div>`; return; }
      const data = await r.json();
      const worlds = data.worlds || [];
      if (worlds.length === 0) {
        list.innerHTML = '<div style="color:var(--text-muted)">No worlds yet for this profile.</div>';
        return;
      }
      list.innerHTML = worlds.map(w => `
        <div class="streamed-world-row" data-world-id="${escapeHtml(w.id)}">
          <span class="streamed-world-name">${escapeHtml(w.name)}</span>
          <span class="streamed-world-meta">${escapeHtml(w.updated_at || w.created_at || '')}</span>
          <div class="streamed-world-actions">
            <button class="btn btn-sm" data-action="rename" data-id="${escapeHtml(w.id)}" data-name="${escapeHtml(w.name)}">Rename</button>
            <button class="btn btn-sm" data-action="delete" data-id="${escapeHtml(w.id)}">Delete</button>
          </div>
        </div>`).join('');
    } catch (e) {
      list.innerHTML = `<div>Failed: ${escapeHtml(String(e.message || e))}</div>`;
    }
  };

  list.addEventListener('click', async (ev) => {
    const btn = ev.target.closest('button[data-action]');
    if (!btn) return;
    const id = btn.dataset.id;
    if (btn.dataset.action === 'delete') {
      if (!confirm('Delete this world? Saves inside it will be lost.')) return;
      const r = await fetch(`/api/game-stream/worlds/${encodeURIComponent(id)}`, {
        method: 'DELETE', credentials: 'same-origin',
      });
      if (!r.ok) { showToast('Delete failed', 'error'); return; }
      await refresh();
    } else if (btn.dataset.action === 'rename') {
      const next = prompt('Rename to:', btn.dataset.name || '');
      if (!next || !next.trim()) return;
      let r;
      try {
        r = await fetch(`/api/game-stream/worlds/${encodeURIComponent(id)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ name: next.trim() }),
        });
      } catch (err) {
        showToast('Rename failed (network)', 'error');
        return;
      }
      if (!r.ok) { showToast('Rename failed', 'error'); return; }
      await refresh();
    }
  });

  modal.querySelector('#streamed-worlds-create').addEventListener('click', async () => {
    const nameEl = modal.querySelector('#streamed-worlds-name');
    const name = (nameEl?.value || '').trim();
    if (!name) { nameEl?.focus(); return; }
    let r;
    try {
      r = await fetch('/api/game-stream/worlds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ profile_id: profileId, name }),
      });
    } catch (err) {
      showToast('Create failed (network)', 'error');
      return;
    }
    if (!r.ok) { showToast('Create failed', 'error'); return; }
    if (nameEl) nameEl.value = '';
    await refresh();
  });

  await refresh();
}

// Expose globally so the existing library-card click delegation can
// reach the modal opener. Cards are produced by _renderStreamedCard
// above, but click handling happens through library.js's delegated
// event listener — we register here so wiring stays local.
document.addEventListener('click', (ev) => {
  const btn = ev.target.closest('[data-streamed-worlds]');
  if (!btn) return;
  ev.stopPropagation();
  const profileId = btn.dataset.streamedWorlds;
  if (profileId) _openWorldsModal(profileId);
});

// Profiles that are NOT meant to be launched standalone from the
// Streamed marquee — they need per-session context (a ROM blob, a
// system_id, etc.) that only flows through specific code paths.
// The emulator-streamed profile is one of these: it dispatches to
// Dolphin/PCSX2/etc. based on AUGMENTUM_EMULATOR set by the title
// runtime, so a marquee click without that context spawns a
// container that immediately exits with code 64 (no emulator
// chosen). Filtering keeps the surface honest — you can only reach
// the Dolphin runtime by clicking a GameCube/Wii ROM card.
const _STREAMED_MARQUEE_HIDDEN = new Set(['emulator-streamed']);

async function _streamedFetch(/* {sort, page} */) {
  const r = await fetch('/api/game-stream/profiles');
  if (!r.ok) {
    // 503 = master toggle off or backend unavailable. Surface as
    // "no items" so the caller renders the empty-state message
    // rather than an error.
    if (r.status === 503) return { items: [], hasMore: false };
    throw new Error(`profiles fetch failed: ${r.status}`);
  }
  const body = await r.json();
  const profiles = Array.isArray(body.profiles) ? body.profiles : [];
  return {
    items: profiles.filter(p => !_STREAMED_MARQUEE_HIDDEN.has(p.id)),
    hasMore: false,
  };
}

async function _streamedLaunch(profile) {
  // Mount the in-app stream stage. The stage owns its own session
  // lifecycle: it POSTs /api/game-stream/sessions, mounts an iframe
  // pointing at Selkies' viewer on the allocated port, and stops
  // the session on close. Stop button + ESC + window-X all route
  // through the same teardown path so we don't leak containers.
  try {
    const m = await import('./stream-stage.js');
    await m.openStreamStage(profile);
  } catch (e) {
    showToast(
      `Couldn't open stream stage: ${e.message}`,
      'error',
    );
  }
}

// ── Emulator (ROMs) renderer + launch flow ─────────────────────────

// libretro-thumbnails directory names (the public CDN organises art
// by these exact strings). Mapping covers our supported systems;
// unsupported systems fall back to the generic gamepad glyph.
const _LIBRETRO_THUMBS_DIR = {
  nes: 'Nintendo - Nintendo Entertainment System',
  snes: 'Nintendo - Super Nintendo Entertainment System',
  gb: 'Nintendo - Game Boy',
  gbc: 'Nintendo - Game Boy Color',
  gba: 'Nintendo - Game Boy Advance',
  n64: 'Nintendo - Nintendo 64',
  nds: 'Nintendo - Nintendo DS',
  genesis: 'Sega - Mega Drive - Genesis',
  sms: 'Sega - Master System - Mark III',
  gg: 'Sega - Game Gear',
  saturn: 'Sega - Saturn',
  psx: 'Sony - PlayStation',
  atari2600: 'Atari - 2600',
  lynx: 'Atari - Lynx',
  pce: 'NEC - PC Engine - TurboGrafx 16',
  colecovision: 'Coleco - ColecoVision',
  // Disc-based systems were missing entirely, so every GameCube/Wii/PS2
  // upload fell straight through to the glyph no matter how well-named
  // the file was. These directory names are libretro's exact strings.
  gamecube: 'Nintendo - GameCube',
  wii: 'Nintendo - Wii',
  ps2: 'Sony - PlayStation 2',
  psp: 'Sony - PlayStation Portable',
  dreamcast: 'Sega - Dreamcast',
  segacd: 'Sega - Mega-CD - Sega CD',
  n3ds: 'Nintendo - Nintendo 3DS',
  wonderswan: 'Bandai - WonderSwan',
  ngp: 'SNK - Neo Geo Pocket',
};

// libretro-thumbnails stores files under names where every character
// that is illegal in a Windows filename has been replaced by `_`. So
// "Kirby & The Amazing Mirror (USA)" is served as
// "Kirby _ The Amazing Mirror (USA).png". Without this substitution
// every title containing `&` (a large slice of any real collection)
// 404s and shows placeholder art.
function _libretroSanitize(stem) {
  return stem.replace(/[&*/:`<>?|"\\]/g, '_');
}

// Exported so library/cover.js's All-tab thumbnail path uses the exact
// same guesses the Games-tab _renderEmulatorCard does. Without this, a
// ROM with an empty metadata.thumbnail_url would show box-art on one
// tab and procedural art on the other.
//
// Ordered candidate list — callers walk it in order and drop to the
// procedural cover once every entry has 404'd.
export function libretroThumbCandidates(item) {
  return _libretroThumbCandidates(item);
}

function _libretroThumbCandidates(item) {
  // libretro-thumbnails serves PNG box-art via a stable path:
  //   https://thumbnails.libretro.com/<system>/Named_Boxarts/<rom>.png
  // ROM basename matches the no-intro / redump naming the community
  // catalogues use; uploads from those sources hit instantly. For
  // mismatches, the <img onerror> chain drops to the procedural cover.
  const systemId = item.metadata?.system_id;
  if (!systemId) return [];
  const dir = _LIBRETRO_THUMBS_DIR[systemId];
  if (!dir) return [];

  // `original_filename` is a RELATIVE PATH, not a basename — the folder
  // picker reports webkitRelativePath, so it arrives as
  // "gba/Metroid Fusion (USA).gba" or "roms/wii/....rvz". Feeding that
  // straight to encodeURIComponent produced "gba%2FMetroid..." and every
  // single card 404'd. Strip the directory before anything else.
  const raw = String(item.metadata?.original_filename || '');
  const base = raw.split(/[/\\]/).pop() || '';
  const stems = [
    base.replace(/\.[^.]+$/, '').trim(),
    // The stored title is derived from the basename but is sometimes
    // cleaner (or present when the filename isn't), so try it second.
    String(item.title || '').trim(),
  ];

  const seen = new Set();
  const urls = [];
  for (const stem of stems) {
    if (!stem) continue;
    const name = _libretroSanitize(stem);
    if (seen.has(name)) continue;
    seen.add(name);
    urls.push(
      `https://thumbnails.libretro.com/${encodeURIComponent(dir)}/Named_Boxarts/${encodeURIComponent(name)}.png`,
    );
  }
  return urls;
}

function _renderEmulatorCard(item) {
  // Two flavours of "card" share this renderer:
  //   * the synthetic upload affordance (item._action === 'upload')
  //   * actual installed ROM titles
  // Keeping both in the same grid (instead of a separate header
  // action) means the upload affordance is always visible and stops
  // looking lonely once the user has installed a few titles.
  if (item._action === 'upload') {
    return `
      <div class="library-card is-browse is-emulator is-upload"
           data-emulator-action="upload"
           data-emulator-dropzone="1">
        <div class="library-card-preview">
          <span class="library-preview-icon">+</span>
        </div>
        <div class="library-card-body">
          <div class="library-card-title">Add ROMs</div>
          <div class="library-card-tagline">Drop a folder, or click to pick · auto-detects NES, SNES, GB/GBA, N64, PSX, Genesis…</div>
        </div>
        <div class="library-card-actions">
          <button class="library-discover-action" data-emulator-action="upload">
            Pick folder
          </button>
        </div>
      </div>`;
  }
  const name = escapeHtml(item.title || 'Untitled ROM');
  const systemLabel = item.metadata?.system_label
    || item.metadata?.system_id
    || item.metadata?.system
    || '';
  // Try the user's explicit override (cover picker / scraper) first,
  // then the libretro guesses. This is the ONLY card renderer that used
  // to hand-roll its <img> and collapse to a bare gamepad emoji on
  // miss — every other source goes through _ensurePreviewHTML and gets
  // the procedural gradient cover. Routing through it makes a ROM with
  // no box-art look like a library item instead of a broken row.
  const candidates = [
    item.metadata?.thumbnail_url,
    ..._libretroThumbCandidates(item),
  ].filter(Boolean);
  const preview = _ensurePreviewHTML(candidates, item.title || 'Untitled ROM', {
    systemLabel,
    kind: 'emulator_rom',
    source: 'emulator',
  });

  // Stat row: last played + total play time. library_state comes
  // straight off the title manifest (set by title_runs aggregation).
  const ls = item.library_state || {};
  const lastPlayed = _formatRelativeTime(ls.last_played_at);
  const playTime = _formatPlayTime(ls.total_play_time_s || 0);
  const stats = [];
  if (lastPlayed) stats.push(`<span title="Last played">${escapeHtml(lastPlayed)}</span>`);
  if (playTime) stats.push(`<span title="Total play time">${escapeHtml(playTime)}</span>`);
  const statRow = stats.length
    ? `<div class="library-card-stats">${stats.join('<span class="library-card-stat-sep">·</span>')}</div>`
    : '';

  // Clickable badge — surfaces the auto-detected system AND lets the
  // user override it. Mis-detection happens (a PS2 ISO sometimes
  // looks PSX-shaped to the magic-byte rule, an obscurely-named
  // file lands in the wrong bucket); this is the user's escape hatch
  // without re-uploading. Shows current system text; click opens a
  // small picker modal that PATCHes /api/titles/{id}.
  const systemId = item.metadata?.system_id || '';
  const badgeText = systemLabel
    ? String(systemLabel).toUpperCase()
    : 'SET SYSTEM';
  const badgeTitle = systemLabel
    ? `${systemLabel} — click to change`
    : 'Click to set system';
  const systemBadge = `
    <button class="library-card-system-badge"
            data-change-system="${escapeHtml(item.id)}"
            data-current-system="${escapeHtml(systemId)}"
            title="${escapeHtml(badgeTitle)}">
      ${escapeHtml(badgeText)}
    </button>`;

  return `
    <div class="library-card is-browse is-emulator"
         data-emulator-id="${escapeHtml(item.id)}">
      <div class="library-card-preview">
        ${preview}
        ${systemBadge}
        <button class="library-card-cover-edit" data-cover-edit="${escapeHtml(item.id)}"
                title="Change cover art" aria-label="Change cover art">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20h9"/>
            <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
          </svg>
        </button>
        <button class="library-card-remove" data-remove-emulator="${escapeHtml(item.id)}"
                title="Remove from library" aria-label="Remove from library">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            <line x1="10" y1="11" x2="10" y2="17"/>
            <line x1="14" y1="11" x2="14" y2="17"/>
          </svg>
        </button>
      </div>
      <div class="library-card-body">
        <button class="library-card-title library-card-title-rename"
                data-rename-title="${escapeHtml(item.id)}"
                title="Click to rename">${name}</button>
        ${statRow}
      </div>
      <div class="library-card-actions">
        <button class="library-discover-action" data-launch-emulator="${escapeHtml(item.id)}">
          Launch
        </button>
      </div>
    </div>`;
}


// ── System picker (manual classification override) ─────────────────


export async function openSystemPicker(titleId, currentSystemId, opts = {}) {
  // Modal that lists every registered ROM system and PATCHes the
  // title's metadata.system_id when the user picks one. Used to
  // rescue mis-detected ROMs (PS2 ISO classified as PSX, etc.)
  // without re-uploading the file. Catalog comes from
  // /api/titles/_/systems so the list stays in sync with
  // rom_systems.py without a frontend constant.
  if (!titleId) return;
  if (document.querySelector('.system-picker-overlay')) return;

  const overlay = document.createElement('div');
  overlay.className = 'system-picker-overlay bios-vault-overlay';
  overlay.innerHTML = `
    <div class="bios-vault-card system-picker-card" role="dialog" aria-label="Change system">
      <div class="bios-vault-header">
        <div class="bios-vault-title">Change system</div>
        <button class="bios-vault-close" aria-label="Close">×</button>
      </div>
      <div class="bios-vault-intro">
        Pick the right system for this ROM. Auto-detection guesses
        from header magic bytes and file extension; if the dump
        landed in the wrong bucket, override here — the file isn't
        re-uploaded, just re-tagged.
      </div>
      <div class="bios-vault-status-msg" data-pick-msg></div>
      <div class="bios-vault-list" data-pick-list>
        <div class="bios-vault-loading">Loading systems…</div>
      </div>
      <div class="bios-vault-actions">
        <button class="bios-vault-done">Cancel</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  const list = overlay.querySelector('[data-pick-list]');
  const msg = overlay.querySelector('[data-pick-msg]');

  const close = () => {
    if (!overlay.isConnected) return;
    document.body.removeChild(overlay);
  };
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });
  overlay.querySelector('.bios-vault-close').addEventListener('click', close);
  overlay.querySelector('.bios-vault-done').addEventListener('click', close);
  document.addEventListener('keydown', function escHandler(e) {
    if (e.key === 'Escape') {
      close();
      document.removeEventListener('keydown', escHandler);
    }
  });

  const setMsg = (text, kind) => {
    if (!msg) return;
    msg.textContent = text || '';
    msg.className = 'bios-vault-status-msg' + (kind ? ` is-${kind}` : '');
  };

  let systems = [];
  try {
    const r = await fetch('/api/titles/_/systems');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const body = await r.json();
    systems = Array.isArray(body.systems) ? body.systems : [];
  } catch (e) {
    list.innerHTML = `<div class="bios-vault-error">Couldn't load systems: ${escapeHtml(e.message || e)}</div>`;
    return;
  }

  if (!list.isConnected) return;
  if (systems.length === 0) {
    list.innerHTML = `<div class="bios-vault-empty">No systems registered.</div>`;
    return;
  }

  // Sort alphabetically by label so the picker is scannable. Keep
  // current system pinned to top so the user sees where they were
  // before changing.
  systems.sort((a, b) => String(a.label).localeCompare(String(b.label)));

  const renderRow = (s) => {
    const isCurrent = s.id === currentSystemId;
    const exts = (s.extensions || []).slice(0, 4).join(' · ');
    return `
      <button class="system-picker-row${isCurrent ? ' is-current' : ''}"
              data-pick-system="${escapeHtml(s.id)}">
        <div class="system-picker-row-main">
          <span class="system-picker-row-label">${escapeHtml(s.label)}</span>
          ${isCurrent ? '<span class="system-picker-row-badge">current</span>' : ''}
        </div>
        <div class="system-picker-row-meta">
          <code>${escapeHtml(s.id)}</code>
          ${exts ? `<span class="system-picker-row-exts">${escapeHtml(exts)}</span>` : ''}
        </div>
      </button>`;
  };

  const current = systems.find(s => s.id === currentSystemId);
  const others = systems.filter(s => s.id !== currentSystemId);
  const ordered = current ? [current, ...others] : systems;
  list.innerHTML = ordered.map(renderRow).join('');

  list.querySelectorAll('[data-pick-system]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const newSid = btn.dataset.pickSystem;
      if (!newSid || newSid === currentSystemId) {
        close();
        return;
      }
      setMsg(`Updating to ${newSid}…`, 'info');
      try {
        const r = await fetch(`/api/titles/${encodeURIComponent(titleId)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            metadata: { system_id: newSid },
          }),
        });
        if (!r.ok) {
          const txt = await r.text().catch(() => '');
          throw new Error(`HTTP ${r.status} ${txt.slice(0, 120)}`);
        }
        setMsg('Saved.', 'ok');
        if (typeof opts.onChange === 'function') opts.onChange(newSid);
        // Tell the library grid to refresh so the chip + tile reflect
        // the new system without a manual reload.
        window.dispatchEvent(new CustomEvent('library:games-source-refresh', {
          detail: { source: 'emulator' },
        }));
        setTimeout(close, 350);
      } catch (e) {
        setMsg(`Update failed: ${e.message || e}`, 'error');
      }
    });
  });
}


// ── Rename ─────────────────────────────────────────────────────────


export async function openTitleRename(titleId, currentTitle) {
  // ROM filenames are frequently dump-shaped ("Metroid Fusion (USA)
  // (Rev 1) [!]") and the imported title inherits that. The store
  // already supports renaming -- update_metadata promotes a `title`
  // in the patch to the artifact's display_name column -- there was
  // simply no UI in front of it.
  //
  // Renaming also feeds cover lookup: the title is one of the stems
  // _libretro_filename_variants tries, so correcting a mangled name
  // can be what makes the scrape find art on a re-run.
  if (!titleId) return;
  if (document.querySelector('.title-rename-overlay')) return;

  const overlay = document.createElement('div');
  overlay.className = 'title-rename-overlay bios-vault-overlay';
  overlay.innerHTML = `
    <div class="bios-vault-card title-rename-card" role="dialog" aria-label="Rename title">
      <div class="bios-vault-header">
        <div class="bios-vault-title">Rename</div>
        <button class="bios-vault-close" aria-label="Close">×</button>
      </div>
      <div class="bios-vault-intro">
        Renaming changes how this title is displayed and how cover art
        is looked up. The ROM file itself isn't touched.
      </div>
      <div class="bios-vault-status-msg" data-rename-msg></div>
      <input class="title-rename-input" type="text" data-rename-input
             value="${escapeHtml(currentTitle || '')}"
             aria-label="Title">
      <div class="bios-vault-actions">
        <button class="bios-vault-done" data-rename-cancel>Cancel</button>
        <button class="library-discover-action" data-rename-save>Save</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  const input = overlay.querySelector('[data-rename-input]');
  const msg = overlay.querySelector('[data-rename-msg]');

  const close = () => {
    if (!overlay.isConnected) return;
    document.body.removeChild(overlay);
  };
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });
  overlay.querySelector('.bios-vault-close').addEventListener('click', close);
  overlay.querySelector('[data-rename-cancel]').addEventListener('click', close);

  const setMsg = (text, kind) => {
    if (!msg) return;
    msg.textContent = text || '';
    msg.className = 'bios-vault-status-msg' + (kind ? ` is-${kind}` : '');
  };

  const save = async () => {
    const next = String(input.value || '').trim();
    if (!next) {
      setMsg('Title cannot be empty.', 'error');
      return;
    }
    if (next === String(currentTitle || '').trim()) {
      close();
      return;
    }
    setMsg('Saving…', 'info');
    try {
      const r = await fetch(`/api/titles/${encodeURIComponent(titleId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        // `title` inside the metadata patch is what store.update_metadata
        // promotes to the display_name column — see its docstring.
        body: JSON.stringify({ metadata: { title: next } }),
      });
      if (!r.ok) {
        const txt = await r.text().catch(() => '');
        throw new Error(`HTTP ${r.status} ${txt.slice(0, 120)}`);
      }
      setMsg('Saved.', 'ok');
      window.dispatchEvent(new CustomEvent('library:games-source-refresh', {
        detail: { source: 'emulator' },
      }));
      setTimeout(close, 300);
    } catch (e) {
      setMsg(`Rename failed: ${e.message || e}`, 'error');
    }
  };

  overlay.querySelector('[data-rename-save]').addEventListener('click', save);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') save();
    if (e.key === 'Escape') close();
  });
  input.focus();
  input.select();
}


// ── Cover picker ───────────────────────────────────────────────────


export async function openCoverPicker(titleId, currentTitle) {
  // Fetch candidates + open a modal where the user picks one,
  // uploads their own, or clears the override.
  const overlay = document.createElement('div');
  overlay.className = 'cover-picker-overlay';
  overlay.innerHTML = `
    <div class="cover-picker-card">
      <div class="cover-picker-header">
        <div class="cover-picker-title">${escapeHtml(currentTitle || 'Cover art')}</div>
        <button class="cover-picker-close" aria-label="Close">×</button>
      </div>
      <div class="cover-picker-status">Searching for covers...</div>
      <div class="cover-picker-grid"></div>
      <div class="cover-picker-actions">
        <label class="cover-picker-upload">
          Upload custom...
          <input type="file" accept="image/*" hidden>
        </label>
        <button class="cover-picker-clear">Use auto-fetch (libretro)</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  const status = overlay.querySelector('.cover-picker-status');
  const grid = overlay.querySelector('.cover-picker-grid');
  const closeBtn = overlay.querySelector('.cover-picker-close');
  const upload = overlay.querySelector('.cover-picker-upload input');
  const clearBtn = overlay.querySelector('.cover-picker-clear');

  const close = () => { if (overlay.parentNode) overlay.parentNode.removeChild(overlay); };
  closeBtn.addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  const apply = async (url) => {
    status.textContent = 'Saving...';
    try {
      // Two paths:
      //   * data: URL (user upload)   -> PATCH metadata directly
      //   * empty string (clear)      -> PATCH metadata directly
      //   * http(s) URL (candidate)   -> POST /cover-from-url so the
      //     server fetches the bytes and saves as a data: URL. Avoids
      //     CSP img-src whitelisting random web hosts.
      const isHttp = /^https?:\/\//i.test(url);
      let r;
      if (isHttp) {
        r = await fetch(
          `/api/titles/${encodeURIComponent(titleId)}/cover-from-url`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
          },
        );
      } else {
        r = await fetch(`/api/titles/${encodeURIComponent(titleId)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ metadata: { thumbnail_url: url } }),
        });
      }
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${r.status}`);
      }
      showToast('Cover updated.', 'info');
      window.dispatchEvent(new CustomEvent('library:games-source-refresh', {
        detail: { source: 'emulator' },
      }));
      close();
    } catch (e) {
      status.textContent = `Save failed: ${e.message}`;
    }
  };

  upload.addEventListener('change', async () => {
    const file = upload.files?.[0];
    if (!file) return;
    // Encode as a data: URL. Avoids needing a new "user-uploaded
    // covers" blob endpoint just for this. Capped at ~2 MB so the
    // metadata column stays sane (sqlite TEXT max is 1 GB but
    // selecting it pollutes every list query). Offer to resize
    // larger images via a canvas.
    const MAX = 2 * 1024 * 1024;
    let dataUrl;
    if (file.size <= MAX && file.type.startsWith('image/')) {
      dataUrl = await new Promise((res, rej) => {
        const r = new FileReader();
        r.onload = () => res(r.result);
        r.onerror = () => rej(r.error);
        r.readAsDataURL(file);
      });
    } else {
      dataUrl = await _resizeImageToDataUrl(file, 800);
    }
    await apply(dataUrl);
  });

  clearBtn.addEventListener('click', () => apply(''));

  try {
    const r = await fetch(`/api/titles/${encodeURIComponent(titleId)}/cover-candidates`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const body = await r.json();
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    if (!candidates.length) {
      status.textContent = 'No automatic matches found. Upload a custom image below.';
    } else {
      status.textContent = `Click to set (${candidates.length} candidate${candidates.length > 1 ? 's' : ''})`;
      for (const c of candidates) {
        const tile = document.createElement('button');
        tile.className = 'cover-picker-tile';
        tile.title = c.label || c.source || '';
        // Route the preview through our same-origin proxy so the
        // browser doesn't need img-src whitelist for arbitrary hosts.
        // libretro URLs go direct (already in img-src allowlist) so
        // they hit the libretro CDN's caching headers; everything
        // else goes through /_/cover-proxy.
        const isLibretro = c.url.includes('thumbnails.libretro.com');
        const previewSrc = isLibretro
          ? c.url
          : `/api/titles/_/cover-proxy?url=${encodeURIComponent(c.url)}`;
        tile.innerHTML = `
          <img src="${escapeHtml(previewSrc)}" alt="" loading="lazy" decoding="async"
               referrerpolicy="no-referrer"
               onerror="this.parentNode.style.display='none'" />
          <span class="cover-picker-tile-label">${escapeHtml(c.label || c.source || '')}</span>
        `;
        tile.addEventListener('click', () => apply(c.url));
        grid.appendChild(tile);
      }
    }
  } catch (e) {
    status.textContent = `Couldn't load candidates: ${e.message}`;
  }
}


async function _resizeImageToDataUrl(file, maxSide) {
  // Downscale large user uploads to keep the metadata payload tight.
  // Aspect-preserving scale to maxSide px on the longest edge,
  // re-encode as JPEG @ 0.85 quality.
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * scale);
  const h = Math.round(bmp.height * scale);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(bmp, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.85);
}


// ── Formatting helpers ─────────────────────────────────────────────


function _formatRelativeTime(iso) {
  // ISO-8601 → "5 min ago" / "yesterday" / "3 days ago" / "Mar 14".
  // Returns empty string if the input is missing or unparseable so
  // the caller can omit the row entirely.
  if (!iso) return '';
  const ts = Date.parse(iso);
  if (!Number.isFinite(ts)) return '';
  const deltaSec = Math.floor((Date.now() - ts) / 1000);
  if (deltaSec < 60) return 'just now';
  if (deltaSec < 3600) return `${Math.floor(deltaSec / 60)}m ago`;
  if (deltaSec < 86400) return `${Math.floor(deltaSec / 3600)}h ago`;
  if (deltaSec < 86400 * 2) return 'yesterday';
  if (deltaSec < 86400 * 14) return `${Math.floor(deltaSec / 86400)}d ago`;
  // Older than 2 weeks: short month-day (omit year for current year).
  const d = new Date(ts);
  const sameYear = d.getFullYear() === new Date().getFullYear();
  return d.toLocaleDateString(undefined, sameYear
    ? { month: 'short', day: 'numeric' }
    : { month: 'short', day: 'numeric', year: 'numeric' });
}


function _formatPlayTime(seconds) {
  // "Played 47s" / "12m" / "3h 14m". Total time across all runs.
  const s = Math.max(0, Math.floor(Number(seconds) || 0));
  if (s === 0) return '';
  if (s < 60) return `Played ${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `Played ${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `Played ${h}h ${rm}m` : `Played ${h}h`;
}

// ── Emulator filter/sort + BIOS status ───────────────────────────────
//
// Filters apply client-side over the already-fetched ROM list — toggling
// a chip never refetches. BIOS-ready check needs the per-system status
// roll-up, which library.js fetches lazily when the emulator source
// becomes active and refetches after each install/delete.


// Hardcoded so the BIOS-ready filter + launch-path pre-flight work
// before /status loads. This must match the server-side definition:
// "system_id has at least one BiosFile with optional=False" in
// augmentum/titles/bios_catalog.py. Verify by running
//   python -c "from augmentum.titles.bios_catalog import *;
//     print(sorted({s for s in systems_with_bios()
//       if any(not f.optional for f in all_for_system(s))}))"
// A drift here means systems either get over-filtered (chip filter
// hides games unnecessarily) or skip the pre-flight prompt at launch.
const _BIOS_REQUIRED_SYSTEMS = new Set([
  '3do', '3ds', 'amiga', 'atari5200', 'colecovision', 'dc',
  'dreamcast', 'intellivision', 'lynx', 'neogeo', 'pcecd', 'pcfx',
  'ps2', 'ps3', 'psx', 'saturn', 'segacd', 'switch', 'wiiu',
]);


async function _fetchBiosStatus() {
  // Returns {systemId: {system_id, system_label, ready, ...}} for every
  // system the BIOS catalog knows about. Empty object on error so the
  // caller can treat "BIOS panel unavailable" as "no BIOS data" rather
  // than null-checking everywhere.
  try {
    const r = await fetch('/api/titles/bios/status');
    if (!r.ok) return {};
    const body = await r.json();
    const out = {};
    for (const s of (body.systems || [])) {
      if (s && s.system_id) out[s.system_id] = s;
    }
    return out;
  } catch (_) {
    return {};
  }
}


function _isBiosReady(systemId, biosStatus) {
  if (!systemId) return true;
  if (!_BIOS_REQUIRED_SYSTEMS.has(systemId)) return true;
  const s = (biosStatus || {})[systemId];
  return !!(s && s.ready);
}


function _emuSystemBuckets(items) {
  // {systemId: {label, count}} aggregated from the current ROM list
  // (skip the synthetic upload card). Sorted by count desc so the
  // user's biggest libraries land left in the chip row; ties break on
  // label so chip ordering is stable across renders.
  const map = new Map();
  for (const it of items) {
    if (it._action) continue;
    const sid = it.metadata?.system_id;
    if (!sid) continue;
    const label = it.metadata?.system_label
      || String(sid).toUpperCase().replace(/_/g, ' ');
    const cur = map.get(sid) || { systemId: sid, label, count: 0 };
    cur.count += 1;
    map.set(sid, cur);
  }
  return Array.from(map.values()).sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    return a.label.localeCompare(b.label);
  });
}


function _renderEmulatorFilters(items, filterState, biosStatus) {
  // Hide the filter bar entirely until the user has at least one ROM —
  // an empty chip row + sort dropdown over a single "Add ROMs" tile is
  // visual noise without payoff.
  const buckets = _emuSystemBuckets(items);
  if (buckets.length === 0) return '';

  const totalCount = buckets.reduce((acc, b) => acc + b.count, 0);
  const activeSid = filterState.systemId || 'all';
  const biosReadyOnly = !!filterState.biosReadyOnly;
  const sort = filterState.sort || 'recent';

  const chip = (sid, label, count) => `
    <button class="lib-emu-chip${activeSid === sid ? ' is-active' : ''}"
            data-emu-system="${escapeHtml(sid)}">
      <span class="lib-emu-chip-label">${escapeHtml(label)}</span>
      <span class="lib-emu-chip-count">${count}</span>
    </button>`;

  const chips = [
    chip('all', 'All', totalCount),
    ...buckets.map(b => chip(b.systemId, b.label, b.count)),
  ].join('');

  // BIOS card: only render when the user actually has at least one
  // ROM whose system requires a BIOS. A user with only NES/SNES/GBA
  // games has nothing to act on; surfacing "0/5 systems ready" against
  // an abstract catalog feels like an unresolvable to-do. We scope the
  // count to systems that appear in the user's chip row, so 2 of 3
  // means "of the 3 BIOS-required systems you have games for, 2 are
  // ready to play". Proactive BIOS install (before importing any
  // ROMs) is still reachable via the inline launch-time prompt.
  let biosCard = '';
  if (biosStatus) {
    const userBiosSystems = new Set(
      buckets
        .map(b => b.systemId)
        .filter(sid => _BIOS_REQUIRED_SYSTEMS.has(sid)),
    );
    if (userBiosSystems.size > 0) {
      const total = userBiosSystems.size;
      const ready = Array.from(userBiosSystems).filter(
        sid => biosStatus[sid] && biosStatus[sid].ready,
      ).length;
      const missing = total - ready;
      const isReady = missing === 0;

      if (isReady) {
        // Quiet "✓ all set" treatment — no toggle (nothing to filter
        // out), single-line, minimal visual weight.
        biosCard = `
          <div class="lib-emu-bios-card is-ready">
            <div class="lib-emu-bios-card-inner">
              <span class="lib-emu-bios-glyph is-check">✓</span>
              <div class="lib-emu-bios-text">
                <div class="lib-emu-bios-head">Every system is set up.</div>
                <div class="lib-emu-bios-sub">All your BIOS-gated games are ready to play.</div>
              </div>
            </div>
            <div class="lib-emu-bios-actions">
              <button class="lib-emu-bios-cta is-quiet" data-emu-bios-manage="1">Manage</button>
            </div>
          </div>`;
      } else {
        // Pending state: count callout + conversational prompt + the
        // toggle (now lives beside its CTA, not lost in the sort row).
        const head = ready === 0
          ? `Get your retro library ready.`
          : `${ready} of ${total} ${total === 1 ? 'system' : 'systems'} set up.`;
        const sub = ready === 0
          ? `${total} ${total === 1 ? 'system needs' : 'systems need'} a BIOS dump from your console to play.`
          : `${missing} still ${missing === 1 ? 'needs' : 'need'} a BIOS dump from your console to play.`;
        biosCard = `
          <div class="lib-emu-bios-card is-pending">
            <div class="lib-emu-bios-card-inner">
              <span class="lib-emu-bios-glyph is-count">
                <span class="lib-emu-bios-glyph-num">${ready}</span><span class="lib-emu-bios-glyph-slash">/</span><span class="lib-emu-bios-glyph-tot">${total}</span>
              </span>
              <div class="lib-emu-bios-text">
                <div class="lib-emu-bios-head">${escapeHtml(head)}</div>
                <div class="lib-emu-bios-sub">${escapeHtml(sub)}</div>
              </div>
            </div>
            <div class="lib-emu-bios-actions">
              <label class="lib-emu-bios-toggle">
                <input type="checkbox" data-emu-bios-ready ${biosReadyOnly ? 'checked' : ''} />
                <span>Hide games that need one</span>
              </label>
              <button class="lib-emu-bios-cta" data-emu-bios-manage="1">Set up BIOS</button>
            </div>
          </div>`;
      }
    }
  }

  return `
    <div class="lib-emu-filters" data-emu-filters="1">
      <div class="lib-emu-chip-row">${chips}</div>
      <div class="lib-emu-sort-row">
        <label class="lib-emu-filter-sort">
          <span>Sort:</span>
          <select data-emu-sort>
            <option value="recent" ${sort === 'recent' ? 'selected' : ''}>Recently added</option>
            <option value="played" ${sort === 'played' ? 'selected' : ''}>Recently played</option>
            <option value="title"  ${sort === 'title'  ? 'selected' : ''}>Title</option>
            <option value="system" ${sort === 'system' ? 'selected' : ''}>System</option>
          </select>
        </label>
      </div>
      ${biosCard}
    </div>`;
}


function _applyEmulatorFilters(items, filterState, biosStatus) {
  // Pure: always returns a new array. The synthetic upload card stays
  // pinned at index 0 so the affordance is reachable even when a
  // system filter would otherwise empty the grid.
  if (!Array.isArray(items)) return items;
  const upload = items.find(i => i._action === 'upload');
  const roms = items.filter(i => !i._action);

  const sid = filterState.systemId || 'all';
  const biosOnly = !!filterState.biosReadyOnly;
  const filtered = roms.filter(it => {
    const itemSid = it.metadata?.system_id;
    if (sid !== 'all' && itemSid !== sid) return false;
    if (biosOnly && !_isBiosReady(itemSid, biosStatus)) return false;
    return true;
  });

  const sort = filterState.sort || 'recent';
  const titleOf = a => String(a.title || a.metadata?.original_filename || '');
  const sysOf = a => String(a.metadata?.system_id || 'zzz');
  filtered.sort((a, b) => {
    if (sort === 'title') return titleOf(a).localeCompare(titleOf(b));
    if (sort === 'system') {
      return sysOf(a).localeCompare(sysOf(b))
        || titleOf(a).localeCompare(titleOf(b));
    }
    if (sort === 'played') {
      const ta = Date.parse(a.library_state?.last_played_at || 0) || 0;
      const tb = Date.parse(b.library_state?.last_played_at || 0) || 0;
      return tb - ta;
    }
    // 'recent': created_at desc, falling back to import metadata.
    const ta = Date.parse(a.created_at || a.metadata?.imported_at || 0) || 0;
    const tb = Date.parse(b.created_at || b.metadata?.imported_at || 0) || 0;
    return tb - ta;
  });

  return upload ? [upload, ...filtered] : filtered;
}


// Re-exports — library.js fetches BIOS status when the emulator source
// becomes active and after install/delete events.
export const fetchEmulatorBiosStatus = _fetchBiosStatus;
export const BIOS_REQUIRED_SYSTEMS = _BIOS_REQUIRED_SYSTEMS;


// ── BIOS Vault modal ─────────────────────────────────────────────────


function _formatBytes(n) {
  const v = Number(n) || 0;
  if (v < 1024) return `${v} B`;
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(0)} KB`;
  return `${(v / 1024 / 1024).toFixed(1)} MB`;
}


async function _fetchBiosCatalogAndStatus() {
  // Parallel fetch: full catalog (file expectations per system) + per-
  // system status (which files are installed). Merged by canonical
  // filename so the renderer can show the expected list with installed
  // markers next to it. Catalog is the source of truth for what
  // *should* exist; status is the source of truth for what *does*.
  const [catalogR, statusR] = await Promise.all([
    fetch('/api/titles/bios/_/catalog'),
    fetch('/api/titles/bios/status'),
  ]);
  const catalog = catalogR.ok ? await catalogR.json() : { systems: [] };
  const statusRoll = statusR.ok ? await statusR.json() : { systems: [] };
  const statusBySys = {};
  for (const s of (statusRoll.systems || [])) statusBySys[s.system_id] = s;

  // For each system in the catalog, fetch the per-file detail (status
  // endpoint with ?system_id) so we can show "scph7502.bin → installed
  // as user-uploaded-name.bin". One roundtrip per system; small N (~6)
  // and parallel-fetched so latency is bounded by the slowest one.
  const systems = catalog.systems || [];
  const detailFetches = systems.map(s =>
    fetch(`/api/titles/bios/status?system_id=${encodeURIComponent(s.system_id)}`)
      .then(r => (r.ok ? r.json() : { entries: [] }))
      .catch(() => ({ entries: [] })),
  );
  const details = await Promise.all(detailFetches);
  const detailBySys = {};
  for (let i = 0; i < systems.length; i++) {
    detailBySys[systems[i].system_id] = details[i];
  }

  return systems.map(s => ({
    system_id: s.system_id,
    system_label: s.system_label,
    // 'bundled' | 'streaming_required' | 'experimental' | 'unsupported'
    // — drives the "playable now / coming later / not yet" badge.
    core_status: s.core_status || 'bundled',
    playable: s.playable !== false && (s.core_status || 'bundled') === 'bundled',
    catalog_files: s.files || [],
    detail_entries: detailBySys[s.system_id]?.entries || [],
    summary: statusBySys[s.system_id] || null,
  }));
}


function _biosPlayabilityBadge(coreStatus) {
  // Honest "can I actually play this system?" hint next to the row
  // label. Without it a "0/1 BIOS installed" row for PS3 / Switch /
  // Dreamcast reads as broken when really the runtime just isn't here
  // yet (or ever, for systems we only catalogue so the importer can
  // recognise their dumps).
  switch (coreStatus) {
    case 'bundled':
      return '';  // playable now — no badge needed
    case 'streaming_required':
      return `<span class="bios-vault-playbadge is-soon"
                    title="Needs the streamed emulator runtime, which isn't wired up yet">streamed runtime — soon</span>`;
    case 'experimental':
      return `<span class="bios-vault-playbadge is-soon"
                    title="The core exists upstream but isn't bundled in our build yet">core not bundled yet</span>`;
    default:  // 'unsupported' / anything unexpected
      return `<span class="bios-vault-playbadge is-na"
                    title="Catalogued so we can recognise its BIOS dumps; no in-app runtime for it">no runtime yet</span>`;
  }
}


// Recursively collect every File from a drop, walking into folders.
//
// `DataTransfer.files` does NOT recurse: drop a folder and you get a
// single zero-byte entry for the directory itself, which is why
// dropping a BIOS folder used to do nothing at all. The Entries API
// (`webkitGetAsEntry`) is the only way to read a dropped directory,
// and the DataTransferItemList is neutered as soon as the handler
// yields, so entries must be snapshotted synchronously before any
// await -- that's why the caller grabs them in the drop handler.
//
// Falls back to the flat file list where the API is missing.
async function _collectDroppedFiles(entries, fallbackFiles) {
  if (!entries || !entries.length) return Array.from(fallbackFiles || []);

  const out = [];
  const readEntry = (entry, path) => new Promise((resolve) => {
    if (entry.isFile) {
      entry.file(
        (f) => { out.push(f); resolve(); },
        () => resolve(),   // unreadable file: skip, don't abort the drop
      );
      return;
    }
    if (!entry.isDirectory) { resolve(); return; }
    // readEntries() returns at most 100 entries per call and must be
    // drained in a loop until it yields an empty batch -- a BIOS pack
    // with 150 files would otherwise import only the first 100.
    const reader = entry.createReader();
    const batch = () => reader.readEntries(async (results) => {
      if (!results.length) { resolve(); return; }
      const sub = `${path ? `${path}/` : ''}${entry.name}`;
      for (const r of results) await readEntry(r, sub);   // eslint-disable-line no-await-in-loop
      batch();
    }, () => resolve());
    batch();
  });

  for (const e of entries) await readEntry(e, '');   // eslint-disable-line no-await-in-loop
  return out;
}


function _verifyBadge(status, matchedBy) {
  // Honest labelling, in the RetroArch/EmuDeck spirit: we say exactly
  // how much we know. 'verified' means a cryptographic digest matched
  // a known dump; 'named' means the filename and size line up but the
  // bytes were never checked; 'unverified' means the user vouched for
  // it and we stored it on that basis.
  if (status === 'verified') {
    return `<span class="bios-vault-verify is-verified"
                  title="Hash matches a known good dump (${escapeHtml(matchedBy || 'hash')})">verified</span>`;
  }
  if (status === 'named') {
    return `<span class="bios-vault-verify is-named"
                  title="Filename and size match a known dump, but the contents weren't hash-checked">name match</span>`;
  }
  return `<span class="bios-vault-verify is-unverified"
                title="Stored at your request — we don't recognise this file. It may still work.">unverified</span>`;
}


function _renderBiosVaultRow(sys) {
  // Merge catalog file list with per-file installed status. Each row
  // shows: system header (label, file count, status), expanded file
  // list, and a per-system drop zone that scopes the upload to this
  // system (server still classifies by hash/name; the scope is a hint
  // for ambiguous files).
  const detailByName = {};
  for (const e of sys.detail_entries) {
    detailByName[e.canonical_filename] = e;
  }
  const required = sys.catalog_files.filter(f => !f.optional);
  const requiredPresent = required.filter(
    f => detailByName[f.canonical_filename]?.present,
  ).length;
  const allReady = requiredPresent === required.length && required.length > 0;
  const anyOptional = sys.catalog_files.some(f => f.optional);
  const headerStatus = allReady
    ? `<span class="bios-vault-status is-ready">✓</span>`
    : `<span class="bios-vault-status is-missing">✗</span>`;

  const fileRows = sys.catalog_files.map(f => {
    const e = detailByName[f.canonical_filename];
    const present = !!e?.present;
    const installedName = e?.installed_filename || '';
    const matchedBy = e?.matched_by || '';
    const sizeStr = _formatBytes(f.size_bytes);
    const badge = f.optional
      ? `<span class="bios-vault-badge bios-vault-badge--optional">optional</span>`
      : '';
    let statusCell;
    if (present) {
      const ann = matchedBy && installedName !== f.canonical_filename
        ? ` <span class="bios-vault-installed-as">(uploaded as ${escapeHtml(installedName)})</span>`
        : '';
      statusCell = `
        <span class="bios-vault-file-status is-installed">
          ✓ installed${ann}
        </span>
        ${_verifyBadge(e?.verify_status, matchedBy)}
        <button class="bios-vault-file-action"
                data-bios-delete="1"
                data-bios-system="${escapeHtml(sys.system_id)}"
                data-bios-name="${escapeHtml(f.canonical_filename)}"
                title="Remove this BIOS file">×</button>`;
    } else {
      statusCell = `<span class="bios-vault-file-status is-missing">missing</span>`;
    }
    const desc = f.description
      ? `<div class="bios-vault-file-desc">${escapeHtml(f.description)}</div>`
      : '';
    return `
      <div class="bios-vault-file ${present ? 'is-installed' : 'is-missing'}">
        <div class="bios-vault-file-name">
          <code>${escapeHtml(f.canonical_filename)}</code>
          <span class="bios-vault-file-size">${sizeStr}</span>
          ${badge}
        </div>
        ${desc}
        <div class="bios-vault-file-status-row">${statusCell}</div>
      </div>`;
  }).join('');

  // Files the user stored that occupy no catalog slot. Rendering them
  // is what makes store-first legible: without this the file is on
  // disk and served to the emulator but invisible here, which reads
  // exactly like the upload having failed.
  const extras = sys.detail_entries.filter(e => e.is_extra);
  const extraRows = extras.map(e => `
      <div class="bios-vault-file is-installed is-extra">
        <div class="bios-vault-file-name">
          <code>${escapeHtml(e.canonical_filename)}</code>
          <span class="bios-vault-file-size">${_formatBytes(e.size_bytes)}</span>
          <span class="bios-vault-badge bios-vault-badge--extra">extra</span>
        </div>
        <div class="bios-vault-file-desc">${escapeHtml(e.description || '')}</div>
        <div class="bios-vault-file-status-row">
          <span class="bios-vault-file-status is-installed">✓ stored</span>
          ${_verifyBadge(e.verify_status, e.matched_by)}
          <button class="bios-vault-file-action"
                  data-bios-delete="1"
                  data-bios-system="${escapeHtml(sys.system_id)}"
                  data-bios-name="${escapeHtml(e.canonical_filename)}"
                  title="Remove this file">×</button>
        </div>
      </div>`).join('');

  return `
    <div class="bios-vault-row${allReady ? ' is-ready' : ' is-missing'}"
         data-bios-row-system="${escapeHtml(sys.system_id)}">
      <div class="bios-vault-row-head">
        ${headerStatus}
        <div class="bios-vault-row-label">
          <strong>${escapeHtml(sys.system_label)}</strong>
          ${_biosPlayabilityBadge(sys.core_status)}
          <span class="bios-vault-row-summary">
            ${required.length
              ? `${requiredPresent}/${required.length} required installed`
              : 'no BIOS required'}${extras.length ? ` · ${extras.length} extra stored` : ''}${anyOptional ? ' · optional files supported' : ''}
          </span>
        </div>
        <div class="bios-vault-row-actions">
          <button class="bios-vault-row-upload"
                  data-bios-upload="files"
                  data-bios-system="${escapeHtml(sys.system_id)}">
            Add files
          </button>
          <button class="bios-vault-row-upload is-quiet"
                  data-bios-upload="folder"
                  data-bios-system="${escapeHtml(sys.system_id)}"
                  title="Pick a folder — subfolders are included">
            Add folder
          </button>
        </div>
      </div>
      <div class="bios-vault-files">${fileRows}${extraRows}</div>
    </div>`;
}


function _offerManualBiosInstall(sys, file, unknownEntry) {
  // Last-resort override: the bulk-import classifier didn't recognise
  // `file`, but the user dropped it on `sys`'s row, so they think it's
  // a BIOS for that system. Show a slot picker and force-install via
  // POST /api/titles/bios/{system}/{slot}. Resolves true if installed.
  return new Promise((resolve) => {
    const slots = Array.isArray(sys?.catalog_files) ? sys.catalog_files : [];
    if (!slots.length || document.querySelector('.bios-manual-overlay')) {
      resolve(false);
      return;
    }
    const detailByName = {};
    for (const e of (sys.detail_entries || [])) detailByName[e.canonical_filename] = e;
    // Missing-required slots first (most likely target), then the rest.
    const sorted = [...slots].sort((a, b) => {
      const ap = detailByName[a.canonical_filename]?.present ? 1 : 0;
      const bp = detailByName[b.canonical_filename]?.present ? 1 : 0;
      if (ap !== bp) return ap - bp;
      return (a.optional ? 1 : 0) - (b.optional ? 1 : 0);
    });
    const opts = sorted.map(f => {
      const present = !!detailByName[f.canonical_filename]?.present;
      const tail = `${f.optional ? 'optional' : 'required'}${present ? ' · already installed' : ''}`;
      const desc = f.description ? `${f.description} — ${tail}` : tail;
      return `<option value="${escapeHtml(f.canonical_filename)}">`
        + `${escapeHtml(f.canonical_filename)} — ${escapeHtml(desc)} (${_formatBytes(f.size_bytes)})`
        + `</option>`;
    }).join('');

    const overlay = document.createElement('div');
    overlay.className = 'bios-vault-overlay bios-manual-overlay';
    overlay.innerHTML = `
      <div class="bios-vault-card system-picker-card" role="dialog" aria-label="Install file as BIOS">
        <div class="bios-vault-header">
          <div class="bios-vault-title">Install as BIOS?</div>
          <button class="bios-vault-close" aria-label="Close">×</button>
        </div>
        <div class="bios-vault-intro">
          We didn't recognise <code>${escapeHtml(file.name)}</code>
          (${_formatBytes(file.size)})${unknownEntry?.reason ? ` — ${escapeHtml(unknownEntry.reason)}` : ''}.
          If you know it's a <strong>${escapeHtml(sys.system_label)}</strong> BIOS, pick the slot it
          belongs in. The bytes are stored exactly as uploaded — a wrong pick just won't boot, it
          won't break anything.
        </div>
        <div class="bios-manual-field">
          <label>BIOS slot
            <select data-manual-slot>${opts}</select>
          </label>
        </div>
        <div class="bios-vault-status-msg" data-manual-msg></div>
        <div class="bios-vault-actions">
          <button class="bios-vault-done is-secondary" data-manual-skip>Skip</button>
          <button class="bios-vault-done" data-manual-install>Install</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);

    let done = false;
    const finish = (val) => {
      if (done) return;
      done = true;
      if (overlay.isConnected) document.body.removeChild(overlay);
      resolve(val);
    };
    overlay.addEventListener('click', (e) => { if (e.target === overlay) finish(false); });
    overlay.querySelector('.bios-vault-close').addEventListener('click', () => finish(false));
    overlay.querySelector('[data-manual-skip]').addEventListener('click', () => finish(false));
    const msg = overlay.querySelector('[data-manual-msg]');
    overlay.querySelector('[data-manual-install]').addEventListener('click', async () => {
      const slot = overlay.querySelector('[data-manual-slot]')?.value || '';
      if (!slot) return;
      if (msg) msg.textContent = 'Installing…';
      try {
        const fd = new FormData();
        fd.append('file', file, file.name);
        const r = await fetch(
          `/api/titles/bios/${encodeURIComponent(sys.system_id)}/${encodeURIComponent(slot)}`,
          { method: 'POST', body: fd },
        );
        if (!r.ok) {
          const txt = await r.text().catch(() => '');
          throw new Error(`${r.status} ${txt.slice(0, 200)}`);
        }
        finish(true);
      } catch (e) {
        if (msg) msg.textContent = `Install failed: ${e.message || e}`;
      }
    });
  });
}


export function openBiosVault({ onChange, focusSystemId } = {}) {
  // Single-instance modal — guard against double-open.
  if (document.querySelector('.bios-vault-overlay')) return;

  const overlay = document.createElement('div');
  overlay.className = 'bios-vault-overlay';
  overlay.innerHTML = `
    <div class="bios-vault-card" role="dialog" aria-label="BIOS Vault">
      <div class="bios-vault-header">
        <div class="bios-vault-title">BIOS Vault</div>
        <button class="bios-vault-close" aria-label="Close">×</button>
      </div>
      <div class="bios-vault-intro">
        Some systems can't run without a BIOS dump from your own console.
        Drop files or a whole folder on a row — we identify each one by
        hash against the libretro BIOS database and file it in the right
        slot. Anything we don't recognise is still stored on the row you
        dropped it on and marked <em>unverified</em>, so an odd regional
        dump isn't a dead end. BIOS files stay private to your account.
      </div>
      <div class="bios-vault-status-msg" data-bios-statusmsg></div>
      <div class="bios-vault-list" data-bios-list>
        <div class="bios-vault-loading">Loading BIOS catalog…</div>
      </div>
      <div class="bios-vault-actions">
        <button class="bios-vault-done">Done</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  const list = overlay.querySelector('[data-bios-list]');
  const statusMsg = overlay.querySelector('[data-bios-statusmsg]');

  const close = () => {
    if (!overlay.isConnected) return;
    document.body.removeChild(overlay);
  };

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });
  overlay.querySelector('.bios-vault-close').addEventListener('click', close);
  overlay.querySelector('.bios-vault-done').addEventListener('click', close);
  document.addEventListener('keydown', function escHandler(e) {
    if (e.key === 'Escape') {
      close();
      document.removeEventListener('keydown', escHandler);
    }
  });

  let _systems = [];
  let _refreshing = false;
  let _didFocus = false;

  const setStatus = (text, kind) => {
    if (!statusMsg) return;
    statusMsg.textContent = text || '';
    statusMsg.className = 'bios-vault-status-msg' + (kind ? ` is-${kind}` : '');
  };

  const refresh = async () => {
    if (_refreshing) return;
    _refreshing = true;
    try {
      const allSystems = await _fetchBiosCatalogAndStatus();
      // Every system with a BIOS slot gets a row, required or not.
      //
      // This used to filter to systems with at least one REQUIRED
      // file, on the reasoning that an all-optional system (GBA,
      // GameCube, Wii, NDS) is "nothing to do". But the vault is the
      // only place to install a BIOS, so filtering the row also
      // removed the only upload target those systems had — a user
      // holding a real gba_bios.bin had nowhere to put it. Optional
      // means "boots without it", not "can't have one".
      //
      // Systems that need nothing sort last and render collapsed, so
      // the panel still leads with actual work.
      _systems = (allSystems || []).filter(
        s => Array.isArray(s.catalog_files) && s.catalog_files.length > 0,
      );
      _systems.sort((a, b) => {
        const aReq = a.catalog_files.some(f => !f.optional) ? 0 : 1;
        const bReq = b.catalog_files.some(f => !f.optional) ? 0 : 1;
        if (aReq !== bReq) return aReq - bReq;
        return (a.system_label || '').localeCompare(b.system_label || '');
      });
      if (!list.isConnected) return;
      if (_systems.length === 0) {
        list.innerHTML = `<div class="bios-vault-empty">
          No systems with BIOS support are registered.
        </div>`;
        return;
      }
      list.innerHTML = _systems.map(_renderBiosVaultRow).join('');
      _wireRows();

      // If the caller asked us to focus a specific system (launch-
      // path opened the vault because that system is missing BIOS),
      // scroll its row into view + briefly highlight it. Only fires
      // on the first refresh — subsequent install/delete refreshes
      // shouldn't re-snap the user's scroll.
      if (focusSystemId && !_didFocus) {
        _didFocus = true;
        const row = list.querySelector(
          `[data-bios-row-system="${CSS.escape(focusSystemId)}"]`,
        );
        if (row) {
          row.scrollIntoView({ behavior: 'smooth', block: 'center' });
          row.classList.add('is-focus');
          setTimeout(() => row.classList.remove('is-focus'), 2400);
        }
      }
    } catch (e) {
      list.innerHTML = `<div class="bios-vault-error">Couldn't load BIOS catalog: ${escapeHtml(e.message || String(e))}</div>`;
    } finally {
      _refreshing = false;
    }
  };

  const uploadFiles = async (files, systemHint) => {
    if (!files || !files.length) return;
    setStatus(`Uploading ${files.length} file${files.length === 1 ? '' : 's'}…`, 'info');
    try {
      // Reuse the bulk-import path — server auto-classifies BIOS files
      // by SHA1 against the catalog, then falls back to (filename,
      // size) match. The classifier is deterministic for catalogued
      // dumps, so a file dropped on the PSX row lands in PSX whether
      // or not we forward the hint. systemHint is sent as a query
      // param for forward-compat (server can opt-in later) and
      // logging; the current backend ignores unknown query params.
      const fd = new FormData();
      for (const f of files) fd.append('files', f, f.name);
      const url = systemHint
        ? `/api/titles/bulk-import?system_id=${encodeURIComponent(systemHint)}`
        : '/api/titles/bulk-import';
      const r = await fetch(url, { method: 'POST', body: fd });
      if (!r.ok) {
        const txt = await r.text().catch(() => '');
        throw new Error(`upload failed: ${r.status} ${txt.slice(0, 200)}`);
      }
      const body = await r.json().catch(() => ({}));
      const summary = body.summary || {};
      const bios = Array.isArray(summary.bios) ? summary.bios : [];
      const dups = (summary.duplicates || []).length;
      const errs = Array.isArray(summary.errors) ? summary.errors : [];
      const junk = (summary.junk || []).length;
      const unknowns = Array.isArray(summary.unknown) ? summary.unknown : [];
      const verified = bios.filter(b => b.verify_status === 'verified').length;
      const unverified = bios.length - verified;

      // Report what happened to EVERY file. The old summary counted
      // installs and "not recognised" but stayed silent about junk-
      // filtered files, so dropping a folder of 32 mixed files could
      // report nothing at all and read as a no-op.
      const parts = [];
      if (verified) parts.push(`${verified} verified`);
      if (unverified) parts.push(`${unverified} stored unverified`);
      if (dups) parts.push(`${dups} already installed`);
      if (junk) parts.push(`${junk} skipped (not BIOS files)`);
      if (unknowns.length) parts.push(`${unknowns.length} not recognised`);
      if (errs.length) parts.push(`${errs.length} error${errs.length === 1 ? '' : 's'}`);
      let kind = 'ok';
      if (errs.length) kind = 'warn';
      if (!bios.length && !dups) kind = errs.length ? 'error' : 'warn';
      setStatus(
        parts.length
          ? `${parts.join(' · ')}${errs.length ? ` — ${escapeHtml(errs[0].error || '')}` : ''}`
          : 'Nothing to install from that selection',
        kind,
      );
      if (typeof onChange === 'function') onChange();

      // Files we couldn't auto-classify but the user dropped on a
      // specific system row — offer a manual "install as <slot>"
      // override so an uncatalogued-but-real BIOS dump isn't a dead
      // end. One picker at a time; we still have the File objects in
      // scope so no re-upload prompt is needed.
      if (unknowns.length && systemHint) {
        const sys = _systems.find(s => s.system_id === systemHint);
        if (sys) {
          // Match on the LEAF name. A zip member is reported as
          // `pack.zip!bios/scph1001.bin`, which never equals any
          // File.name, so the picker used to be unreachable for
          // anything that arrived inside an archive.
          const leafOf = (n) => String(n || '')
            .split('!').pop().split('/').pop().split('\\').pop();
          let manualInstalled = 0;
          for (const u of unknowns) {
            const target = leafOf(u.filename);
            const f = files.find(file => leafOf(file.name) === target);
            if (!f) continue;
            // eslint-disable-next-line no-await-in-loop
            if (await _offerManualBiosInstall(sys, f, u)) manualInstalled += 1;
          }
          if (manualInstalled) {
            setStatus(`${manualInstalled} installed manually`, 'ok');
            if (typeof onChange === 'function') onChange();
          }
        }
      }
      await refresh();
    } catch (e) {
      setStatus(`Upload failed: ${e.message || e}`, 'error');
    }
  };

  const deleteBios = async (systemId, name) => {
    if (!systemId || !name) return;
    if (!confirm(`Remove ${name} for ${systemId}?`)) return;
    try {
      const r = await fetch(
        `/api/titles/bios/${encodeURIComponent(systemId)}/${encodeURIComponent(name)}`,
        { method: 'DELETE' },
      );
      if (!r.ok && r.status !== 404) {
        const txt = await r.text().catch(() => '');
        throw new Error(`delete failed: ${r.status} ${txt.slice(0, 200)}`);
      }
      setStatus(`Removed ${name}.`, 'ok');
      if (typeof onChange === 'function') onChange();
      await refresh();
    } catch (e) {
      setStatus(`Delete failed: ${e.message || e}`, 'error');
    }
  };

  const _wireRows = () => {
    list.querySelectorAll('[data-bios-upload]').forEach(btn => {
      btn.addEventListener('click', () => {
        const systemHint = btn.dataset.biosSystem || '';
        const wantFolder = btn.dataset.biosUpload === 'folder';
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = true;
        if (wantFolder) {
          // The only way to get a folder out of a file dialog. Chrome/
          // Edge/Safari use webkitdirectory; Firefox honours the
          // `directory` attribute. Setting both is the portable form.
          input.webkitdirectory = true;
          input.setAttribute('webkitdirectory', '');
          input.setAttribute('directory', '');
        }
        input.style.display = 'none';
        document.body.appendChild(input);
        input.addEventListener('change', () => {
          const files = Array.from(input.files || []);
          document.body.removeChild(input);
          if (files.length) uploadFiles(files, systemHint);
        });
        input.click();
      });
    });
    list.querySelectorAll('[data-bios-delete]').forEach(btn => {
      btn.addEventListener('click', () => {
        deleteBios(btn.dataset.biosSystem, btn.dataset.biosName);
      });
    });

    // Per-row drag-drop. Highlight the row the user is hovering over
    // so the scoped upload target is obvious. dragover MUST preventDefault
    // or drop never fires.
    list.querySelectorAll('.bios-vault-row').forEach(row => {
      row.addEventListener('dragover', (e) => {
        e.preventDefault();
        row.classList.add('is-drag-over');
      });
      row.addEventListener('dragleave', () => {
        row.classList.remove('is-drag-over');
      });
      row.addEventListener('drop', (e) => {
        e.preventDefault();
        row.classList.remove('is-drag-over');
        const sid = row.dataset.biosRowSystem || '';
        // Snapshot the entries SYNCHRONOUSLY. DataTransferItemList is
        // neutered the moment this handler yields, so grabbing them
        // after an await returns null for every item and folder drops
        // silently collapse to nothing.
        const entries = Array.from(e.dataTransfer?.items || [])
          .map(it => (typeof it.webkitGetAsEntry === 'function'
            ? it.webkitGetAsEntry() : null))
          .filter(Boolean);
        const flat = Array.from(e.dataTransfer?.files || []);
        _collectDroppedFiles(entries, flat)
          .then(files => { if (files.length) uploadFiles(files, sid); })
          .catch(err => setStatus(`Couldn't read that drop: ${err.message || err}`, 'error'));
      });
    });
  };

  refresh();
}


async function _emulatorFetch(/* {sort, page} */) {
  // Always prepend a synthetic upload card -- the user needs the
  // affordance whether or not they have ROMs installed yet.
  const uploadCard = { _action: 'upload', id: '__upload__' };
  try {
    const r = await fetch('/api/titles/?kind=emulator_rom&limit=200');
    if (!r.ok) {
      if (r.status === 503) return { items: [uploadCard], hasMore: false };
      throw new Error(`titles fetch failed: ${r.status}`);
    }
    const body = await r.json();
    const titles = Array.isArray(body.titles) ? body.titles : [];
    return { items: [uploadCard, ...titles], hasMore: false };
  } catch (e) {
    showToast(`Couldn't list ROMs: ${e.message}`, 'error');
    return { items: [uploadCard], hasMore: false };
  }
}

async function _emulatorLaunch(item) {
  if (item._action === 'upload') {
    return _emulatorUploadFlow();
  }

  // Pre-flight BIOS check for systems that hard-require it (PSX,
  // Saturn, 3DO, Lynx, Amiga, PS2). Handing off to the emulator
  // surface with no BIOS produces a much worse failure mode — a
  // streamed PCSX2 container exits 70 inside the WebRTC session, an
  // EmulatorJS instance renders a blank canvas with no error. Better
  // to catch it here, surface the vault scoped to the missing system,
  // and auto-retry the launch the moment the user finishes installing.
  const sid = item.metadata?.system_id;
  if (sid && _BIOS_REQUIRED_SYSTEMS.has(sid)) {
    const status = await _checkBiosReady(sid);
    if (status === 'missing') {
      let hasRetried = false;
      openBiosVault({
        focusSystemId: sid,
        onChange: async () => {
          // Auto-relaunch the original ROM the moment the system
          // flips from missing → ready. Guard against repeated
          // firing (a user deleting + re-uploading a BIOS would
          // otherwise loop) by latching hasRetried.
          if (hasRetried) return;
          const next = await _checkBiosReady(sid);
          if (next === 'ready') {
            hasRetried = true;
            document.querySelector('.bios-vault-overlay')?.remove();
            _emulatorLaunch(item);
          }
        },
      });
      return;
    }
    // status === 'unknown' (network blip, BIOS subsystem unavailable)
    // falls through — better to attempt the launch and let the
    // emulator surface its own error than block on a transient.
  }

  try {
    const m = await import('./emulator-stage.js');
    await m.openEmulatorStage(item);
  } catch (e) {
    showToast(`Couldn't open emulator: ${e.message}`, 'error');
  }
}


export async function removeEmulatorTitle(titleId, titleName = '') {
  // Delete an installed ROM title: drops the artifact, its run
  // history, every save slot, and releases the ROM blob server-side
  // (see DELETE /api/titles/{id}). Confirms first -- save progress
  // goes with it and the ROM has to be re-uploaded to come back.
  if (!titleId) return false;
  const label = titleName ? `"${titleName}"` : 'this ROM';
  const ok = window.confirm(
    `Remove ${label} from your library?\n\n`
    + 'This deletes the ROM and any saved progress. You can re-add it '
    + 'by uploading the ROM file again.',
  );
  if (!ok) return false;
  try {
    const r = await fetch(`/api/titles/${encodeURIComponent(titleId)}`, {
      method: 'DELETE',
    });
    if (!r.ok) {
      const txt = await r.text().catch(() => '');
      throw new Error(`${r.status} ${txt.slice(0, 200)}`);
    }
    showToast('ROM removed from library', 'success');
    window.dispatchEvent(new CustomEvent('library:games-source-refresh', {
      detail: { source: 'emulator' },
    }));
    return true;
  } catch (e) {
    showToast(`Couldn't remove ROM: ${e.message}`, 'error');
    return false;
  }
}


async function _checkBiosReady(systemId) {
  // Returns 'ready' | 'missing' | 'unknown'. The launch path uses
  // 'unknown' as a soft-pass so a transient BIOS-subsystem failure
  // doesn't permanently block ROM launches.
  try {
    const r = await fetch(
      `/api/titles/bios/status?system_id=${encodeURIComponent(systemId)}`,
    );
    if (!r.ok) return 'unknown';
    const body = await r.json();
    return body.all_required_present ? 'ready' : 'missing';
  } catch (_) {
    return 'unknown';
  }
}

function _emulatorUploadFlow() {
  // Folder picker. Modern browsers honour `webkitdirectory` to let
  // the user pick a whole tree; the resulting FileList contains every
  // descendant file with `.webkitRelativePath` set. We accept a
  // single-file selection too (the user can multi-select inside the
  // dialog), and the drag-drop path uses webkitGetAsEntry to walk a
  // dropped folder. All paths funnel into ``_bulkImportRoms()`` which
  // does the actual sequential upload + progress UI.
  const input = document.createElement('input');
  input.type = 'file';
  input.multiple = true;
  // Both attributes -- some browsers prefer `webkitdirectory`,
  // Firefox honours `directory`. Setting both opens a folder picker
  // where the user can also fall back to picking individual files.
  input.setAttribute('webkitdirectory', '');
  input.setAttribute('directory', '');
  input.accept = '';
  input.style.display = 'none';
  document.body.appendChild(input);
  input.addEventListener('change', async () => {
    const files = Array.from(input.files || []);
    document.body.removeChild(input);
    if (!files.length) return;
    await _bulkImportRoms(files);
  });
  input.click();
}


// Drag-drop entry point used by library.js. Receives entries and
// fallback files that the caller already snapshotted from the
// drop event SYNCHRONOUSLY (DataTransfer goes inert the moment a
// drop handler returns control, so any await between the drop and
// reading items wipes the list -- caller's job to capture them
// first, ours to walk).
async function _emulatorImportFromEntries(entries, fallbackFiles) {
  const collected = [];
  if (entries && entries.length) {
    // Walk every dropped entry (file or directory) recursively.
    for (const e of entries) {
      try {
        await _walkEntry(e, '', collected);
      } catch (err) {
        // One bad subtree shouldn't kill the whole drop. Log + move on.
        console.warn('[rom-import] walk failed for', e?.name, err);
      }
    }
  }
  // If the drop carried no entries (some sources omit them) but did
  // carry top-level files (typical for drag from a browser tab), use
  // those as the fallback. Top-level files only -- we can't recurse
  // without entries.
  if (!collected.length && fallbackFiles && fallbackFiles.length) {
    for (const f of fallbackFiles) collected.push(f);
  }
  if (!collected.length) {
    showToast('No files in drop.', 'warn');
    return;
  }
  await _bulkImportRoms(collected);
}


async function _walkEntry(entry, prefix, out) {
  // Recursively walk a FileSystemEntry. Files get their relative
  // path stamped onto the resulting File object so progress UI shows
  // "Nintendo - Game Boy/Tetris.gb" instead of just "Tetris.gb".
  // Directories are read in batches (the DOM API caps a single
  // readEntries() call at ~100 children, so we loop until empty).
  if (!entry) return;
  if (entry.isFile) {
    const f = await new Promise((resolve, reject) => {
      entry.file(resolve, reject);
    });
    if (f) {
      try {
        Object.defineProperty(f, 'webkitRelativePath', {
          value: prefix + f.name,
          configurable: true,
        });
      } catch (_) { /* property is read-only on some engines; non-fatal */ }
      out.push(f);
    }
    return;
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    const childPrefix = prefix + entry.name + '/';
    // Loop until readEntries returns an empty batch -- the spec
    // guarantees we'll see every child eventually but in batches.
    while (true) {
      const batch = await new Promise((resolve) => {
        reader.readEntries(resolve, () => resolve([]));
      });
      if (!batch || !batch.length) break;
      for (const child of batch) {
        await _walkEntry(child, childPrefix, out);
      }
    }
  }
}


// Back-compat shim used to be wired against a live DataTransfer; the
// drop-handler in library.js now snapshots entries synchronously and
// calls _emulatorImportFromEntries directly. Kept exported for any
// future callers that pass a DataTransfer through.
async function _emulatorImportFromDataTransfer(dataTransfer) {
  if (!dataTransfer) return;
  const items = Array.from(dataTransfer.items || [])
    .map(it => it.webkitGetAsEntry?.())
    .filter(Boolean);
  const files = Array.from(dataTransfer.files || []);
  return _emulatorImportFromEntries(items, files);
}


// Per-batch caps for the bulk-import POST. Server enforces its own
// per-file size cap; these are the chunking boundaries that bound
// memory usage on both ends and let progress UI tick smoothly. A
// 1000-file folder drop is split into ~30-40 batches with these
// numbers; each batch is one HTTP roundtrip.
const _BULK_BATCH_FILES = 32;
const _BULK_BATCH_BYTES = 200 * 1024 * 1024;   // 200 MB


async function _bulkImportRoms(files) {
  // Drop-anything import: the server classifies every file (ROM,
  // BIOS, archive, junk, unknown) and routes accordingly. We send
  // everything except obvious giant non-ROM-non-BIOS files in
  // chunked batches to /api/titles/bulk-import.
  if (!files || !files.length) return;

  const ui = _openImportProgress(files.length);
  ui.setStatus('Preparing...');

  // Aggregated summary across all batches. Mirrors the server-side
  // bucket shape so we can render the same digest the user would see
  // for a single batch.
  const totals = {
    imported: [], bios: [], duplicates: [],
    junk: [], unknown: [], errors: [],
  };

  // Chunk into batches by count + cumulative byte size, whichever
  // hits first. Single-file batches ride through unchanged when one
  // file is bigger than the byte budget.
  const batches = [];
  let current = [];
  let currentBytes = 0;
  for (const f of files) {
    const size = Number(f.size) || 0;
    if (
      current.length >= _BULK_BATCH_FILES ||
      (current.length > 0 && currentBytes + size > _BULK_BATCH_BYTES)
    ) {
      batches.push(current);
      current = [];
      currentBytes = 0;
    }
    current.push(f);
    currentBytes += size;
  }
  if (current.length) batches.push(current);

  let processedFiles = 0;
  for (let bi = 0; bi < batches.length; bi++) {
    const batch = batches[bi];
    ui.setStatus(
      `Batch ${bi + 1} of ${batches.length} (${batch.length} files)...`,
    );
    try {
      const summary = await _postBulkImportBatch(batch);
      // Merge each bucket. Server returns arrays; we concatenate.
      for (const k of Object.keys(totals)) {
        if (Array.isArray(summary[k])) {
          totals[k] = totals[k].concat(summary[k]);
        }
      }
    } catch (e) {
      // Whole-batch failure -- surface every file in it as an error.
      for (const f of batch) {
        totals.errors.push({
          filename: f.webkitRelativePath || f.name,
          error: `batch failed: ${e.message}`,
        });
      }
    }
    processedFiles += batch.length;
    ui.setProgress((processedFiles / files.length) * 100);
  }

  ui.setProgress(100);
  ui.setStatus('Done');

  const lines = [
    totals.imported.length ? `${totals.imported.length} ROMs imported` : '',
    totals.bios.length ? `${totals.bios.length} BIOS installed` : '',
    totals.duplicates.length ? `${totals.duplicates.length} duplicates` : '',
    totals.junk.length ? `${totals.junk.length} junk skipped` : '',
    totals.unknown.length ? `${totals.unknown.length} need review` : '',
    totals.errors.length ? `${totals.errors.length} errors` : '',
  ].filter(Boolean).join(' · ') || 'Nothing to import';

  // Build the failed/notes list. Errors and unknowns are the user-
  // facing follow-ups; junk and duplicates are shown as counts only.
  const followUps = [];
  for (const u of totals.unknown) {
    followUps.push(`${u.filename} - ${u.reason || 'unknown file'}`);
  }
  for (const e of totals.errors) {
    followUps.push(`${e.filename} - ${e.error}`);
  }
  ui.setSummary(lines, followUps);

  // Refresh the grid if anything was actually imported. BIOS installs
  // also count -- the launch path now consumes them, so a refresh
  // unblocks any title that was previously bios-missing.
  if (totals.imported.length || totals.bios.length) {
    window.dispatchEvent(new CustomEvent('library:games-source-refresh', {
      detail: { source: 'emulator' },
    }));
  }
}


async function _postBulkImportBatch(files) {
  // One multipart POST containing every file in this batch. Each
  // file is appended as a 'files' field so the server's form.getlist
  // walk picks them all up. Filename preserves webkitRelativePath
  // when present so the server's diagnostics show "Roms/Sega/Sonic.md"
  // instead of just "Sonic.md".
  const fd = new FormData();
  for (const f of files) {
    const name = f.webkitRelativePath || f.name;
    fd.append('files', f, name);
  }
  const r = await fetch('/api/titles/bulk-import', {
    method: 'POST', body: fd,
  });
  if (!r.ok) {
    if (r.status === 413) {
      throw new Error('batch too large; reduce batch size');
    }
    if (r.status === 401) {
      throw new Error('not signed in');
    }
    throw new Error(`HTTP ${r.status}`);
  }
  const body = await r.json();
  return body.summary || {};
}


function _openImportProgress(total) {
  // Lightweight modal -- backdrop + card with progress bar, status
  // line, and a Close button (disabled until the run finishes).
  const overlay = document.createElement('div');
  overlay.className = 'rom-import-overlay';
  overlay.innerHTML = `
    <div class="rom-import-card">
      <div class="rom-import-header">Importing ROMs (${total})</div>
      <div class="rom-import-status"></div>
      <div class="rom-import-bar"><div class="rom-import-bar-fill"></div></div>
      <div class="rom-import-summary"></div>
      <div class="rom-import-failed"></div>
      <div class="rom-import-actions">
        <button class="rom-import-close" disabled>Working...</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  const status = overlay.querySelector('.rom-import-status');
  const fill = overlay.querySelector('.rom-import-bar-fill');
  const summary = overlay.querySelector('.rom-import-summary');
  const failedList = overlay.querySelector('.rom-import-failed');
  const closeBtn = overlay.querySelector('.rom-import-close');

  closeBtn.addEventListener('click', () => {
    if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
  });

  return {
    setStatus(text) { status.textContent = text; },
    setProgress(pct) { fill.style.width = `${Math.max(0, Math.min(100, pct))}%`; },
    setSummary(text, failed) {
      summary.textContent = text;
      if (failed && failed.length) {
        failedList.innerHTML = '<details><summary>Show failures</summary><ul>'
          + failed.map(n => `<li>${escapeHtml(n)}</li>`).join('')
          + '</ul></details>';
      }
      closeBtn.disabled = false;
      closeBtn.textContent = 'Done';
    },
  };
}

// Public re-exports so library.js can wire the drop handler on the
// emulator upload tile without library-game-sources needing to know
// about the grid container.
//
//   importEmulatorRomsFromEntries(entries, fallbackFiles)
//     Preferred entry point. Caller MUST extract entries +
//     dataTransfer.files synchronously inside the drop handler --
//     DataTransfer goes inert the moment the handler returns.
//
//   importEmulatorRomsFromDataTransfer(dataTransfer)
//     Convenience wrapper if the caller can guarantee no async
//     work happens between the drop and the call (rare).
export const importEmulatorRomsFromEntries = _emulatorImportFromEntries;
export const importEmulatorRomsFromDataTransfer = _emulatorImportFromDataTransfer;


// ── Registry ───────────────────────────────────────────────────────

export const GAME_SOURCES = {
  js13k: {
    id: 'js13k',
    label: 'js13k',
    hint: 'Plays in-app',
    isEnabled: (s) => s.gamePortalEnabled !== false
                       && _csvIncludes(s.gamePortalDefaultSources, 'js13k'),
    subtitle: 'Plays in-app · saves follow your account · MIT-licensed 13KB games',
    sortable: false,
    fetch: _browseGamesFetch('js13k'),
    renderCard: _sharedRenderBrowseCard,
  },

  marketplace: {
    id: 'marketplace',
    label: 'Curated',
    hint: 'Hand-picked · in-app',
    // Default-on: the curated catalog ships in-tree as a static JS array,
    // so it has no backend dependency. Gated on the games-portal master
    // toggle (same as js13k) — which defaults to true — so existing users
    // see the tab without touching Settings. The legacy `marketplaceEnabled`
    // flag still controls the future server-managed marketplace API and
    // is intentionally independent of this client-side curated list.
    isEnabled: (s) => s.gamePortalEnabled !== false,
    subtitle: 'Hand-picked free browser games · pin one and it lives in your library',
    sortable: false,
    fetch: _marketplaceFetch,
    // Shared browse-card renderer (same shape as js13k cards) — keeps
    // the marquee surface visually unified. The Pin button uses the
    // generic [data-pin-src]/[data-pin-sid] handler in library.js.
    renderCard: _sharedRenderBrowseCard,
  },

  streamed: {
    id: 'streamed',
    label: 'Streamed',
    hint: 'In-app · server-rendered',
    isEnabled: (s) => s.gameStreamEnabled === true,
    subtitle: 'Server-rendered open-source games · streams to your browser · plays on any device',
    sortable: false,
    fetch: _streamedFetch,
    renderCard: _renderStreamedCard,
    onLaunch: _streamedLaunch,
  },

  emulator: {
    id: 'emulator',
    label: 'ROMs',
    hint: 'Retro · plays in-app',
    // Gated on titlesEnabled (master toggle for the AXF Title
    // framework, which is what /api/titles/* requires). Hiding the
    // tab when the framework is off avoids a stream of 503 errors
    // the user can't fix from inside this surface.
    isEnabled: (s) => s.titlesEnabled !== false,
    subtitle: 'Your installed ROMs · NES, SNES, GBA and more · saves follow your account',
    sortable: false,
    fetch: _emulatorFetch,
    renderCard: _renderEmulatorCard,
    renderFilters: _renderEmulatorFilters,
    applyFilters: _applyEmulatorFilters,
    onLaunch: _emulatorLaunch,
  },

  // ── Future sources slot in here exactly the same way ────────────
  //
  // emulator: {
  //   id: 'emulator',
  //   label: 'Retro',
  //   hint: 'Emulated · in-app',
  //   isEnabled: (s) => s.emulatorEnabled === true,
  //   ...
  // },
  // github: {
  //   id: 'github',
  //   label: 'GitHub',
  //   hint: 'Open-source builds',
  //   isEnabled: (s) => s.githubGamesEnabled === true,
  //   ...
  // },
};

// ── Helpers ────────────────────────────────────────────────────────

export function enabledSources(settings) {
  return Object.values(GAME_SOURCES).filter(src => src.isEnabled(settings));
}

export function anyEnabled(settings) {
  return enabledSources(settings).length > 0;
}

export function getSource(id) {
  return GAME_SOURCES[id] || null;
}

function _csvIncludes(csv, value) {
  if (typeof csv !== 'string') return false;
  return csv.split(',').map(s => s.trim()).includes(value);
}
