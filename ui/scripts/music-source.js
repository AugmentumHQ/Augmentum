/**
 * music-source.js — surface-agnostic music source layer.
 *
 * The single source of truth for "find me something to play" across every
 * surface: the Grove control center, the cast-comic TV reader's music bed,
 * cast-audio, and anything future. It owns:
 *
 *   - Favorites I/O (server-backed: /api/grove/favorites) + a SomaFM seed.
 *   - The rank matcher that turns a fuzzy "play <query>" into a favorite.
 *   - Combined search across the four real sources — radio stations
 *     (SomaFM + radio-browser), local music files, and YouTube ambient —
 *     each returning DATA, never touching the DOM.
 *   - The tiered resolver (favorite → files → youtube → loose favorite →
 *     radio discovery) that used to live inside grove.js, refactored to
 *     return a NORMALIZED SOURCE DESCRIPTOR instead of playing anything.
 *
 * Why descriptors, not playback: Grove plays a station through an <audio>
 * element + its orb; a cast surface plays it through a different pipeline
 * and mixes it UNDER comic narration via AudioBus. Fusing resolution to
 * one surface's playback (the old grove.js shape) is exactly what blocked
 * reuse. Resolution returns WHAT to play; each surface decides HOW.
 *
 * Normalized source descriptor (the contract every surface consumes):
 *   {
 *     kind:   'station' | 'file' | 'youtube',
 *     id:     string,              // stable id within its kind
 *     name:   string,
 *     genre:  string,              // best-effort; '' when unknown
 *     desc:   string,              // best-effort secondary label
 *     url:    string | null,       // direct stream URL (station/file); null for youtube
 *     videoId:string | null,       // youtube only
 *     poster: string | null,       // favicon / thumbnail when available
 *     source: string,              // provenance tag: 'favorites'|'files'|'youtube'|'discover'|<origin>
 *     raw:    object,              // the untouched upstream row, for surface-specific fields
 *   }
 *
 * NOTHING here reads or writes the DOM, localStorage UI state, or a specific
 * audio element. Recency bias is injected by the caller (a rotation object),
 * so the module stays free of surface state.
 */

// ── Favorites I/O (server-backed) ──────────────────────────────────────────
// Favorites persist server-side per user (CLAUDE.md: default to server-side
// persistence) so the same set follows the user across web + every cast
// surface. A stale comment in the old grove.js claimed localStorage — the
// code always used the endpoint; this module makes that the only path.

/** Load the user's favorites. Falls back to seeding the first four SomaFM
 *  channels on an empty/failed load (and persists that seed), mirroring the
 *  original grove behavior so first-run isn't an empty picker. */
export async function loadFavorites() {
  try {
    const resp = await fetch('/api/grove/favorites', { credentials: 'same-origin' });
    if (resp.ok) {
      const data = await resp.json();
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch { /* fall through to the SomaFM seed */ }

  try {
    const resp = await fetch('/api/grove/stations/soma', { credentials: 'same-origin' });
    if (resp.ok) {
      const soma = await resp.json();
      const seed = Array.isArray(soma) ? soma.slice(0, 4) : [];
      if (seed.length) await saveFavorites(seed);
      return seed;
    }
  } catch { /* no stations available */ }
  return [];
}

/** Persist the favorites array. Best-effort; callers keep their in-memory
 *  copy as the working set. */
export async function saveFavorites(favorites) {
  try {
    await fetch('/api/grove/favorites', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Array.isArray(favorites) ? favorites : []),
    });
    return true;
  } catch {
    return false;
  }
}

// ── Favorite matcher (pure) ─────────────────────────────────────────────────
// Turns a fuzzy query into the best favorite. Rank 0 = exact name (wins
// outright), higher = looser, rank 4 = token overlap (the voice model wraps
// a genre in mood language — "warm grays single instrument ambient jazz"
// must still hit "Smooth Jazz 24/7" on the shared "jazz" token). Ported
// verbatim from grove.js::_findMatchingFavorite so behavior is identical.

const _MATCH_STOPWORDS = new Set(
  ['the', 'a', 'an', 'and', 'of', 'for', 'some', 'with', 'music'],
);

/** Match a favorite without playing it. Returns {ok, pick, rank} or
 *  {ok:false, reason}. `favorites` is the caller's working set. */
export function matchFavorite(query, favorites) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return { ok: false, reason: 'empty-query' };
  if (!Array.isArray(favorites) || favorites.length === 0) {
    return { ok: false, reason: 'no-favourites' };
  }

  const qTokens = q.split(/[^a-z0-9]+/).filter(t => t.length > 2 && !_MATCH_STOPWORDS.has(t));
  let pick = null;
  let pickRank = 99;
  let pickOverlap = 0;
  for (const f of favorites) {
    if (!f || !f.url) continue;
    const name = String(f.name || '').toLowerCase();
    const genre = String(f.genre || '').toLowerCase();
    const desc = String(f.desc || '').toLowerCase();
    let rank = 99;
    let overlap = 0;
    if (name === q) rank = 0;
    else if (genre && (genre === q || genre.includes(q) || q.includes(genre))) rank = 1;
    else if (name.includes(q)) rank = 2;
    else if (desc.includes(q)) rank = 3;
    else if (qTokens.length) {
      const hay = `${name} ${genre} ${desc}`;
      overlap = qTokens.filter(t => hay.includes(t)).length;
      if (overlap > 0) rank = 4;
    }
    const better = rank < pickRank
      || (rank === 4 && rank === pickRank && overlap > pickOverlap);
    if (better) {
      pick = f;
      pickRank = rank;
      pickOverlap = overlap;
      if (rank === 0) break;  // exact name match wins outright
    }
  }

  if (!pick) return { ok: false, reason: 'no-match' };
  return { ok: true, pick, rank: pickRank };
}

// ── Normalization ───────────────────────────────────────────────────────────
// Every upstream row (station / file / youtube video) is folded into one
// descriptor shape so surfaces never branch on provenance to read a name.

/** Normalize a radio-station row (SomaFM / radio-browser / favorites). */
export function stationToSource(s, { source } = {}) {
  if (!s) return null;
  return {
    kind: 'station',
    id: String(s.id ?? ''),
    name: s.name || 'Radio',
    genre: s.genre || '',
    desc: s.desc || s.genre || '',
    url: s.url || null,
    videoId: null,
    poster: s.favicon || null,
    source: source || s.source || 'radio',
    raw: s,
  };
}

/** Normalize a local music file row from /api/files/search. */
export function fileToSource(f, { genre = '' } = {}) {
  if (!f) return null;
  return {
    kind: 'file',
    id: String(f.id ?? ''),
    name: f.name || 'Track',
    genre,
    desc: f.artist || f.album || '',
    // Files play by id through the media pipeline; a direct URL is optional.
    url: f.url || null,
    videoId: null,
    poster: f.poster || f.cover || null,
    source: 'files',
    raw: f,
  };
}

/** Normalize a YouTube ambient video row. */
export function youtubeToSource(v, { genre = '' } = {}) {
  if (!v) return null;
  return {
    kind: 'youtube',
    id: String(v.videoId ?? ''),
    name: v.title || 'Video',
    genre,
    desc: v.channel || '',
    url: null,
    videoId: v.videoId || null,
    poster: v.thumbnail || null,
    source: 'youtube',
    raw: v,
  };
}

// ── Combined search (data only) ─────────────────────────────────────────────

/** Search radio stations. `source` selects the backend leg:
 *    'all'    → /api/grove/stations/search   (SomaFM + radio-browser merge)
 *    'radio'  → /api/grove/stations/radio     (radio-browser only)
 *    'somafm' → /api/grove/stations/soma      (full SomaFM list; client-filter)
 *  Returns an array of raw station rows (filtered for SomaFM when a query or
 *  genre is supplied, matching the old client-side filter). */
export async function searchStations({ q = '', genre = '', source = 'all', limit = 20, signal } = {}) {
  const params = new URLSearchParams();
  if (q) params.set('q', q);
  if (genre) {
    params.set('tag', genre.toLowerCase());
    params.set('genre', genre.toLowerCase());
  }
  params.set('limit', String(limit));

  let url;
  if (source === 'radio') url = `/api/grove/stations/radio?${params}`;
  else if (source === 'somafm') url = '/api/grove/stations/soma';
  else url = `/api/grove/stations/search?${params}`;

  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 8000);
  const composite = _anySignal([signal, ctl.signal]);
  try {
    const resp = await fetch(url, { signal: composite, credentials: 'same-origin' });
    if (!resp.ok) throw new Error('fetch failed');
    let stations = await resp.json();
    if (!Array.isArray(stations)) stations = stations?.stations || [];

    // SomaFM has no server-side query — filter client-side, same as before.
    if (source === 'somafm' && q) {
      const needle = q.toLowerCase();
      stations = stations.filter(s =>
        (s.name || '').toLowerCase().includes(needle) ||
        (s.desc || '').toLowerCase().includes(needle) ||
        (s.genre || '').toLowerCase().includes(needle));
    }
    if (source === 'somafm' && genre) {
      const g = genre.toLowerCase();
      stations = stations.filter(s => (s.genre || '').toLowerCase().includes(g));
    }
    return stations;
  } finally {
    clearTimeout(timer);
  }
}

/** Search local music files (entity_kind=music only — never an audiobook
 *  chapter for a genre ask). Returns raw file rows with an id. */
export async function searchFiles({ q = '', limit = 8, signal } = {}) {
  const params = new URLSearchParams({
    q, kind: 'audio', entity_kind: 'music', limit: String(limit),
  });
  const resp = await fetch(`/api/files/search?${params}`, {
    credentials: 'same-origin', signal,
  });
  if (!resp.ok) return [];
  const data = await resp.json();
  return (data?.files || []).filter(f => f && f.id);
}

// YouTube search cache — module-level so every surface shares warmed results
// (Grove's prewarm, hover-prefetch, and a cast picker all hit the same map).
const _ytCache = new Map();          // normKey → { videos, ts }
const _YT_CACHE_TTL = 5 * 60 * 1000;

/** Canonical genre→query map (matches the backend prewarm keys). */
export const YT_GENRE_QUERIES = {
  ambient:    'ambient music',
  'lo-fi':    'lofi hip hop music',
  electronic: 'electronic ambient music',
  classical:  'classical music relaxing',
  jazz:       'jazz music relaxing',
  focus:      'focus music study',
  nature:     'nature sounds ambiance',
  synthwave:  'synthwave retrowave music',
};

/** Normalize a YT query so cache keys don't miss on casing/whitespace. */
export function normYtKey(q) {
  return (q || '').toLowerCase().trim().replace(/\s+/g, ' ');
}

/** Seed the shared YT cache (used by grove's prewarm response). */
export function primeYouTubeCache(query, videos, ts = _nowOr(0)) {
  if (!query || !Array.isArray(videos) || !videos.length) return;
  _ytCache.set(normYtKey(query), { videos, ts });
}

/** Synchronous cache peek — returns the cached video array if present and
 *  unexpired, else null. Lets a surface render instantly on a hit instead
 *  of flashing a loading state before the async search resolves. */
export function peekYouTubeCache(query) {
  const cached = _ytCache.get(normYtKey(query));
  if (cached && (_nowOr(cached.ts + 1) - cached.ts) < _YT_CACHE_TTL) return cached.videos;
  return null;
}

/** Fetch YT results for `query`, memoized in the shared cache. Returns the
 *  video array (possibly cached) or null on failure. Pure data. */
export async function searchYouTube(query, { limit = 12, signal } = {}) {
  const key = normYtKey(query);
  const cached = _ytCache.get(key);
  if (cached && (_nowOr(cached.ts + 1) - cached.ts) < _YT_CACHE_TTL) return cached.videos;

  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 10000);
  const composite = _anySignal([signal, ctl.signal]);
  try {
    const resp = await fetch(
      `/api/grove/youtube/search?q=${encodeURIComponent(query)}&limit=${limit}`,
      { signal: composite },
    );
    if (!resp.ok) return null;
    const videos = await resp.json();
    if (Array.isArray(videos) && videos.length > 0) {
      _ytCache.set(key, { videos, ts: _nowOr(0) });
      if (_ytCache.size > 20) _ytCache.delete(_ytCache.keys().next().value);
    }
    return videos;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// ── Tiered resolver → descriptor (no playback) ──────────────────────────────
// The resolution ladder (Matt's tier spec 2026-06-10, radio demoted same
// day), ported from grove.js::findAndPlayMatchingOrDiscover but returning a
// descriptor instead of driving <audio>/orb/media-player:
//   1. Favorite, EXACT name only
//   2. Local music files
//   3. YouTube ambient
//   4. Loosely-matched favorite (unless just played)
//   5. Radio station discovery
// A `rotation` object ({ note, preferFresh, isRecent }) biases each tier
// away from what was just played; pass grove-rotation.js's createRotation()
// or omit for no bias.

const _NOOP_ROTATION = {
  note() {},
  isRecent() { return false; },
  preferFresh(list) { return list[0]; },
};

/**
 * Resolve a fuzzy query to a normalized source descriptor.
 * @returns {Promise<{ok:true, tier:number, descriptor:object} | {ok:false, reason:string}>}
 */
export async function resolveSource(query, {
  favorites = [],
  genreHints = [],
  rotation = _NOOP_ROTATION,
  log = () => {},
} = {}) {
  // Tier 1 — exact favorite.
  const fav = matchFavorite(query, favorites);
  if (fav.ok && fav.rank === 0) {
    log('tier 1 exact favourite', query, fav.pick.name);
    rotation.note(fav.pick.id);
    return { ok: true, tier: 1, descriptor: stationToSource(fav.pick, { source: 'favorites' }) };
  }

  const q = String(query || '').trim();
  if (!q) return fav.ok ? fav : { ok: false, reason: fav.reason || 'empty-query' };

  const attempts = [q];
  for (const g of genreHints) {
    const gq = String(g || '').trim();
    if (gq && !attempts.includes(gq)) attempts.push(gq);
  }
  log('favourites missed; ladder attempts', attempts.join(', '));

  // Tier 2 — local music files.
  for (const attempt of attempts) {
    try {
      const files = await searchFiles({ q: attempt, limit: 8 });
      if (!files.length) continue;
      const pick = rotation.preferFresh(files, f => f.id);
      log('tier 2 files', attempt, pick.name);
      rotation.note(pick.id);
      return { ok: true, tier: 2, descriptor: fileToSource(pick, { genre: attempt }) };
    } catch (err) {
      log('files tier failed', attempt, String(err));
    }
  }

  // Tier 3 — YouTube ambient.
  for (const attempt of attempts) {
    try {
      const videos = await searchYouTube(`${attempt} music`);
      const playable = (videos || []).filter(v => v && v.videoId);
      if (!playable.length) continue;
      const pick = rotation.preferFresh(playable, v => v.videoId);
      log('tier 3 youtube', attempt, pick.title);
      rotation.note(pick.videoId);
      return { ok: true, tier: 3, descriptor: youtubeToSource(pick, { genre: attempt }) };
    } catch (err) {
      log('youtube tier failed', attempt, String(err));
    }
  }

  // Tier 4 — loosely-matched favorite (unless it's what we just played).
  if (fav.ok && !rotation.isRecent(fav.pick.id)) {
    log('tier 4 loose favourite', query, fav.pick.name);
    rotation.note(fav.pick.id);
    return { ok: true, tier: 4, descriptor: stationToSource(fav.pick, { source: 'favorites' }) };
  }
  if (fav.ok) log('tier 4 skipped (just played); trying discovery', fav.pick.name);

  // Tier 5 — radio station discovery.
  for (const attempt of attempts) {
    try {
      const list = await searchStations({ q: attempt, limit: 12, source: 'all' });
      const playable = list.filter(s => s && s.url);
      if (!playable.length) continue;
      const pick = rotation.preferFresh(playable, s => s.id, 5);
      log('tier 5 stations', attempt, pick.name);
      rotation.note(pick.id);
      return { ok: true, tier: 5, descriptor: stationToSource(pick, { source: 'discover' }) };
    } catch (err) {
      log('station tier failed', attempt, String(err));
    }
  }
  return { ok: false, reason: 'no-match-anywhere' };
}

// ── Internals ────────────────────────────────────────────────────────────────

/** Merge multiple abort signals into one. */
function _anySignal(signals) {
  const ctl = new AbortController();
  for (const s of signals) {
    if (!s) continue;
    if (s.aborted) { ctl.abort(); break; }
    s.addEventListener('abort', () => ctl.abort(), { once: true });
  }
  return ctl.signal;
}

// Date.now() is unavailable in some sandboxed contexts and is the only
// wall-clock this module needs (cache TTL). Guard it so the module still
// loads where it's stubbed; a missing clock just disables TTL expiry
// (cache still bounded by size), which is safe.
function _nowOr(fallback) {
  try { return Date.now(); } catch { return fallback; }
}
