/**
 * The Grove — Control Center
 * Atmosphere / Typography / Soundscape / Vitals
 */

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------
import { escapeHtml, setTextScale, ensureTypographyFonts, app, showToast } from './app.js';
import * as ambient from './grove-ambient.js';
import * as orbDetach from './grove-orb-detach.js';
import * as playlist from './playlist.js';
import { AudioBus } from './audio-bus.js';
import { MediaSessionBridge } from './media-session.js';
import {
  recordLastPlayed,
  getLastPlayed,
  isPromptDismissed,
  markPromptDismissed,
} from './grove-resume.js';
import { getLastPlayed as _mediaGetLastPlayed } from './media-resume.js';
import { mountCastButton } from './cast-button.js';
import { createRotation } from './grove-rotation.js';
import * as musicSource from './music-source.js';

// ---------------------------------------------------------------------------
// DOM helper
// ---------------------------------------------------------------------------
const $ = id => document.getElementById(id);

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const THEMES = ['dark', 'light', 'midnight', 'sepia'];
const THEME_META = {
  dark:     { label: 'Dark',     color: '#6c8aff', desc: 'Soft dark with blue accent' },
  light:    { label: 'Light',    color: '#5b73d9', desc: 'Warm stone neutrals' },
  midnight: { label: 'Midnight', color: '#38bdf8', desc: 'Deep inky immersion' },
  sepia:    { label: 'Sepia',    color: '#d4a55a', desc: 'Candlelit warmth' },
};

// Typography presets — mirrors TYPO_PRESETS in app.js
// Each has body (--font-sans), display (--font-narrative-display), mono (--font-mono)
const FONTS = [
  { key: 'system',     label: 'System Default',
    body: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif',
    mono: '"SF Mono", "Cascadia Code", "Fira Code", Consolas, monospace', display: null },
  { key: 'literary',   label: 'Literary',
    body: '"Literata", Georgia, serif',
    mono: '"Fira Code", "Cascadia Code", Consolas, monospace', display: '"EB Garamond", "Palatino Linotype", serif' },
  { key: 'classic',    label: 'Classic',
    body: '"Lora", "Palatino Linotype", Georgia, serif',
    mono: '"Fira Code", Consolas, monospace', display: '"Crimson Text", "Book Antiqua", Palatino, serif' },
  { key: 'editorial',  label: 'Editorial',
    body: '"Source Serif 4", Georgia, serif',
    mono: '"JetBrains Mono", "Fira Code", monospace', display: '"Source Sans 3", "Segoe UI", sans-serif' },
  { key: 'modern',     label: 'Modern',
    body: '"DM Sans", "Inter", sans-serif',
    mono: '"JetBrains Mono", "Fira Code", monospace', display: '"Inter", "DM Sans", sans-serif' },
  { key: 'technical',  label: 'Technical',
    body: '"Inter", -apple-system, sans-serif',
    mono: '"JetBrains Mono", "Cascadia Code", monospace', display: '"Inter", sans-serif' },
  { key: 'readable',   label: 'Readable',
    body: '"Atkinson Hyperlegible", "Verdana", sans-serif',
    mono: '"JetBrains Mono", Consolas, monospace', display: '"Atkinson Hyperlegible", "Verdana", sans-serif' },
  { key: 'typewriter', label: 'Typewriter',
    body: '"JetBrains Mono", "Fira Code", monospace',
    mono: '"JetBrains Mono", "Fira Code", monospace', display: '"JetBrains Mono", "Fira Code", monospace' },
];

const GENRE_CHIPS = [
  'Ambient', 'Lo-fi', 'Electronic', 'Classical',
  'Jazz', 'Focus', 'Nature', 'Synthwave',
];

const PLAY_SVG  = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
const PAUSE_SVG = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';

// Scale step is 0.1 (10% increments). Range (0.7–1.4) is enforced by
// app.js:setTextScale, which is the single source of truth for persistence.
const SCALE_STEP = 0.1;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let isOpen = false;
let audio = null;          // lazily created HTML5 Audio
let isPlaying = false;
let currentStation = null; // { id, name, desc, genre, url, source }
let favorites = [];
let favIndex = -1;         // index in favorites of current station
let discoverSource = 'all';
let discoverGenre = '';
let searchTimeout = null;
let _castBtn = null;       // Cast-to-TV button mounted into transport row
// Recently-played rotation — biases each play-matching tier away from what was
// just played so repeat genre asks ("jazz" again) cycle through options instead
// of replaying one video forever. See grove-rotation.js.
const _rotation = createRotation(6);

// ---------------------------------------------------------------------------
// Settings helpers
// ---------------------------------------------------------------------------
function getSettings() {
  try {
    const raw = localStorage.getItem('augmentum_settings');
    return raw ? JSON.parse(raw) : {};
  } catch { return {}; }
}

function setSetting(key, val) {
  const s = getSettings();
  s[key] = val;
  localStorage.setItem('augmentum_settings', JSON.stringify(s));
}

// ---------------------------------------------------------------------------
// Panel toggle
// ---------------------------------------------------------------------------
function openGrove() {
  const panel = $('grove-panel');
  if (!panel) return;
  // If the panel was soft-closed (orb detached while panel was "closed"),
  // lift that mask now. The panel is already .visible in DOM — soft-close
  // just visually hides it without display:none.
  panel.classList.remove('soft-closed');
  panel.classList.add('visible');
  isOpen = true;
  // Vitals removed — visible in header resource status

  // Warm YouTube search cache for all default genres. The request is
  // fire-and-forget — if it arrives before the user touches Discover, the
  // first click is instant; if it arrives after, it just seeds the cache
  // for the next interaction. No loading UI, no blocking.
  _warmYouTubeCache();
}

// ---------------------------------------------------------------------------
// Prewarm — kick off a server fetch that returns cached results for the
// 8 default genres. Populates the shared music-source YT cache so chip clicks /
// discover-open render instantly.
// ---------------------------------------------------------------------------
let _prewarmed = false;
let _prewarmInFlight = null;

async function _warmYouTubeCache() {
  if (_prewarmed || _prewarmInFlight) return;
  _prewarmInFlight = (async () => {
    try {
      const resp = await fetch('/api/grove/youtube/prewarm');
      if (!resp.ok) return;
      const data = await resp.json();
      const queries = data?.queries || {};   // { genre_key: query_string }
      const results = data?.results || {};   // { genre_key: [video, ...] }
      for (const [genre, videos] of Object.entries(results)) {
        if (!Array.isArray(videos) || videos.length === 0) continue;
        const query = queries[genre];
        if (query) musicSource.primeYouTubeCache(query, videos);
      }
      _prewarmed = true;
    } catch { /* silent — prewarm is best-effort */ }
    _prewarmInFlight = null;
  })();
}

function closeGrove() {
  const panel = $('grove-panel');
  if (!panel) return;
  // If the orb is detached, the panel MUST stay in the render tree so the
  // orb (a descendant) doesn't get evicted. Use soft-close instead of the
  // display:none path — .soft-closed sets visibility:hidden, which does not
  // cascade to descendants that declare visibility:visible (like .detached).
  if (orbDetach.isDetached()) {
    panel.classList.add('soft-closed');
  } else {
    panel.classList.remove('visible');
  }
  isOpen = false;
  // Reset discover view to closed state (use class, not inline style,
  // to avoid specificity conflict with .visible class on next open)
  const disc = $('grove-discover');
  const main = $('grove-main');
  if (disc) disc.classList.remove('visible');
  if (main) main.style.display = '';
}

function toggleGrove() {
  isOpen ? closeGrove() : openGrove();
}

// ---------------------------------------------------------------------------
// Atmosphere (themes)
// ---------------------------------------------------------------------------
function renderAtmoGrid() {
  const grid = $('grove-atmo-grid');
  if (!grid) return;
  const current = localStorage.getItem('augmentum-theme') || 'dark';

  grid.innerHTML = THEMES.map(t => {
    const meta = THEME_META[t];
    const active = t === current ? ' active' : '';
    // Each card is a miniature UI preview showing the theme's actual colors
    return `<button class="atmo-card${active}" data-theme="${escapeHtml(t)}" style="--atmo-accent:${meta.color}">
      <div class="atmo-card__preview" data-preview="${escapeHtml(t)}">
        <div class="atmo-card__bar"></div>
        <div class="atmo-card__lines">
          <span></span><span></span><span class="short"></span>
        </div>
        <div class="atmo-card__dot"></div>
      </div>
      <div class="atmo-card__info">
        <span class="atmo-card__name">${escapeHtml(meta.label)}</span>
        <span class="atmo-card__desc">${escapeHtml(meta.desc)}</span>
      </div>
    </button>`;
  }).join('');

  grid.addEventListener('click', e => {
    e.stopPropagation(); // Prevent click-outside handler from closing the panel
    const card = e.target.closest('[data-theme]');
    if (!card) return;
    const theme = card.dataset.theme;
    if (!THEMES.includes(theme)) return;

    localStorage.setItem('augmentum-theme', theme);
    document.documentElement.setAttribute('data-theme', theme);

    grid.querySelectorAll('.atmo-card').forEach(c => c.classList.toggle('active', c.dataset.theme === theme));

    document.dispatchEvent(new CustomEvent('augmentum:theme-changed', { detail: { theme } }));
  });
}

// ---------------------------------------------------------------------------
// Typography
// ---------------------------------------------------------------------------
function initTypography() {
  const wrap = $('grove-font-select');
  if (!wrap) return;

  const savedKey = localStorage.getItem('augmentum-typography') || 'system';
  _renderFontPicker(wrap, savedKey);

  // Size buttons — wired once here (not inside _renderFontPicker, which
  // re-runs on every font change and would stack duplicate listeners).
  $('grove-size-down')?.addEventListener('click', () => bumpScale(-SCALE_STEP));
  $('grove-size-up')?.addEventListener('click', () => bumpScale(SCALE_STEP));
  // grove.js loads before app.js, so app.state.textScale may still be the
  // default 1.0 when this runs. Read the persisted value directly so the
  // display starts on the user's saved scale.
  const savedScale = parseFloat(localStorage.getItem('augmentum-text-scale'));
  const initial = (savedScale && savedScale >= 0.7 && savedScale <= 1.4)
    ? savedScale
    : (app.state.textScale ?? 1.0);
  updateScaleDisplay(initial);

  // Soft typography — drops `text-transform: uppercase` + letter-spacing
  // on chrome labels. The FOUC-prevention script in index.html applies
  // the class to <html> before paint; we mirror state to the body so
  // existing CSS can continue using `body.soft-typography` selectors.
  const caseGroup = $('grove-typo-case');
  if (caseGroup) {
    const initialSoft = localStorage.getItem('augmentum-soft-typography') !== '0';
    document.body.classList.toggle('soft-typography', initialSoft);

    const reflect = (enabled) => {
      caseGroup.querySelectorAll('.grove-typo-case-btn').forEach((btn) => {
        const active = (btn.dataset.soft === '1') === enabled;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
    };
    reflect(initialSoft);

    caseGroup.addEventListener('click', (e) => {
      const btn = e.target.closest('.grove-typo-case-btn');
      if (!btn) return;
      const enabled = btn.dataset.soft === '1';
      localStorage.setItem('augmentum-soft-typography', enabled ? '1' : '0');
      document.documentElement.classList.toggle('soft-typography', enabled);
      document.body.classList.toggle('soft-typography', enabled);
      reflect(enabled);
      fetch('/api/config/ui', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ softTypography: enabled ? 'true' : 'false' }),
      }).catch(() => { /* best-effort; localStorage is the source of truth */ });
    });
  }
}

/** Custom font picker — each option rendered in its own font */
function _renderFontPicker(container, activeKey) {
  const current = FONTS.find(f => f.key === activeKey) || FONTS[0];

  // The visible "selected" display
  container.innerHTML = `
    <div class="grove-font-selected" id="grove-font-selected">
      <span class="grove-font-selected-label" style="font-family:${current.body}">${escapeHtml(current.label)}</span>
      <svg class="grove-select-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><polyline points="6 9 12 15 18 9"/></svg>
    </div>
    <div class="grove-font-dropdown" id="grove-font-dropdown">
      ${FONTS.map(f => `
        <button class="grove-font-option${f.key === activeKey ? ' active' : ''}" data-key="${escapeHtml(f.key)}" style="font-family:${f.body}">
          ${escapeHtml(f.label)}
        </button>
      `).join('')}
    </div>
  `;

  // Toggle dropdown
  const selected = container.querySelector('#grove-font-selected');
  const dropdown = container.querySelector('#grove-font-dropdown');

  selected.addEventListener('click', e => {
    e.stopPropagation();
    const opening = !dropdown.classList.contains('open');
    dropdown.classList.toggle('open');
    // When opening, preload every preset's Google Fonts so the option
    // labels actually render in their own typeface (otherwise the
    // dropdown previews in the system fallback font until you pick one).
    if (opening) {
      for (const f of FONTS) {
        if (f.key !== 'system') ensureTypographyFonts(f.key);
      }
    }
  });

  // Option click
  dropdown.addEventListener('click', e => {
    e.stopPropagation(); // Prevent click-outside from closing the Grove
    const opt = e.target.closest('.grove-font-option');
    if (!opt) return;
    const key = opt.dataset.key;
    applyTypographyPreset(key);
    _renderFontPicker(container, key);
    dropdown.classList.remove('open');
  });

  // Close on click outside (use capture to run before other handlers)
  function closeFontDropdown(e) {
    if (!container.contains(e.target)) dropdown.classList.remove('open');
  }
  // Remove previous listener if any, then add
  document.removeEventListener('click', container._closeFn);
  container._closeFn = closeFontDropdown;
  document.addEventListener('click', closeFontDropdown);
}

function applyTypographyPreset(key) {
  const preset = FONTS.find(f => f.key === key);
  if (!preset) return;

  // Dispatch event so app.js can apply (with lazy font loading + state sync)
  document.dispatchEvent(new CustomEvent('augmentum:typography-changed', { detail: { key } }));
}

/** Delegate to app.js's setTextScale — it handles clamping (0.7–1.4),
 *  state sync, localStorage, and server persistence. Then re-read the
 *  (possibly clamped) value to update our local display. */
function bumpScale(delta) {
  const current = app.state.textScale ?? 1.0;
  setTextScale(Math.round((current + delta) * 100) / 100);
  updateScaleDisplay(app.state.textScale ?? 1.0);
}

function updateScaleDisplay(scale) {
  const el = $('grove-size-val');
  // Percentage — matches the typography dropdown in settings and gives a
  // clean 70%→140% sequence in 10% steps (vs. Math.round(scale * 15) which
  // skipped values like 13, 16, 19 due to rounding).
  if (el) el.textContent = Math.round(scale * 100) + '%';
}

// ---------------------------------------------------------------------------
// Soundscape — Audio
// ---------------------------------------------------------------------------
function getAudio() {
  if (!audio) {
    audio = new Audio();
    // Note: do NOT set crossOrigin — internet radio streams don't serve
    // CORS headers, and we don't need programmatic access to the audio data.
    audio.volume = (getSettings().soundscapeVolume ?? 50) / 100;
    audio.addEventListener('ended', () => { playNext(); });
    audio.addEventListener('error', () => {
      isPlaying = false;
      updatePlayUI();
    });
    // Bus integration: claim on play, release on pause/stop. The duck
    // callback reuses the existing volume handle so user-set volume is
    // the baseline we duck below.
    audio.addEventListener('play',  () => {
      _groveBusHandle?.claim();
      MediaSessionBridge.setPlaybackState('grove-soundscape', 'playing');
    });
    audio.addEventListener('pause', () => {
      _groveBusHandle?.release();
      MediaSessionBridge.setPlaybackState('grove-soundscape', 'paused');
    });
  }
  return audio;
}

const _groveBusHandle = AudioBus.register({
  id: 'grove-soundscape',
  tier: 'ambient',
  // Grove soundscapes are music / lo-fi stations — tagged music so
  // the widget dances. Ambient tier means TTS still ducks over it.
  kind: 'music',
  duck: (level) => {
    if (!audio || _groveDuckBaseline !== null) return;
    _groveDuckBaseline = audio.volume;
    audio.volume = _groveDuckBaseline * level;
  },
  unduck: () => {
    if (!audio || _groveDuckBaseline === null) return;
    audio.volume = _groveDuckBaseline;
    _groveDuckBaseline = null;
  },
  // Music is an exclusive kind on the bus: when another music source
  // (YouTube orb, local music file) starts, the bus stops this stream
  // instead of letting two tracks stack.
  stop: () => stopPlayback(),
});
let _groveDuckBaseline = null;

// Lock-screen / headphone / Bluetooth AVRCP integration. Live radio:
// no duration → no scrubber. Skip handlers walk the favorites list
// the same way the in-app prev/next buttons do, so a steering-wheel
// next-track press behaves identically to tapping the on-screen chip.
MediaSessionBridge.register('grove-soundscape', {
  getMetadata: () => ({
    title:  currentStation?.name || 'Grove Radio',
    artist: currentStation?.desc || currentStation?.genre || 'Soundscape',
    album:  'Augmentum',
    // Stations don't ship cover art — fall through to the Augmentum
    // PWA icon, which keeps the brand on the lock screen.
    artworkUrl: null,
  }),
  // Live stream — omit getPosition so the bridge clears positionState
  // and the lock-screen scrubber stays hidden.
  handlers: {
    play:           () => togglePlayPause(),
    pause:          () => togglePlayPause(),
    stop:           () => stopPlayback(),
    previoustrack:  () => playPrev(),
    nexttrack:      () => playNext(),
  },
});

/**
 * Find a favourite station matching ``query`` and play it.
 *
 * Search order:
 *   1. Exact name match (case-insensitive)
 *   2. Genre substring match
 *   3. Name substring match
 *
 * Returns ``{ ok: true, station }`` on a hit, ``{ ok: false, reason }``
 * on miss. The architect's grove.play_matching primitive calls this
 * after the user says "play <query>"; the inference layer has already
 * narrowed the search to the user's history, but the final lookup
 * happens here against the live favourites list (which lives in
 * localStorage, not on the server).
 */
export function findAndPlayMatching(query) {
  const found = _findMatchingFavorite(query);
  if (!found.ok) return found;
  playStation(found.pick);
  return {
    ok: true,
    station: { id: found.pick.id, name: found.pick.name, genre: found.pick.genre },
  };
}

/** Match a favourite without playing it. Returns {ok, pick, rank} or
 *  {ok: false, reason}. Rank 0 = exact name; higher = looser. Delegates to
 *  the shared music-source matcher against the live favourites list. */
function _findMatchingFavorite(query) {
  return musicSource.matchFavorite(query, favorites);
}

/**
 * findAndPlayMatching, then fall through to station DISCOVERY on a
 * favourites miss. The station catalog (SomaFM + radio search) holds
 * hundreds of options — a genre ask should land on one of them, not
 * on a "no match" toast. Picks among the top few hits rather than
 * always the first, so repeat asks get variety.
 *
 * The architect's grove.play surface event sets ``discover_ok`` when
 * this fallback is welcome (it always is for voice asks; programmatic
 * exact-track plays omit it).
 */
export async function findAndPlayMatchingOrDiscover(query, { genreHints = [] } = {}) {
  // The resolution ladder itself now lives in music-source.js
  // (resolveSource) so every surface shares one tier spec. Grove supplies
  // its live favourites + recency rotation and PLAYS the returned
  // descriptor here — resolution decides WHAT, this decides HOW (radio
  // <audio>, media-player, or the ambient orb). Return shape preserved for
  // the architect's grove.play primitive.
  const res = await musicSource.resolveSource(query, {
    favorites,
    genreHints,
    rotation: _rotation,
    log: (...a) => console.info('[Grove]', ...a),
  });
  if (!res.ok) return res;
  return _playDescriptor(res.descriptor);
}

/** Play a normalized music-source descriptor on the right Grove surface and
 *  return the legacy { ok, station, source } shape. Radio → <audio> via
 *  playStation; local file → media-player; youtube → ambient orb. */
async function _playDescriptor(d) {
  if (!d) return { ok: false, reason: 'no-descriptor' };
  const legacy = (src) => ({
    ok: true,
    station: { id: d.id, name: d.name, genre: d.genre },
    source: src,
  });
  if (d.kind === 'station') {
    playStation(d.raw);
    return legacy(d.source === 'discover' ? 'discover' : 'favorites');
  }
  if (d.kind === 'file') {
    const mp = await import('./media-player.js');
    await mp.play(d.id);
    return legacy('files');
  }
  if (d.kind === 'youtube') {
    ambient.loadVideo(d.raw);
    return legacy('youtube');
  }
  return { ok: false, reason: 'unknown-kind' };
}

function playStation(station) {
  if (!station || !station.url) return;
  currentStation = station;
  // Refresh lock-screen widget so a station-switch via the in-app
  // chip updates the now-playing tile immediately (instead of waiting
  // for the next AudioBus event).
  MediaSessionBridge.notifyMetadataChanged('grove-soundscape');

  // Track index in favorites
  favIndex = favorites.findIndex(f => f.id === station.id);

  const a = getAudio();
  console.debug('[Grove] Playing:', station.name, station.url);
  a.src = station.url;
  a.play().then(() => {
    isPlaying = true;
    updatePlayUI();
    recordLastPlayed({ type: 'radio', name: station.name || 'radio station' });
    // Companion presence: grove stations live in localStorage only, so
    // without this report she has no idea what's playing.
    import('./architect-observer.js')
      .then(m => m.reportAttention('surface.audio.station_playing', {
        label: station.name || 'radio station',
        kind: station.genre ? `radio · ${station.genre}` : 'radio',
        ref: String(station.id || ''),
      }))
      .catch(() => {});
    // Playback state flips the favorite-current `when` guard — re-sync
    // the agent catalog so app.act sees it live.
    import('./command-palette.js')
      .then(m => m.refreshAgentCatalog?.())
      .catch(() => {});
  }).catch(err => {
    console.warn('[Grove] Playback failed:', err.message, '— URL:', station.url);
    isPlaying = false;
    updatePlayUI();
  });

  // Track in recents (most recent first) and persist
  addToRecent(station);
  setSetting('soundscapeLastStation', station);

  // Auto-add to favorites if not present. Tagged `auto` so matching /
  // future hygiene can tell a deliberate favourite from played-once
  // residue — auto-adds are why genre asks kept landing on radio.
  if (favIndex === -1) {
    favorites.push({ ...station, auto: true });
    favIndex = favorites.length - 1;
    saveFavorites();
    renderStationChips();
  }
}

function stopPlayback() {
  if (audio) {
    audio.pause();
    audio.removeAttribute('src');
    audio.load();
  }
  isPlaying = false;
  updatePlayUI();
  import('./command-palette.js')
    .then(m => m.refreshAgentCatalog?.())
    .catch(() => {});
}

function togglePlayPause() {
  if (!currentStation) {
    // Play first favorite if available
    if (favorites.length > 0) {
      playStation(favorites[0]);
    }
    return;
  }
  if (isPlaying) {
    audio?.pause();
    isPlaying = false;
  } else {
    const a = getAudio();
    if (!a.src || a.src === location.href) {
      a.src = currentStation.url;
    }
    a.play().then(() => { isPlaying = true; updatePlayUI(); }).catch(() => {});
  }
  updatePlayUI();
}

function toggleMute() {
  if (isPlaying) {
    stopPlayback();
  } else if (currentStation) {
    playStation(currentStation);
  } else if (favorites.length > 0) {
    playStation(favorites[0]);
  }
}

function playPrev() {
  if (favorites.length === 0) return;
  favIndex = (favIndex - 1 + favorites.length) % favorites.length;
  playStation(favorites[favIndex]);
}

function playNext() {
  if (favorites.length === 0) return;
  favIndex = (favIndex + 1) % favorites.length;
  playStation(favorites[favIndex]);
}

function updatePlayUI() {
  // Play/pause button icon
  const playBtn = $('grove-play');
  if (playBtn) playBtn.innerHTML = isPlaying ? PAUSE_SVG : PLAY_SVG;

  // Now-playing card: always visible, toggle .playing class for visual state
  const card = $('grove-now-playing');
  if (card) card.classList.toggle('playing', isPlaying);

  // Stream info
  const title = $('grove-stream-title');
  const desc = $('grove-stream-desc');
  if (currentStation) {
    if (title) title.textContent = currentStation.name || '\u2014';
    if (desc) desc.textContent = currentStation.desc || currentStation.genre || '';
  } else {
    if (title) title.textContent = 'No station selected';
    if (desc) desc.textContent = 'Pick a station below to begin';
  }

  // Live badge
  const meta = $('grove-live-meta');
  if (meta) {
    meta.innerHTML = isPlaying
      ? '<span class="grove-live-badge"><span class="grove-live-dot"></span> LIVE</span>'
      : '';
  }

  // Header playing indicator
  const groveBtn = $('grove-btn');
  if (groveBtn) groveBtn.classList.toggle('playing', isPlaying);

  // Cast button — only enabled when there's a station to cast.
  if (_castBtn) {
    _castBtn.disabled = !currentStation;
    _castBtn.title = currentStation
      ? `Cast "${currentStation.name || 'station'}" to TV`
      : 'Pick a station first';
  }

  // Station chip active states
  document.querySelectorAll('.grove-station-chip').forEach(chip => {
    chip.classList.toggle('active', currentStation && chip.dataset.stationId === currentStation.id);
  });
}

// ---------------------------------------------------------------------------
// Soundscape — Volume
// ---------------------------------------------------------------------------
function initVolume() {
  const slider = $('grove-vol-slider');
  const pct = $('grove-vol-pct');
  if (!slider) return;

  const saved = getSettings().soundscapeVolume ?? 50;
  slider.value = saved;
  if (pct) pct.textContent = saved + '%';

  slider.addEventListener('input', () => {
    const val = parseInt(slider.value, 10);
    if (pct) pct.textContent = val + '%';
    if (audio) audio.volume = val / 100;
    setSetting('soundscapeVolume', val);
  });
}

// ---------------------------------------------------------------------------
// Soundscape — Station chips (recent + recommended)
// ---------------------------------------------------------------------------

// Recently played stations (most recent first), persisted to localStorage
const RECENT_KEY = 'augmentum-soundscape-recent';
const MAX_RECENT = 8;
const MAX_CHIPS = 5;

function getRecentStations() {
  try { return JSON.parse(localStorage.getItem(RECENT_KEY)) || []; } catch { return []; }
}

function addToRecent(station) {
  if (!station || !station.id) return;
  let recent = getRecentStations();
  // Remove if already present, then prepend
  recent = recent.filter(s => s.id !== station.id);
  recent.unshift(station);
  if (recent.length > MAX_RECENT) recent = recent.slice(0, MAX_RECENT);
  localStorage.setItem(RECENT_KEY, JSON.stringify(recent));
}

function renderStationChips() {
  const container = $('grove-station-chips');
  if (!container) return;

  // Build chip list: recent stations first, then fill from favorites (recommended)
  const recent = getRecentStations();
  const seen = new Set();
  const chips = [];

  // Recent first
  for (const s of recent) {
    if (chips.length >= MAX_CHIPS) break;
    if (!seen.has(s.id)) { seen.add(s.id); chips.push({ ...s, isRecent: true }); }
  }

  // Fill remaining with favorites/recommended
  for (const s of favorites) {
    if (chips.length >= MAX_CHIPS) break;
    if (!seen.has(s.id)) { seen.add(s.id); chips.push(s); }
  }

  let html = chips.map(s => {
    const active = currentStation && currentStation.id === s.id ? ' active' : '';
    return `<button class="grove-station-chip${active}" data-station-id="${escapeHtml(s.id)}" title="${escapeHtml(s.desc || s.genre || '')}">${escapeHtml(s.name)}</button>`;
  }).join('');

  html += `<button class="grove-station-chip discover-more" id="grove-discover-btn">Discover more &rarr;</button>`;
  container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Soundscape — Favorites persistence
// ---------------------------------------------------------------------------
async function loadFavorites() {
  // Server-backed load + SomaFM seed both live in the shared module now.
  favorites = await musicSource.loadFavorites();
}

async function saveFavorites() {
  await musicSource.saveFavorites(favorites);
}

// ---------------------------------------------------------------------------
// Discover view
// ---------------------------------------------------------------------------
function openDiscover() {
  const disc = $('grove-discover');
  const main = $('grove-main');
  if (disc) disc.classList.add('visible');
  if (main) main.style.display = 'none';
  renderGenreChips();
  searchStations('');
}

function closeDiscover() {
  const disc = $('grove-discover');
  const main = $('grove-main');
  if (disc) disc.classList.remove('visible');
  if (main) main.style.display = '';
}

let _speculativeTimeout = null;

function initDiscover() {
  // Back button
  $('grove-discover-back')?.addEventListener('click', closeDiscover);

  // Source tabs
  const tabsContainer = $('grove-source-tabs');
  tabsContainer?.addEventListener('click', e => {
    const tab = e.target.closest('.grove-source-tab');
    if (!tab) return;
    discoverSource = tab.dataset.source || 'all';
    tabsContainer.querySelectorAll('.grove-source-tab').forEach(t => t.classList.toggle('active', t === tab));
    searchStations($('grove-discover-input')?.value || '');
  });

  // Search input — 300ms debounce for the visible search, plus a 150ms
  // speculative fetch that silently warms cache. By the time the debounce
  // fires, the cache is usually already warm → results render as a cache
  // hit, not a loading spinner.
  $('grove-discover-input')?.addEventListener('input', e => {
    const value = e.target.value;
    clearTimeout(searchTimeout);
    clearTimeout(_speculativeTimeout);

    // Speculative fetch (YouTube only — stations are fast enough already)
    if (discoverSource === 'youtube' && value.length >= 2) {
      _speculativeTimeout = setTimeout(() => {
        _fetchYouTube(_resolveYtQuery(value));  // silent cache warm
      }, 150);
    }

    searchTimeout = setTimeout(() => searchStations(value), 300);
  });

  // Delegated click handlers (wired once, not per-render)
  $('grove-discover-results')?.addEventListener('click', _handleDiscoverClick);
  $('grove-genre-chips')?.addEventListener('click', _handleGenreClick);

  // Hover-prefetch on genre chips — by the time the user clicks, the
  // cache is warm. Uses mouseenter so it fires on first hover, not on
  // every mousemove.
  $('grove-genre-chips')?.addEventListener('mouseenter', e => {
    const chip = e.target.closest?.('.grove-filter-chip');
    if (!chip || discoverSource !== 'youtube') return;
    const genre = chip.dataset.genre;
    if (!genre) return;
    const q = _YT_GENRE_QUERIES[genre.toLowerCase()] || `${genre} music`;
    _fetchYouTube(q);  // silent — just seeds cache
  }, true);  // capture, because mouseenter doesn't bubble from children
}

function renderGenreChips() {
  const container = $('grove-genre-chips');
  if (!container) return;

  container.innerHTML = GENRE_CHIPS.map(g => {
    const active = discoverGenre.toLowerCase() === g.toLowerCase() ? ' active' : '';
    return `<button class="grove-filter-chip${active}" data-genre="${escapeHtml(g)}">${escapeHtml(g)}</button>`;
  }).join('');
}

function _handleGenreClick(e) {
    const chip = e.target.closest('.grove-filter-chip');
    if (!chip) return;
    const genre = chip.dataset.genre;
    discoverGenre = discoverGenre === genre ? '' : genre;
    const container = $('grove-genre-chips');
    if (container) container.querySelectorAll('.grove-filter-chip').forEach(c =>
      c.classList.toggle('active', c.dataset.genre === discoverGenre)
    );
    searchStations($('grove-discover-input')?.value || '');
}

async function searchStations(query) {
  const results = $('grove-discover-results');
  if (!results) return;

  results.innerHTML = '<div class="grove-discover-loading">Searching stations\u2026</div>';

  const source = discoverSource;
  if (source === 'youtube') return _searchYouTube(query);

  try {
    // The fetch + SomaFM client-filter live in the shared module now; grove
    // keeps only the discover-panel rendering.
    const stations = await musicSource.searchStations({
      q: query,
      genre: discoverGenre,
      source,
      limit: 20,
    });
    renderDiscoverResults(stations);
  } catch {
    results.innerHTML = '<div class="grove-discover-empty">Could not load stations.</div>';
  }
}

// YT search cache + genre-query map now live in music-source.js so every
// surface shares one warmed cache. These thin wrappers keep grove's call
// sites unchanged.
const _YT_GENRE_QUERIES = musicSource.YT_GENRE_QUERIES;

/** Resolve user-facing query or active genre to the canonical YT query. */
function _resolveYtQuery(query) {
  if (query) return query;
  if (discoverGenre) {
    return _YT_GENRE_QUERIES[discoverGenre.toLowerCase()] || `${discoverGenre} music`;
  }
  return 'ambient music';
}

/** Fetch YT results for `query` via the shared cache/searcher. Returns the
 *  video array (possibly cached) or null on failure. */
async function _fetchYouTube(query) {
  return musicSource.searchYouTube(query, { limit: 12 });
}

async function _searchYouTube(query) {
  const results = $('grove-discover-results');
  if (!results) return;

  const q = _resolveYtQuery(query);

  // Cache hit — instant render. This covers: prewarm hits, hover-prefetch
  // hits, speculative-search hits, and plain repeat-query hits.
  const cached = musicSource.peekYouTubeCache(q);
  if (cached) {
    _renderYouTubeResults(cached);
    return;
  }

  results.innerHTML = '<div class="grove-discover-loading">Searching YouTube\u2026</div>';
  const videos = await _fetchYouTube(q);
  if (videos === null) {
    results.innerHTML = '<div class="grove-discover-empty">Could not search YouTube.</div>';
    return;
  }
  _renderYouTubeResults(videos || []);
}

function _renderYouTubeResults(videos) {
  const results = $('grove-discover-results');
  if (!results) return;

  if (!videos || videos.length === 0) {
    results.innerHTML = `<div class="grove-discover-empty">
      <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.5" style="opacity:0.4;margin-bottom:6px"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <div>No ambient videos found</div>
      <div style="font-size:9px;margin-top:2px;opacity:0.6">Try a different search or genre</div>
    </div>`;
    return;
  }

  results.innerHTML = videos.map((v, i) => {
    const isFav = ambient.isFavorite(v.videoId);
    const badge = v.isLivestream
      ? '<span class="grove-station-row-source" style="color:#34d399">LIVE</span>'
      : `<span class="grove-station-row-source">${escapeHtml(v.duration || '')}</span>`;
    return `<button class="grove-station-row grove-yt-result" data-idx="${i}">
      <div class="grove-yt-thumb"><img src="${escapeHtml(v.thumbnail)}" alt="" loading="lazy"></div>
      <div class="grove-station-row-info">
        <span class="grove-sr-name">${escapeHtml(v.title)}</span>
        <span class="grove-sr-desc">${escapeHtml(v.channel)}</span>
      </div>
      ${badge}
      <span class="grove-station-row-fav grove-yt-fav${isFav ? ' active' : ''}" title="Favorite" data-vid="${escapeHtml(v.videoId)}">\u2605</span>
      <div class="grove-sr-play"><svg viewBox="0 0 24 24" fill="currentColor" width="11" height="11"><path d="M8 5v14l11-7z"/></svg></div>
    </button>`;
  }).join('');

  // Store video data for click handler
  results._ytVideos = videos;
}

// Store station data by index to avoid JSON-in-attributes issues
let _discoverStations = [];

// Station icon fallback — first letter in a colored circle
const _STATION_COLORS = [
  '#34d399', '#38bdf8', '#a78bfa', '#f59e0b', '#f472b6',
  '#fb923c', '#60a5fa', '#4ade80', '#c084fc', '#fbbf24',
];

function _stationIcon(station) {
  const letter = (station.name || '?')[0].toUpperCase();
  const colorIdx = (station.name || '').length % _STATION_COLORS.length;
  const color = _STATION_COLORS[colorIdx];
  return `<div class="grove-station-art" style="background:color-mix(in srgb, ${color} 12%, var(--surface));color:${color}">${escapeHtml(letter)}</div>`;
}

function renderDiscoverResults(stations) {
  const results = $('grove-discover-results');
  if (!results) return;

  if (!stations || stations.length === 0) {
    results.innerHTML = `<div class="grove-discover-empty">
      <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.5" style="opacity:0.4;margin-bottom:6px"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <div>No stations found</div>
      <div style="font-size:9px;margin-top:2px;opacity:0.6">Try a different search or genre</div>
    </div>`;
    _discoverStations = [];
    return;
  }

  _discoverStations = stations;

  results.innerHTML = stations.map((s, i) => {
    const isFav = favorites.some(f => f.id === s.id);
    const isActive = currentStation && currentStation.id === s.id;
    const sourceBadge = s.source === 'somafm'
      ? '<span class="grove-station-row-source somafm">SomaFM</span>'
      : '<span class="grove-station-row-source">' + escapeHtml(s.source || '') + '</span>';
    return `<button class="grove-station-row${isActive ? ' playing' : ''}" data-idx="${i}">
      ${_stationIcon(s)}
      <div class="grove-station-row-info">
        <span class="grove-sr-name">${escapeHtml(s.name)}</span>
        <span class="grove-sr-desc">${escapeHtml(s.desc || s.genre || '')}</span>
      </div>
      ${sourceBadge}
      ${isFav ? '<span class="grove-station-row-fav" title="Favorite">\u2605</span>' : ''}
      <div class="grove-sr-play"><svg viewBox="0 0 24 24" fill="currentColor" width="11" height="11"><path d="M8 5v14l11-7z"/></svg></div>
    </button>`;
  }).join('');
}

// Single delegated click handler — wired once in initDiscover()
function _handleDiscoverClick(e) {
  // YouTube favorite toggle
  const favEl = e.target.closest('.grove-yt-fav');
  if (favEl) {
    e.stopPropagation();
    const vid = favEl.dataset.vid;
    const results = $('grove-discover-results');
    if (ambient.isFavorite(vid)) {
      ambient.removeFavorite(vid);
      favEl.classList.remove('active');
    } else {
      const videos = results?._ytVideos || [];
      const v = videos.find(x => x.videoId === vid);
      if (v) { ambient.addFavorite(v); favEl.classList.add('active'); }
    }
    return;
  }

  // YouTube result click → load into orb
  const ytRow = e.target.closest('.grove-yt-result');
  if (ytRow) {
    const idx = parseInt(ytRow.dataset.idx, 10);
    const results = $('grove-discover-results');
    const videos = results?._ytVideos || [];
    if (videos[idx]) {
      ambient.loadVideo(videos[idx]);
      closeDiscover();
    }
    return;
  }

  const row = e.target.closest('.grove-station-row');
  if (!row) return;
  const idx = parseInt(row.dataset.idx, 10);
  const station = _discoverStations[idx];
  if (!station) { console.warn('[Grove] No station at index', idx); return; }

  playStation(station);
  renderStationChips();

  // Update row active states
  const results = $('grove-discover-results');
  if (results) results.querySelectorAll('.grove-station-row').forEach(r => r.classList.remove('active'));
  row.classList.add('active');
}

// Vitals section removed — resource info already visible in header bar

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
function initKeyboard() {
  document.addEventListener('keydown', e => {
    // Ctrl+\ — toggle grove
    if (e.ctrlKey && e.key === '\\') {
      e.preventDefault();
      toggleGrove();
      return;
    }
    // Ctrl+M — mute/unmute
    if (e.ctrlKey && e.key === 'm') {
      e.preventDefault();
      toggleMute();
      return;
    }
    // Ctrl+T — cycle theme
    if (e.ctrlKey && e.key === 't') {
      // Only if not in an input/textarea
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;
      e.preventDefault();
      cycleTheme();
      return;
    }
    // Escape closes grove
    if (e.key === 'Escape' && isOpen) {
      closeGrove();
    }
  });
}

function cycleTheme() {
  const current = localStorage.getItem('augmentum-theme') || 'dark';
  const idx = THEMES.indexOf(current);
  const next = THEMES[(idx + 1) % THEMES.length];
  localStorage.setItem('augmentum-theme', next);
  document.documentElement.setAttribute('data-theme', next);
  document.dispatchEvent(new CustomEvent('augmentum:theme-changed', { detail: { theme: next } }));
  renderAtmoGrid();
}

// ---------------------------------------------------------------------------
// Click outside to close
// ---------------------------------------------------------------------------
function initClickOutside() {
  document.addEventListener('click', e => {
    if (!isOpen) return;
    const panel = $('grove-panel');
    const btn = $('grove-btn');
    if (!panel || !btn) return;
    // Use composedPath, not .contains(e.target). Some in-panel handlers
    // (e.g. ambient favorite chip → loadVideo → _renderChips) detach the
    // clicked element from the DOM before this bubbling listener runs,
    // which would make .contains() report "outside" and close the grove.
    const path = e.composedPath();
    if (path.includes(panel) || path.includes(btn)) return;
    closeGrove();
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  // Wire grove button
  $('grove-btn')?.addEventListener('click', e => {
    e.stopPropagation();
    toggleGrove();
  });

  // Playback controls
  $('grove-play')?.addEventListener('click', togglePlayPause);
  $('grove-prev')?.addEventListener('click', playPrev);
  $('grove-next')?.addEventListener('click', playNext);

  // Cast-to-TV — route the current radio stream to a paired receiver.
  // The cast-audio TV surface renders cover art + title + VU meter
  // even though the audio source is an arbitrary HTTP stream URL.
  const controls = document.querySelector('#grove-now-playing .grove-stream-controls');
  if (controls) {
    _castBtn = mountCastButton({
      capability: 'media.audio_play@1',
      size: 'sm',
      className: 'grove-stream-cast',
      title: 'Cast to TV',
      getContent: () => {
        if (!currentStation || !currentStation.url) return null;
        const sourceLabel = ({
          radiobrowser: 'Internet Radio',
          soma: 'SomaFM',
          fma: 'Free Music Archive',
        })[currentStation.source] || 'Radio';
        return {
          streamUrl: currentStation.url,
          title:     currentStation.name || 'Radio',
          author:    currentStation.desc || currentStation.genre || '',
          posterUrl: currentStation.favicon || '',
          source:    `Grove · ${sourceLabel}`,
          contentKey: `radio:${currentStation.id}`,
          metadata: {
            source: 'grove_soundscape',
            genre:  currentStation.genre || '',
            origin: currentStation.source || '',
          },
        };
      },
    });
    controls.appendChild(_castBtn);
  }

  // Station chips — delegated click handler (wired once)
  $('grove-station-chips')?.addEventListener('click', e => {
    const chip = e.target.closest('.grove-station-chip');
    if (!chip) return;
    if (chip.id === 'grove-discover-btn') { openDiscover(); return; }
    const sid = chip.dataset.stationId;
    const station = favorites.find(f => f.id === sid);
    if (station) {
      if (currentStation && currentStation.id === sid && isPlaying) stopPlayback();
      else playStation(station);
    }
  });


  renderAtmoGrid();
  initTypography();
  initVolume();
  initDiscover();
  initKeyboard();
  initClickOutside();

  // Load favorites + last station
  await loadFavorites();
  await ambient.init();
  await playlist.init();
  orbDetach.init();
  renderStationChips();

  // Restore last station (but don't auto-play)
  const last = getSettings().soundscapeLastStation;
  if (last && last.url) {
    currentStation = last;
    favIndex = favorites.findIndex(f => f.id === last.id);
    updatePlayUI();
  }

  // App menu: the companion can press this via app.act ("love this
  // one, save it"). Promotes the current station from auto-added
  // residue to a deliberate favorite — the auto tag is why genre asks
  // kept landing on radio, so clearing it IS the meaningful outcome.
  import('./command-palette.js').then(({ registerCommand }) => {
    registerCommand({
      id: 'grove.favorite-current',
      label: 'Favorite current station',
      group: 'Grove',
      keywords: 'favorite save like love keep station music song',
      when: () => !!currentStation && isPlaying,
      agent: {
        description: 'Add the currently playing station to favorites',
        speak: "Saved — it's in your favorites.",
      },
      run: () => {
        if (!currentStation) return;
        const i = favorites.findIndex(f => f.id === currentStation.id);
        if (i >= 0) delete favorites[i].auto;
        else favorites.push({ ...currentStation });
        favIndex = favorites.findIndex(f => f.id === currentStation.id);
        saveFavorites();
        renderStationChips();
        showToast('Added to favorites', 'success');
      },
    });
  }).catch(() => {});

  // Listen for external theme changes to keep atmo grid in sync
  document.addEventListener('augmentum:theme-changed', () => renderAtmoGrid());

  // Ambient orb "+" click → open Discover with YouTube tab
  document.addEventListener('grove:open-discover-youtube', () => {
    openDiscover();
    discoverSource = 'youtube';
    const tabs = $('grove-source-tabs');
    if (tabs) {
      tabs.querySelectorAll('.grove-source-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.source === 'youtube');
      });
    }
    searchStations('');
  });

  // Orb detach → close grove panel so the orb has room in the chat view.
  // Offset the close by 100ms so the user sees the orb "lift off" *first*,
  // then the panel hides behind it — reads as one continuous motion.
  document.addEventListener('grove:orb-detached', (e) => {
    // Don't close on restoration (page reload with orb already floating)
    if (e.detail?.restored) return;
    if (!isOpen) return;
    setTimeout(() => closeGrove(), 100);
  });

  // Orb redock → panel should fully close (no longer needs soft-close
  // mask since the orb is back in its slot).
  document.addEventListener('grove:orb-redocked', () => {
    const panel = $('grove-panel');
    if (!panel) return;
    panel.classList.remove('soft-closed');
    // If the panel was being kept artificially visible for the orb, close
    // it properly now that the orb is home (unless user explicitly has it
    // open — track via isOpen).
    if (!isOpen) panel.classList.remove('visible');
  });

  // Dismiss (×) on the floating orb → pause playback. redock has already
  // happened by the time this fires.
  document.addEventListener('grove:orb-dismiss-requested', () => {
    void ambient.dismissCurrent?.();
  });

  _maybeOfferResume();
}

/**
 * Offer to resume whatever grove source (radio / ambient orb) the user had
 * going before the refresh. Shows a persistent toast; the action button
 * resumes, the built-in X dismisses permanently.
 *
 * Preconditions checked before showing:
 *   - a last-played entry exists (someone actually listened to something)
 *   - the user hasn't permanently dismissed this prompt before
 *   - the relevant source has its state restored (currentStation for radio,
 *     ambient.getState().currentVideo for orb) so the Play action has
 *     something to act on
 */
function _maybeOfferResume() {
  if (isPromptDismissed()) return;
  const last = getLastPlayed();
  if (!last) return;
  // Dedupe with media-resume.js's parallel toast. If the user's
  // most recent play was an Emby video / audiobook (stored in the
  // media-resume registry), let that toast carry the prompt — two
  // near-identical "Resume X?" toasts side-by-side is what made
  // the user click Play on the wrong title (Grove song) while
  // looking at the other (Emby music video).
  const mediaLast = _mediaGetLastPlayed();
  if (mediaLast && (mediaLast.t || 0) > (last.t || 0)) return;

  let resume;
  if (last.type === 'radio') {
    if (!currentStation) return; // nothing to resume
    resume = () => playStation(currentStation);
  } else if (last.type === 'ambient') {
    const saved = ambient.getState?.().currentVideo;
    if (!saved || !saved.videoId) return;
    resume = () => ambient.loadVideo(saved);
  } else {
    return;
  }

  // showToast returns the toast id; we grab the element afterward to layer
  // a permanent-dismiss listener on the built-in X. showToast's own dismiss
  // listener still runs and closes the toast; ours just writes the
  // localStorage flag in addition, so the prompt never returns.
  const id = showToast(`Resume ${last.name}?`, 'info', 0, {
    description: last.type === 'radio' ? 'Grove radio' : 'Ambient orb',
    action: { label: 'Play', onClick: resume },
    dismissible: true,
  });

  // Attach the "don't ask again" behavior to the X. Run on the next frame
  // so the toast is in the DOM.
  requestAnimationFrame(() => {
    const el = document.querySelector(`.toast[data-id="${id}"]`);
    el?.querySelector('.toast-dismiss')?.addEventListener('click', markPromptDismissed);
  });
}

document.addEventListener('DOMContentLoaded', init);

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------
export { closeGrove, openGrove, toggleMute };

// Companion transport + volume surface (intent-action-router's
// media.transport / media.volume legs). pauseGrove / nextGroveTrack /
// previousGroveTrack are the names the router has dispatched since the
// transport channel landed — they just never existed here, so the
// typeof-guard silently no-opped the entire Grove leg. Wiring program
// Phase 1 (2026-06-12) closes that and adds the volume pair.
export function isGrovePlaying() {
  return !!(audio && !audio.paused);
}

export function pauseGrove() {
  if (isGrovePlaying()) stopPlayback();
}

export function nextGroveTrack() {
  playNext();
}

export function previousGroveTrack() {
  playPrev();
}

export function setGroveVolume(pct) {
  const val = Math.max(0, Math.min(100, Math.round(Number(pct) || 0)));
  if (audio) audio.volume = val / 100;
  setSetting('soundscapeVolume', val);
  const slider = $('grove-vol-slider');
  if (slider) slider.value = val;
  const pctEl = $('grove-vol-pct');
  if (pctEl) pctEl.textContent = val + '%';
  return true;
}

export function adjustGroveVolume(deltaPct) {
  const cur = audio
    ? audio.volume * 100
    : (getSettings().soundscapeVolume ?? 50);
  return setGroveVolume(cur + (Number(deltaPct) || 0));
}

export function setGroveMuted(muted) {
  if (!audio) return false;
  audio.muted = !!muted;
  return true;
}
