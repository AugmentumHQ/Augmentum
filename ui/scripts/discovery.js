/**
 * Discovery Engine — For You tab with History view.
 * Rich history items (favicons, video thumbnails), infinite scroll,
 * junk title filtering, dynamic recommendations.
 */

const escapeHtml = window.escapeHtml || ((s) => String(s).replace(/[&<>"`]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','`':'&#96;'}[c]||c)));

// ---------------------------------------------------------------------------
// Junk title filter — client-side safety net for pre-existing bad entries
// ---------------------------------------------------------------------------
const _JUNK_RE = /^(page not found|404|403|access denied|just a moment|attention required|error \d{3}|untitled|loading\.{0,3}|please enable|checking your? browser|blocked|unauthorized|security check|captcha)/i;
function _isJunkTitle(title) {
  if (!title || title.trim().length < 4) return true;
  return _JUNK_RE.test(title.trim());
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let _activeView = 'history';
let _historyPage = 1;
let _historyQuery = '';
let _historyItems = [];
let _hasMore = false;
let _loading = false;
let _container = null;

/**
 * Cap on how many history items we keep in memory at once. Power users
 * accumulate thousands of visited URLs; without this cap, _historyItems
 * grows unbounded across infinite-scroll pages and _renderHistory (which
 * rebuilds the whole list via innerHTML on every page load) gets
 * quadratically slower. When the cap is hit we drop the oldest items —
 * those pages are still on the server and can be reached by searching.
 */
const _HISTORY_MAX_ITEMS = 1000;
let _searchDebounce = null;
let _scrollObserver = null;

// ---------------------------------------------------------------------------
// For You refresh state — polling-aware, exclusion-based.
// ---------------------------------------------------------------------------
const _ZONES = ['core', 'frontier', 'adjacent', 'fresh'];
const _emptyZones = () => ({ core: [], frontier: [], adjacent: [], fresh: [] });
const _POLL_CADENCE_MS = 3 * 60 * 1000;
const _MAX_EXCLUDE = 180;           // keep query param + in-memory Set bounded
const _SEEN_URLS_CAP = 360;         // 2x _MAX_EXCLUDE — FIFO eviction beyond this
const _PENDING_PER_ZONE_CAP = 30;   // protect against long idle tabs

// FIFO-ordered URLs paired with a Set for O(1) contains. On overflow we
// drop from the front; lost exclusions just mean the recommender *might*
// re-show them, which is acceptable given the query-param cap anyway.
const _seenOrder = [];
const _seenUrls = new Set();
let _pendingByZone = _emptyZones();
let _pollTimer = null;
let _pollInFlight = false;
let _lastPollAt = 0;

// Resume-listening row. `null` = not yet loaded; `[]` = loaded, hide strip.
let _resumeItems = null;

// ---------------------------------------------------------------------------
// Sticky last-render cache — SWR between tab switches.
//
// On the first ever fetch the user pays the full ~600-900ms wait while
// SearXNG aggregates four zones. After that, switching away from
// Discovery (to Browse / Notes) and back used to re-pay that cost
// every time. The cache below holds the last successful zones payload
// so subsequent shows render instantly from memory while a fresh fetch
// runs in the background. If the new payload differs, we re-render
// (cards have stable identity by URL, so card-level diffing isn't
// required — replacing the bento is fast and the staggered entrance
// animation re-fires for the new shape, which reads as "new picks").
//
// Tunables:
//   FRESH window — render cached without any indicator; a refresh in
//   <60s would be noise, so we treat the cache as authoritative.
//   USABLE window — render cached AND show a quiet refresh hint so
//   the user knows we're checking; if the fetch returns the same set
//   we hide the hint without a re-render.
//   STALE window — beyond this, ignore cache and show skeletons. The
//   user's intent was likely to see "what's new" and stale data
//   masks that.
// ---------------------------------------------------------------------------
let _lastLibraryCache = null; // { zones, urlSet, fetchedAt } — library
// Web-feed recs hold for 24 h, persisted to localStorage so a page reload
// doesn't wipe them. The /for-you fan-out is expensive (one SearXNG call
// per core cluster) and the underlying signal — your interest clusters —
// turns over slowly. A day-long window means at most one regeneration per
// calendar day per browser. Within the window: render cache, no refetch.
const _CACHE_TTL_MS = 24 * 60 * 60 * 1000;
// Per-user storage key — appended with `::u:<userId>` so Profile A's
// For-You recommendations don't surface under Profile B on the next
// reload (the old global key leaked across profiles).
const _CACHE_STORAGE_KEY_BASE = 'augmentum_for_you_cache_v1';

function _cacheStorageKey() {
  // discovery.js is a classic <script> (not an ES module), so it can't
  // import getCurrentUser() from auth.js directly. auth.js publishes
  // the active user on window.__augmentumUser whenever it changes;
  // null = not yet logged in, in which case we skip storage entirely
  // rather than fall back to a global key.
  const uid = (window.__augmentumUser && window.__augmentumUser.id) || null;
  return uid ? `${_CACHE_STORAGE_KEY_BASE}::u:${uid}` : null;
}

function _loadRecsCacheFromStorage() {
  const key = _cacheStorageKey();
  if (!key) return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed.fetchedAt !== 'number' || !parsed.zones) return null;
    if (Date.now() - parsed.fetchedAt > _CACHE_TTL_MS) {
      localStorage.removeItem(key);
      return null;
    }
    parsed.urlSet = _zoneUrlSet(parsed.zones);
    return parsed;
  } catch { return null; }
}

function _saveRecsCacheToStorage(cache) {
  const key = _cacheStorageKey();
  if (!key) return;
  try {
    if (!cache || !cache.zones) {
      localStorage.removeItem(key);
      return;
    }
    // urlSet (Set) isn't JSON-serializable; rebuild on load.
    localStorage.setItem(
      key,
      JSON.stringify({ zones: cache.zones, fetchedAt: cache.fetchedAt }),
    );
  } catch { /* localStorage full or unavailable — non-critical */ }
}

// Hydrated lazily inside init() once the current user is known.
// Eager hydration at module load (the old behaviour) ran before auth
// had populated window.__augmentumUser, so the cache was always loaded
// from the unscoped legacy key and leaked across profiles.
let _lastRecsCache = null;   // { zones, urlSet, fetchedAt } — web feeds, 24h persisted

// Library zones get a longer cache window than web feeds — local SQLite
// data turns over slowly (a new audiobook is rare; a new HN post is
// minute-by-minute). 30s fresh / 10min usable / 30min stale lets revisits
// feel instant without showing genuinely stale "in progress" state.
const _LIB_CACHE_FRESH_MS = 30 * 1000;
const _LIB_CACHE_USABLE_MS = 10 * 60 * 1000;
const _LIB_CACHE_STALE_MS = 30 * 60 * 1000;

// Opportunistic prefetch — when the user lands on the History view (the
// default first view), kick off the For You fetches in the background
// so the cache is warm by the time they switch. Without this, the
// first switch to For You shows a skeleton-then-fill flash even though
// the user has been sitting on the History view for several seconds —
// time we could have spent loading what they're about to want. The
// flag below guards against firing the prefetch in parallel with itself
// across rapid history-enters / panel-show events. */
let _forYouPrefetchInFlight = false;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

function init(containerEl) {
  _container = containerEl;
  // Lazy-hydrate the For-You cache now that auth has populated
  // window.__augmentumUser. Doing it at module load (the old
  // behaviour) ran before auth resolved and pulled from the unscoped
  // legacy key, leaking recommendations across profiles.
  if (_lastRecsCache === null) {
    _lastRecsCache = _loadRecsCacheFromStorage();
  }
  _render();
}

function show() {
  if (!_container) return;
  _container.classList.remove('hidden');
  if (_activeView === 'history') {
    _enterHistory();
    // Warm the For You cache while the user reads History so a later
    // switch lands instantly instead of showing a skeleton-flash.
    _maybePrefetchForYou();
  } else {
    _renderRecommendations();
  }
}

async function _enterHistory() {
  // Load resume + history side-by-side; render only after both settle so
  // the user doesn't see the strip paint empty then rewrite with history.
  await Promise.allSettled([_loadResumeListening(), _loadHistory({ render: false })]);
  if (_activeView === 'history') _renderHistory();
}

// Schedule a background fetch of /for-you + /library when the cache is
// stale-or-missing AND the user is currently on the History view. The
// fetch runs at idle time so the History paint can land first; on
// browsers without requestIdleCallback, a 1.5s timeout is the floor.
//
// The prefetch deliberately does NOT start polling — that's the job
// of the eventual For You navigation. We just want the cache warm.
function _maybePrefetchForYou() {
  if (_forYouPrefetchInFlight) return;
  // Skip if both caches are still inside their freshness window — a
  // switch to For You would render from cache instantly anyway, making
  // the prefetch pure waste. Recs cache is the 24h persisted one;
  // library cache stays on its short usable window.
  const now = Date.now();
  const recsAge = _lastRecsCache ? now - _lastRecsCache.fetchedAt : Infinity;
  const libAge = _lastLibraryCache ? now - _lastLibraryCache.fetchedAt : Infinity;
  if (recsAge < _CACHE_TTL_MS && libAge < _LIB_CACHE_USABLE_MS) return;

  const fire = () => _prefetchForYouNow();
  if (typeof window.requestIdleCallback === 'function') {
    window.requestIdleCallback(fire, { timeout: 2000 });
  } else {
    setTimeout(fire, 1500);
  }
}

async function _prefetchForYouNow() {
  if (_forYouPrefetchInFlight) return;
  // Bail if the user already left History — they're either on For
  // You (which has its own cache+fetch) or the panel is hidden.
  if (_activeView !== 'history') return;
  if (document.hidden) return;
  _forYouPrefetchInFlight = true;
  try {
    const [libRes, recsRes] = await Promise.allSettled([
      _fetchLibraryZones(),
      _fetchRecommendations(),
    ]);

    if (libRes.status === 'fulfilled' && libRes.value) {
      const zones = libRes.value.zones || {};
      _lastLibraryCache = {
        zones,
        urlSet: _libraryZoneUrlSet(zones),
        fetchedAt: Date.now(),
      };
    }

    if (
      recsRes.status === 'fulfilled' &&
      recsRes.value &&
      recsRes.value.recommendations &&
      recsRes.value.recommendations.length
    ) {
      const zones = recsRes.value.zones || {};
      _lastRecsCache = {
        zones,
        urlSet: _zoneUrlSet(zones),
        fetchedAt: Date.now(),
      };
      _saveRecsCacheToStorage(_lastRecsCache);
      _trackSeen(zones);
    }
  } catch {
    // Silent — this was an opportunistic warm-up. The user's manual
    // For You click will retry through the normal cache+fetch path on
    // its own and surface any genuine error there.
  } finally {
    _forYouPrefetchInFlight = false;
  }
}

function hide() {
  if (_container) _container.classList.add('hidden');
  _disconnectObserver();
  _stopPolling();
}

window._discovery = { init, show, hide };

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function _render() {
  if (!_container) return;

  _container.innerHTML = `
    <div class="discovery-view-toggle">
      <button class="discovery-toggle-btn ${_activeView === 'recommendations' ? 'active' : ''}"
              data-view="recommendations">For You</button>
      <button class="discovery-toggle-btn ${_activeView === 'history' ? 'active' : ''}"
              data-view="history">History</button>
      <button class="discovery-sources-btn" title="Feed sources" aria-label="Feed sources">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 11a9 9 0 0 1 9 9"/><path d="M4 4a16 16 0 0 1 16 16"/><circle cx="5" cy="19" r="1.6" fill="currentColor" stroke="none"/></svg>
      </button>
    </div>
    <div class="discovery-content"></div>
  `;

  _container.querySelector('.discovery-sources-btn')
    ?.addEventListener('click', () => { _openSourcesModal(); });

  _container.querySelectorAll('.discovery-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      _activeView = btn.dataset.view;
      _container.querySelectorAll('.discovery-toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _disconnectObserver();
      if (_activeView === 'history') {
        _stopPolling();
        _enterHistory();
        // Cycle History → For You is common; warm the cache while the
        // user is on History so the round-trip back to For You feels
        // free. The freshness gate inside _maybePrefetchForYou stops
        // this from refetching when we already prefetched moments ago.
        _maybePrefetchForYou();
      } else {
        _renderRecommendations();
      }
    });
  });

  if (_activeView === 'history') _loadHistory();
  else _renderRecommendations();
}

// ---------------------------------------------------------------------------
// Feed sources editor — the front door for what For-You ingests.
// Previously the four discovery_feeds_* keys had no edit path at all:
// the fetchers read them, nothing wrote them. RSS lines accept
// rsshub:// shorthands (compose.rsshub overlay), e.g.
//   rsshub://github/release/DIYgod/RSSHub
// ---------------------------------------------------------------------------

async function _openSourcesModal() {
  if (document.querySelector('.discovery-sources-overlay')) return;

  let cfg = { hn: true, reddit_subs: [], arxiv_cats: [], rss_urls: [] };
  try {
    const resp = await fetch('/api/discovery/feeds', { credentials: 'same-origin' });
    if (resp.ok) cfg = await resp.json();
  } catch (_) { /* defaults render; save still works */ }

  const overlay = document.createElement('div');
  overlay.className = 'discovery-sources-overlay';
  overlay.innerHTML = `
    <div class="discovery-sources-modal" role="dialog" aria-label="Feed sources">
      <header>
        <h3>Feed sources</h3>
        <button class="dsm-close" aria-label="Close">&times;</button>
      </header>
      <p class="dsm-hint">What For You pulls in. Becca's curator reads the
        same sources for her notes.</p>
      <label class="dsm-toggle">
        <input type="checkbox" class="dsm-hn" ${cfg.hn ? 'checked' : ''} />
        Hacker News front page
      </label>
      <label class="dsm-field">Subreddits <span>comma-separated, no r/</span>
        <input type="text" class="dsm-reddit" placeholder="selfhosted, localllama"
               value="${escapeHtml((cfg.reddit_subs || []).join(', '))}" />
      </label>
      <label class="dsm-field">arXiv categories <span>comma-separated</span>
        <input type="text" class="dsm-arxiv" placeholder="cs.AI, cs.LG"
               value="${escapeHtml((cfg.arxiv_cats || []).join(', '))}" />
      </label>
      <label class="dsm-field">RSS feeds <span>one per line — rsshub:// shorthands work</span>
        <textarea class="dsm-rss" rows="6"
          placeholder="https://example.com/feed.xml\nrsshub://github/release/owner/repo">${escapeHtml((cfg.rss_urls || []).join('\n'))}</textarea>
      </label>
      <footer>
        <span class="dsm-status" aria-live="polite"></span>
        <button class="dsm-save">Save</button>
      </footer>
    </div>
  `;

  const close = () => overlay.remove();
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  overlay.querySelector('.dsm-close').addEventListener('click', close);

  overlay.querySelector('.dsm-save').addEventListener('click', async () => {
    const status = overlay.querySelector('.dsm-status');
    const splitCsv = (v) => v.split(',').map(s => s.trim()).filter(Boolean);
    const body = {
      hn: overlay.querySelector('.dsm-hn').checked,
      reddit_subs: splitCsv(overlay.querySelector('.dsm-reddit').value),
      arxiv_cats: splitCsv(overlay.querySelector('.dsm-arxiv').value),
      rss_urls: overlay.querySelector('.dsm-rss').value
        .split('\n').map(s => s.trim()).filter(Boolean),
    };
    status.textContent = 'Saving…';
    try {
      const resp = await fetch('/api/discovery/feeds', {
        method: 'PUT',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(String(resp.status));
      const saved = await resp.json();
      const dropped = body.rss_urls.length - (saved.rss_urls || []).length;
      status.textContent = dropped > 0
        ? `Saved — skipped ${dropped} non-URL line${dropped !== 1 ? 's' : ''}.`
        : 'Saved.';
      // Fresh sources deserve a fresh feed on the next render.
      _lastRecsCache = null;
      setTimeout(close, 900);
    } catch (err) {
      console.warn('[discovery] feed save failed', err);
      status.textContent = "Couldn't save — try again?";
    }
  });

  document.body.appendChild(overlay);
}

// ---------------------------------------------------------------------------
// For You — Recommendations (polling-aware)
// ---------------------------------------------------------------------------

async function _fetchRecommendations({ exclude = [] } = {}) {
  const params = new URLSearchParams();
  params.set('seed', String(Date.now() % 2147483647));
  for (const u of exclude.slice(-_MAX_EXCLUDE)) params.append('exclude', u);
  const resp = await fetch(`/api/discovery/for-you?${params.toString()}`);
  if (!resp.ok) throw new Error('Failed to fetch');
  return resp.json();
}

// Flatten a zones dict into a Set of URLs — used to decide whether the
// freshly fetched data actually differs from the cached render. Identical
// sets mean "no visible change," so we skip the re-render entirely and
// just clear the refresh indicator.
function _zoneUrlSet(zones) {
  const out = new Set();
  for (const items of Object.values(zones || {})) {
    for (const r of (items || [])) {
      if (r && r.url) out.add(r.url);
    }
  }
  return out;
}

function _setsEqual(a, b) {
  if (!a || !b || a.size !== b.size) return false;
  for (const v of a) if (!b.has(v)) return false;
  return true;
}

// Skeleton render — shimmering placeholder zones shown when there's no
// cached data to fall back on. Two zones with hero + 2 medium + 2 peek
// each gives the user enough "shape" cues that the eventual fill-in
// reads as "data arriving" rather than "the page rebuilt." Uses the
// shared .skeleton class from components.css for the shimmer animation.
function _renderSkeletonZones() {
  const skelZone = () => `
    <section class="discovery-zone discovery-zone-skel" aria-hidden="true">
      <header class="discovery-zone-header">
        <span class="skeleton skel-zone-kicker"></span>
        <span class="skeleton skel-zone-subtitle"></span>
      </header>
      <div class="discovery-zone-bento">
        <div class="discovery-card discovery-card-skel-hero">
          <div class="skeleton skel-cover"></div>
          <div class="skel-hero-body">
            <span class="skeleton skel-line skel-line-source"></span>
            <span class="skeleton skel-line skel-line-title"></span>
            <span class="skeleton skel-line skel-line-title skel-line-short"></span>
          </div>
        </div>
        <div class="discovery-card discovery-card-skel-medium">
          <div class="skel-medium-body">
            <span class="skeleton skel-line skel-line-source"></span>
            <span class="skeleton skel-line skel-line-title"></span>
            <span class="skeleton skel-line skel-line-snippet"></span>
          </div>
          <div class="skeleton skel-thumb"></div>
        </div>
        <div class="discovery-card discovery-card-skel-medium">
          <div class="skel-medium-body">
            <span class="skeleton skel-line skel-line-source"></span>
            <span class="skeleton skel-line skel-line-title"></span>
            <span class="skeleton skel-line skel-line-snippet skel-line-short"></span>
          </div>
          <div class="skeleton skel-thumb"></div>
        </div>
        <div class="discovery-card discovery-card-skel-peek">
          <span class="skeleton skel-favicon"></span>
          <span class="skeleton skel-line skel-line-peek"></span>
        </div>
        <div class="discovery-card discovery-card-skel-peek">
          <span class="skeleton skel-favicon"></span>
          <span class="skeleton skel-line skel-line-peek skel-line-short"></span>
        </div>
      </div>
    </section>
  `;
  return skelZone() + skelZone();
}

// Stamp the inline animation-delay on each card inside a freshly-rendered
// bento. CSS handles the keyframes; JS only assigns ordinal delays so the
// stagger respects render order across hero/medium/peek without needing
// nth-child sets per zone (which would re-trigger when zones change).
function _applyCardStagger(scope) {
  scope.querySelectorAll('.discovery-zone:not(.discovery-zone-skel) .discovery-zone-bento > .discovery-card')
    .forEach((card, idx) => {
      // Cap at 8 steps so very long tails don't run a 1.5s+ stagger.
      const step = Math.min(idx, 8);
      card.style.animationDelay = `${step * 45}ms`;
    });
}

function _renderZonesIntoRegion(region, zones) {
  const { renderZoneSection } = window.DiscoveryCards || {};
  if (!renderZoneSection) {
    region.innerHTML = '<div class="discovery-recs-empty"><p>Card renderer not loaded</p></div>';
    return false;
  }
  region.innerHTML =
    renderZoneSection('core', zones.core) +
    renderZoneSection('frontier', zones.frontier) +
    renderZoneSection('adjacent', zones.adjacent) +
    renderZoneSection('fresh', zones.fresh) +
    '<div class="discovery-refresh-pill" id="discovery-refresh-pill" hidden></div>';
  _applyCardStagger(region);
  _wireRecommendationCards(region);
  _wirePendingPill(region);
  return true;
}

function _renderRecsEmptyState(region) {
  region.innerHTML = `
    <div class="discovery-recs-empty">
      <svg class="discovery-recs-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="36" height="36">
        <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
      </svg>
      <p class="discovery-recs-empty-title">No recommendations yet</p>
      <p class="discovery-recs-empty-sub">Browse more content to build your interest profile.</p>
    </div>
  `;
}

function _showRefreshIndicator(region) {
  // Idempotent — if it's already there, leave it alone.
  if (region.querySelector('.discovery-refreshing')) return;
  const el = document.createElement('div');
  el.className = 'discovery-refreshing';
  el.setAttribute('aria-live', 'polite');
  el.innerHTML = `
    <span class="discovery-refreshing-dot"></span>
    <span>Checking for new picks</span>
  `;
  region.insertBefore(el, region.firstChild);
}

function _hideRefreshIndicator(region) {
  region.querySelector('.discovery-refreshing')?.remove();
}

// ---------------------------------------------------------------------------
// Library zones — comics / audiobooks / movies / shows from the user's
// own library. Backed by /api/discovery/library, which queries the
// file_index directly (no SearXNG, no upstream API). Typical fetch <100ms,
// so the SWR pattern below is mostly to make tab-switches feel instant
// rather than to mask any real wait.
// ---------------------------------------------------------------------------

async function _fetchLibraryZones() {
  const resp = await fetch('/api/discovery/library');
  if (!resp.ok) throw new Error('Library zones fetch failed');
  return resp.json();
}

function _libraryZoneUrlSet(zones) {
  const out = new Set();
  for (const items of Object.values(zones || {})) {
    for (const r of (items || [])) {
      if (r && r.file_id) out.add(r.file_id);
    }
  }
  return out;
}

function _renderLibrarySkeleton() {
  // Two strips × four cards each — enough scaffolding that the eventual
  // fill-in reads as data arriving, not the layout building. Cards use
  // the same .discovery-library-card chassis as real cards so the
  // measurement (190px wide) is exact.
  const skelStrip = () => `
    <section class="discovery-library-zone discovery-library-zone-skel" aria-hidden="true">
      <header class="discovery-library-header">
        <span class="skeleton skel-zone-kicker"></span>
      </header>
      <div class="discovery-library-strip">
        <div class="discovery-library-card discovery-library-card-skel"><div class="skeleton skel-cover"></div><span class="skeleton skel-line skel-line-title"></span><span class="skeleton skel-line skel-line-snippet skel-line-short"></span></div>
        <div class="discovery-library-card discovery-library-card-skel"><div class="skeleton skel-cover"></div><span class="skeleton skel-line skel-line-title"></span><span class="skeleton skel-line skel-line-snippet"></span></div>
        <div class="discovery-library-card discovery-library-card-skel"><div class="skeleton skel-cover"></div><span class="skeleton skel-line skel-line-title"></span><span class="skeleton skel-line skel-line-snippet skel-line-short"></span></div>
        <div class="discovery-library-card discovery-library-card-skel"><div class="skeleton skel-cover"></div><span class="skeleton skel-line skel-line-title"></span><span class="skeleton skel-line skel-line-snippet"></span></div>
      </div>
    </section>
  `;
  return skelStrip() + skelStrip();
}

const _LIBRARY_ZONE_LABELS = {
  comics:     'In your comics',
  audiobooks: 'On your shelf',
  movies:     'Movies you have',
  shows:      'Shows you follow',
};

// Map kind → cover aspect-ratio class. Library covers don't fit one
// shape: comics/movies/shows are 2:3 portraits, audiobooks are 1:1
// squares. The CSS class drives `aspect-ratio:` so the same card chassis
// renders correctly per kind.
function _libraryCoverClass(card) {
  if (card.kind === 'audio') return 'aspect-square';
  return 'aspect-portrait';
}

function _libraryCardHtml(card) {
  const title = escapeHtml(card.title || '');
  const subtitle = escapeHtml(card.subtitle || '');
  const fileId = escapeHtml(card.file_id || '');
  const kind = escapeHtml(card.kind || '');
  const chip = escapeHtml(card.chip || '');
  const searchHint = escapeHtml(card.search_hint || '');
  const sourceLabel = _libraryFriendlySource(card.source);
  const cover = card.cover_url
    ? `<img src="${escapeHtml(card.cover_url)}" alt="" loading="lazy" onerror="this.style.display='none'">`
    : '';
  const progress = (card.progress_pct > 0)
    ? `<div class="discovery-library-progress"><div class="discovery-library-progress-fill" style="width:${Math.round(card.progress_pct * 100)}%"></div></div>`
    : '';
  // Status pill: "Continuing" for in-progress, otherwise no chip (the
  // "recent" status is implicit — it's the default visual state).
  const statusChip = (card.status === 'in_progress')
    ? '<span class="discovery-library-status">Continuing</span>'
    : '';
  // The subtitle (author, director, etc.) becomes a click-through to
  // Files filtered by that name when the backend gave us a search_hint.
  // Without a hint, render as plain text — a year/status composite isn't
  // a useful search seed and clicking would feel broken. Same logic on
  // the source label: it's always a real chip-target, so always
  // clickable when present.
  const subtitleHtml = subtitle
    ? (searchHint
        ? `<button type="button" class="discovery-library-subtitle is-link" data-action="filter-by-search" data-chip="${chip}" data-search="${searchHint}" title="More by ${subtitle}">${subtitle}</button>`
        : `<div class="discovery-library-subtitle">${subtitle}</div>`)
    : '';
  const sourceHtml = sourceLabel
    ? (chip
        ? `<button type="button" class="discovery-library-source is-link" data-action="filter-by-chip" data-chip="${chip}" title="Show all ${sourceLabel} content">${escapeHtml(sourceLabel)}</button>`
        : `<div class="discovery-library-source">${escapeHtml(sourceLabel)}</div>`)
    : '';
  // Cover + body wrap into a single .discovery-library-card. The card
  // itself is no longer a <button> because nesting <button>s isn't
  // valid HTML — we use an outer div with a click handler, and inner
  // <button>s for the per-link affordances. The outer click still
  // routes through _openLibraryFile so card-level open behavior is
  // preserved; inner buttons stop propagation in their handlers.
  return `
    <div class="discovery-library-card" role="button" tabindex="0"
         data-file-id="${fileId}" data-kind="${kind}"
         data-cover-class="${_libraryCoverClass(card)}"
         aria-label="Open ${title}">
      <div class="discovery-library-cover ${_libraryCoverClass(card)}">
        ${cover}
        <div class="discovery-library-cover-fallback" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" width="28" height="28">
            <rect x="4" y="3" width="16" height="18" rx="2"/>
            <line x1="8" y1="8" x2="16" y2="8"/>
            <line x1="8" y1="12" x2="16" y2="12"/>
            <line x1="8" y1="16" x2="13" y2="16"/>
          </svg>
        </div>
        ${progress}
        ${statusChip}
      </div>
      <div class="discovery-library-body">
        <div class="discovery-library-title">${title}</div>
        ${subtitleHtml}
        ${sourceHtml}
      </div>
    </div>
  `;
}

// Friendly labels for the source pill — file_index stores raw provider
// slugs ("audiobookshelf", "komga"), users see "AudiobookShelf", "Komga".
const _LIBRARY_SOURCE_LABELS = {
  audiobookshelf: 'AudiobookShelf',
  librivox:       'LibriVox',
  komga:          'Komga',
  suwayomi:       'Suwayomi',
  kavita:         'Kavita',
  jellyfin:       'Jellyfin',
  emby:           'Emby',
  plex:           'Plex',
};
function _libraryFriendlySource(slug) {
  if (!slug) return '';
  return _LIBRARY_SOURCE_LABELS[slug] || slug;
}

function _renderLibraryRegion(region, zones) {
  const zoneNames = ['comics', 'audiobooks', 'movies', 'shows'];
  const parts = [];
  for (const zoneName of zoneNames) {
    const items = zones[zoneName] || [];
    if (!items.length) continue;
    const label = _LIBRARY_ZONE_LABELS[zoneName] || zoneName;
    const cards = items.map(_libraryCardHtml).join('');
    parts.push(`
      <section class="discovery-library-zone" data-library-zone="${zoneName}">
        <header class="discovery-library-header">
          <span class="discovery-library-kicker">${escapeHtml(label)}</span>
        </header>
        <div class="discovery-library-strip">${cards}</div>
      </section>
    `);
  }
  region.innerHTML = parts.join('');
  _wireLibraryCards(region);
  _applyLibraryCardStagger(region);
}

function _applyLibraryCardStagger(region) {
  region.querySelectorAll('.discovery-library-zone:not(.discovery-library-zone-skel) .discovery-library-strip > .discovery-library-card')
    .forEach((card, idx) => {
      const step = Math.min(idx, 8);
      card.style.animationDelay = `${step * 35}ms`;
    });
}

// Map (kind → chip) so the Files-navigation fallback can land on the
// right virtual chip when no dedicated player exists for that kind yet.
// Mirrors the backend _KIND_TO_CHIP table — frontend only needs the kind,
// not entity_kind, because audio/document/video all map to a single
// chip per kind in current usage.
const _KIND_TO_FILES_CHIP = {
  audio:    'audiobooks',
  document: 'comics',
  video:    'movies',  // shows fallback via the same chip — close enough
};

// Open a library card by routing to the right viewer per kind. The
// promise here resolves once the routing decision lands; visible
// follow-up (player UI, Files panel) is the job of the receiving
// module. Routes:
//   - kind=audio: dispatch 'discovery:open-file' → media-player.js
//     starts playback (existing wiring at media-player.js:594).
//   - kind=document (comics): dynamic-import comic-reader from an
//     absolute URL and call openComicReader. discovery.js is loaded
//     as a regular script (no type="module"), so relative imports
//     resolve against the document URL — using a leading "/" makes
//     the resolution deterministic regardless of route.
//   - kind=video: no dedicated player wired in this module yet;
//     fall through to the Files-navigation fallback below.
//   - any other kind: same Files-navigation fallback.
//
// The Files-navigation fallback opens the Files panel filtered to
// the right chip with the file scrolled into view. It's not the
// "play this thing" experience but it's a reliable visible action
// for every click, and the user can hit play from there.
async function _openLibraryFile(fileId, kind) {
  if (!fileId) return;
  // Diagnostic — visible in dev console so the click path can be
  // verified at a glance. Cheap; only fires on user intent.
  console.info('[discovery] open library file:', { fileId, kind });

  if (kind === 'audio') {
    window.dispatchEvent(new CustomEvent('discovery:open-file', {
      detail: { file_id: fileId, kind: 'audiobook' },
    }));
    return;
  }

  if (kind === 'document') {
    // Comics live in a dedicated reader module. Use an absolute path
    // because dynamic imports in non-module scripts resolve against
    // the document URL, not the script URL — a relative './comic-
    // reader/index.js' would 404 on any sub-route.
    try {
      const mod = await import('/scripts/comic-reader/index.js?v=surface-handoff-20260512a');
      if (mod && typeof mod.openComicReader === 'function') {
        mod.openComicReader({ id: fileId, kind });
        return;
      }
      console.warn('[discovery] comic-reader module loaded but openComicReader not exported');
    } catch (err) {
      console.warn('[discovery] comic-reader import failed:', err);
    }
    _fallbackToFilesPanel(fileId, kind);
    return;
  }

  // Video and any other kind — no dedicated player available from
  // discovery yet, route through Files so the click still leads
  // somewhere visible.
  _fallbackToFilesPanel(fileId, kind);
}

// Universal "I clicked this and want to see it" fallback. Opens the
// Files panel filtered to the kind's virtual chip with the row
// scrolled into view + selected. The user can then play / read from
// the Files surface. Reuses the files:open-with-filter listener that
// the subtitle/source link clicks already use — no new wiring.
function _fallbackToFilesPanel(fileId, kind) {
  const chip = _KIND_TO_FILES_CHIP[kind] || '';
  console.info('[discovery] falling back to Files panel:', { fileId, kind, chip });
  window.dispatchEvent(new CustomEvent('files:open-with-filter', {
    detail: { chip, fileId },
  }));
}

function _wireLibraryCards(region) {
  region.querySelectorAll('.discovery-library-card').forEach(card => {
    if (card.dataset.wired === '1') return;
    card.dataset.wired = '1';
    const open = () => _openLibraryFile(card.dataset.fileId, card.dataset.kind);
    card.addEventListener('click', (e) => {
      // Inner action buttons handle their own clicks and stopPropagation.
      // If we're here, the user clicked the card body itself.
      if (e.target.closest('[data-action="filter-by-search"]')) return;
      if (e.target.closest('[data-action="filter-by-chip"]')) return;
      open();
    });
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        // Don't intercept Enter on inner buttons — let them fire their
        // own click handler instead of stealing it for card-open.
        if (e.target !== card) return;
        e.preventDefault();
        open();
      }
    });
  });

  // Inner click-through buttons (subtitle "more by X" / source "show all
  // in this chip"). Wired by event delegation on the region so newly-
  // injected cards (post-poll reveal, cache-swap re-render) pick them
  // up automatically without re-walking every card.
  if (region.dataset.libraryLinksWired === '1') return;
  region.dataset.libraryLinksWired = '1';
  region.addEventListener('click', (e) => {
    const link = e.target.closest('[data-action="filter-by-search"], [data-action="filter-by-chip"]');
    if (!link) return;
    e.stopPropagation();
    const action = link.dataset.action;
    const detail = { chip: link.dataset.chip || '' };
    if (action === 'filter-by-search') {
      detail.search = link.dataset.search || '';
    }
    console.info('[discovery] inner link click:', action, detail);
    window.dispatchEvent(new CustomEvent('files:open-with-filter', { detail }));
  });
}

async function _refreshFeedRegion(region) {
  // 24h cache, persisted to localStorage. Within the window: render cache
  // and stop — no refetch, no polling. The /for-you fan-out is expensive
  // and the underlying signal (interest clusters) turns over slowly, so
  // showing the same set across the day is the desired behavior, not a
  // bug. Cold cache (or expired): show skeletons, fetch, persist.
  const cache = _lastRecsCache;
  const cacheAge = cache ? Date.now() - cache.fetchedAt : Infinity;
  const cacheFresh = cache && cacheAge < _CACHE_TTL_MS;

  if (cacheFresh) {
    if (_renderZonesIntoRegion(region, cache.zones)) {
      _trackSeen(cache.zones);
    }
    return;
  }

  region.innerHTML = _renderSkeletonZones();

  try {
    const data = await _fetchRecommendations();

    if (!data.recommendations || data.recommendations.length === 0) {
      _renderRecsEmptyState(region);
      _lastRecsCache = null;
      _saveRecsCacheToStorage(null);
      return;
    }

    const zones = data.zones || {};
    const newUrlSet = _zoneUrlSet(zones);
    _renderZonesIntoRegion(region, zones);

    _lastRecsCache = { zones, urlSet: newUrlSet, fetchedAt: Date.now() };
    _saveRecsCacheToStorage(_lastRecsCache);
    _trackSeen(zones);
  } catch (err) {
    region.innerHTML = `
      <div class="discovery-recs-empty">
        <p class="discovery-recs-empty-title">Could not load recommendations</p>
        <p class="discovery-recs-empty-sub">Check that SearXNG is running.</p>
      </div>
    `;
  }
}

async function _refreshLibraryRegion(region) {
  const cache = _lastLibraryCache;
  const cacheAge = cache ? Date.now() - cache.fetchedAt : Infinity;
  const cacheUsable = cache && cacheAge < _LIB_CACHE_STALE_MS;

  if (cacheUsable) {
    _renderLibraryRegion(region, cache.zones);
  } else {
    region.innerHTML = _renderLibrarySkeleton();
  }

  try {
    const data = await _fetchLibraryZones();
    const zones = (data && data.zones) || {};
    const newUrlSet = _libraryZoneUrlSet(zones);

    // If the fresh payload matches the cache, leave the DOM alone —
    // identical re-render would re-fire entrance animations for no
    // reason and could disrupt scroll position inside a strip.
    if (cacheUsable && _setsEqual(newUrlSet, cache.urlSet)) {
      // No-op
    } else {
      _renderLibraryRegion(region, zones);
    }

    _lastLibraryCache = { zones, urlSet: newUrlSet, fetchedAt: Date.now() };
  } catch {
    // Cached render is still up; on cold load (no cache + fetch
    // failure) the region simply stays empty. Library zones are not
    // critical content — failing silently is correct here.
    if (!cacheUsable) {
      region.innerHTML = '';
    }
  }
}

async function _renderRecommendations() {
  const content = _container?.querySelector('.discovery-content');
  if (!content) return;

  _stopPolling();
  _resetSeen();
  _pendingByZone = _emptyZones();

  // Two-region layout: library (instant, local) sits above feed (slow,
  // web). Each region has its own cache + fetch lifecycle so a slow
  // SearXNG response can't stall the user's own library content.
  if (!content.querySelector('[data-region="library"]')) {
    content.innerHTML = `
      <div class="discovery-library-region" data-region="library"></div>
      <div class="discovery-feed-region" data-region="feed"></div>
    `;
  }
  const libraryRegion = content.querySelector('[data-region="library"]');
  const feedRegion = content.querySelector('[data-region="feed"]');

  // Fire both refreshes in parallel. Library typically wins by ~500ms
  // on a cold load (local SQLite vs SearXNG metasearch); subsequent
  // visits are near-instant from cache for both.
  await Promise.allSettled([
    _refreshLibraryRegion(libraryRegion),
    _refreshFeedRegion(feedRegion),
  ]);
}

function _resetSeen() {
  _seenUrls.clear();
  _seenOrder.length = 0;
}

function _markSeen(url) {
  if (!url || _seenUrls.has(url)) return;
  _seenUrls.add(url);
  _seenOrder.push(url);
  while (_seenOrder.length > _SEEN_URLS_CAP) {
    const evicted = _seenOrder.shift();
    _seenUrls.delete(evicted);
  }
}

function _trackSeen(zones) {
  for (const items of Object.values(zones || {})) {
    for (const r of (items || [])) _markSeen(r && r.url);
  }
}

function _wireRecommendationCards(scope) {
  scope.querySelectorAll('.discovery-card').forEach(card => {
    if (card.dataset.wired === '1') return;
    card.dataset.wired = '1';
    card.addEventListener('click', (e) => {
      if (e.target.closest('.discovery-card-dismiss')) return;
      if (e.target.closest('.discovery-card-hide')) return;
      const url = card.dataset.url;
      if (!url) return;
      const title = card.querySelector('.discovery-card-title')?.textContent || '';
      _sendSignal('discovery_click', url, title);
      window.dispatchEvent(new CustomEvent('discovery:open-url', { detail: { url } }));
    });
  });

  scope.querySelectorAll('.discovery-card-hide').forEach(btn => {
    if (btn.dataset.wired === '1') return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const card = btn.closest('.discovery-card');
      const url = btn.dataset.url;
      if (card) _fadeOut(card);
      if (url) {
        _sendSignal('discovery_hide_url', url, '');
        _evictFromCacheByUrl(url);
      }
    });
  });

  scope.querySelectorAll('.discovery-card-dismiss').forEach(btn => {
    if (btn.dataset.wired === '1') return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const card = btn.closest('.discovery-card');
      const clusterId = btn.dataset.cluster;
      if (card) _fadeOut(card);
      if (clusterId) {
        fetch('/api/discovery/dismiss', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cluster_id: clusterId }),
        }).catch(() => {});
        _evictFromCacheByCluster(clusterId);
      }
    });
  });
}

// Drop matching items from the SWR cache so the next tab-switch render
// doesn't briefly resurrect them. The user just told us "not this" —
// honoring that is more important than cache fidelity.
function _evictFromCacheByUrl(url) {
  if (!_lastRecsCache || !url) return;
  const out = {};
  for (const zoneName of _ZONES) {
    const items = (_lastRecsCache.zones[zoneName] || []).filter(r => r.url !== url);
    out[zoneName] = items;
  }
  _lastRecsCache.zones = out;
  _lastRecsCache.urlSet = _zoneUrlSet(out);
  _saveRecsCacheToStorage(_lastRecsCache);
}

function _evictFromCacheByCluster(clusterId) {
  if (!_lastRecsCache || !clusterId) return;
  const out = {};
  for (const zoneName of _ZONES) {
    const items = (_lastRecsCache.zones[zoneName] || []).filter(r => (r.cluster_id || '') !== clusterId);
    out[zoneName] = items;
  }
  _lastRecsCache.zones = out;
  _lastRecsCache.urlSet = _zoneUrlSet(out);
  _saveRecsCacheToStorage(_lastRecsCache);
}

// ── Polling ─────────────────────────────────────────────────────────────

function _startPolling() {
  _stopPolling();
  _pollTimer = setInterval(_pollTick, _POLL_CADENCE_MS);
}

function _stopPolling() {
  if (_pollTimer) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
}

async function _pollTick() {
  if (_pollInFlight) return;
  if (_activeView !== 'recommendations') return _stopPolling();
  if (document.hidden) return;
  _pollInFlight = true;
  _lastPollAt = Date.now();
  try {
    const data = await _fetchRecommendations({ exclude: _seenOrder });
    const zones = data.zones || {};
    let added = 0;
    for (const zoneName of _ZONES) {
      const items = (zones[zoneName] || []).filter(r => r && r.url && !_seenUrls.has(r.url));
      if (!items.length) continue;
      const merged = (_pendingByZone[zoneName] || []).concat(items);
      _pendingByZone[zoneName] = merged.slice(-_PENDING_PER_ZONE_CAP);
      for (const r of items) _markSeen(r.url);
      added += items.length;
    }
    if (added > 0) _updatePendingPill();
  } catch { /* quiet — a failed poll just means wait for the next one */ }
  finally { _pollInFlight = false; }
}

function _countPending() {
  return Object.values(_pendingByZone).reduce((n, xs) => n + (xs?.length || 0), 0);
}

function _updatePendingPill() {
  const pill = document.getElementById('discovery-refresh-pill');
  if (!pill) return;
  const n = _countPending();
  if (n === 0) { pill.hidden = true; pill.textContent = ''; return; }
  pill.hidden = false;
  pill.innerHTML = `
    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="9"/></svg>
    <span>${n} new · tap to reveal</span>
  `;
}

function _wirePendingPill(scope) {
  const pill = scope.querySelector('#discovery-refresh-pill');
  if (!pill) return;
  pill.addEventListener('click', () => _revealPending());
}

function _revealPending() {
  if (_countPending() === 0) return;
  const content = _container?.querySelector('.discovery-content');
  if (!content) return;
  const { renderZoneSection, renderRecommendationCard } = window.DiscoveryCards || {};
  if (!renderRecommendationCard) return;

  for (const zoneName of _ZONES) {
    const items = _pendingByZone[zoneName] || [];
    if (!items.length) continue;

    let zoneEl = content.querySelector(`.discovery-zone[data-zone="${zoneName}"]`);
    if (!zoneEl) {
      // Zone didn't exist in the initial render (e.g. fresh arriving mid-session)
      const wrap = document.createElement('div');
      wrap.innerHTML = renderZoneSection(zoneName, items);
      zoneEl = wrap.firstElementChild;
      if (zoneEl) {
        zoneEl.classList.add('discovery-zone-flash');
        content.insertBefore(zoneEl, content.querySelector('#discovery-refresh-pill'));
      }
    } else {
      for (const item of items) {
        const cardHtml = renderRecommendationCard(item);
        const tmp = document.createElement('div');
        tmp.innerHTML = cardHtml;
        const card = tmp.firstElementChild;
        if (!card) continue;
        card.classList.add('discovery-card-flash');
        zoneEl.appendChild(card);
      }
    }
  }

  // Merge the just-revealed items into the cache BEFORE clearing the
  // pending map — otherwise a tab-switch immediately after a reveal
  // would render the pre-reveal snapshot (the user would see their
  // new cards vanish and slowly re-fetch). Item shape (title, snippet,
  // thumbnail) survives the merge because we still hold the real
  // objects in _pendingByZone here.
  if (_lastRecsCache) {
    const merged = {};
    for (const zoneName of _ZONES) {
      const existing = (_lastRecsCache.zones && _lastRecsCache.zones[zoneName]) || [];
      const revealed = _pendingByZone[zoneName] || [];
      merged[zoneName] = existing.concat(revealed);
    }
    _lastRecsCache.zones = merged;
    _lastRecsCache.urlSet = _zoneUrlSet(merged);
    _lastRecsCache.fetchedAt = Date.now();
  }

  _pendingByZone = _emptyZones();
  _wireRecommendationCards(content);
  _applyCardStagger(content);
  _updatePendingPill();
}

// Visibility-driven re-poll is intentionally absent — the For You feed
// is now backed by a 24h persisted cache, so backgrounding/foregrounding
// the tab should NOT trigger a refetch. The cache TTL alone gates renewal.

// ---------------------------------------------------------------------------
// Resume Listening (History view top row — audiobook phase 1)
// ---------------------------------------------------------------------------

async function _loadResumeListening() {
  try {
    const resp = await fetch('/api/media/resume-listening');
    if (!resp.ok) { _resumeItems = []; return; }
    const data = await resp.json();
    _resumeItems = Array.isArray(data.items) ? data.items : [];
  } catch {
    _resumeItems = [];
  }
}

function _formatProgress(pct) {
  const p = Math.max(0, Math.min(1, Number(pct) || 0));
  return Math.round(p * 100) + '%';
}

function _remainingLabel(item) {
  const total = Number(item.duration_s || 0);
  const cur = Number(item.current_time_s || 0);
  const left = Math.max(0, total - cur);
  if (!left || !total) return '';
  const mins = Math.round(left / 60);
  if (mins < 60) return `${mins}m left`;
  const hrs = Math.floor(mins / 60);
  const rem = mins % 60;
  return rem ? `${hrs}h ${rem}m left` : `${hrs}h left`;
}

function _renderResumeSection() {
  if (!_resumeItems || _resumeItems.length === 0) return '';
  const cards = _resumeItems.map(item => {
    const pct = _formatProgress(item.progress_pct);
    const remaining = _remainingLabel(item);
    const progressBar = `
      <div class="resume-progress-track">
        <div class="resume-progress-fill" style="width:${_formatProgress(item.progress_pct)}"></div>
      </div>`;
    // Author becomes a click-through to Files filtered by author name —
    // mirrors the discovery-library-subtitle treatment so both surfaces
    // share the same "more by X" affordance. Plain <div> when there's
    // no author so the rest of the card geometry stays identical.
    const authorEsc = escapeHtml(item.author || '');
    const authorHtml = item.author
      ? `<button type="button" class="resume-author is-link" data-action="filter-by-search" data-chip="audiobooks" data-search="${authorEsc}" title="More by ${authorEsc}">${authorEsc}</button>`
      : '';
    return `
      <div class="resume-card" data-file-id="${escapeHtml(item.file_id)}" role="button" tabindex="0">
        <div class="resume-cover">
          <img src="${escapeHtml(item.cover_url)}" alt="" loading="lazy" onerror="this.parentElement.classList.add('resume-cover-empty')">
          <div class="resume-cover-fallback">
            <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 18V5a2 2 0 0 1 2-2h11a2 2 0 0 1 2 2v13"/><path d="M3 18a2 2 0 0 0 2 2h13"/><path d="M8 7h6"/><path d="M8 11h6"/></svg>
          </div>
          <div class="resume-play">
            <svg viewBox="0 0 12 12" width="10" height="10" fill="#fff"><polygon points="3,1 10,6 3,11"/></svg>
          </div>
        </div>
        <div class="resume-body">
          <div class="resume-title">${escapeHtml(item.title)}</div>
          ${authorHtml}
          ${progressBar}
          <div class="resume-meta">
            <span>${pct}</span>
            ${remaining ? `<span class="resume-meta-sep">·</span><span>${escapeHtml(remaining)}</span>` : ''}
          </div>
        </div>
      </div>
    `;
  }).join('');

  return `
    <section class="resume-section" aria-label="Resume listening">
      <header class="resume-header">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 1 3 6.7"/><path d="M3 21v-6h6"/></svg>
        <span>Resume listening</span>
      </header>
      <div class="resume-scroll">${cards}</div>
    </section>
  `;
}

function _wireResumeCards(scope) {
  scope.querySelectorAll('.resume-card').forEach(card => {
    const openFile = () => {
      const id = card.dataset.fileId;
      if (!id) return;
      window.dispatchEvent(new CustomEvent('discovery:open-file', { detail: { file_id: id, kind: 'audiobook' } }));
    };
    card.addEventListener('click', (e) => {
      // Inner author button handles its own click. Without this guard
      // the author click would also trigger card-open (start playback),
      // which would feel like a navigation bug.
      if (e.target.closest('[data-action="filter-by-search"]')) return;
      openFile();
    });
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        if (e.target !== card) return;
        e.preventDefault();
        openFile();
      }
    });
  });

  // Inner click-through buttons — identical pattern to library cards.
  // Wired by event delegation so re-renders don't re-bind.
  if (scope.dataset.resumeLinksWired === '1') return;
  scope.dataset.resumeLinksWired = '1';
  scope.addEventListener('click', (e) => {
    const link = e.target.closest('.resume-card [data-action="filter-by-search"]');
    if (!link) return;
    e.stopPropagation();
    window.dispatchEvent(new CustomEvent('files:open-with-filter', {
      detail: {
        chip: link.dataset.chip || 'audiobooks',
        search: link.dataset.search || '',
      },
    }));
  });
}

// ---------------------------------------------------------------------------
// History — Rich items with favicons, thumbnails, infinite scroll
// ---------------------------------------------------------------------------

function _renderHistory() {
  const content = _container?.querySelector('.discovery-content');
  if (!content) return;

  // Filter junk on client side as safety net
  const clean = _historyItems.filter(i => !_isJunkTitle(i.title));

  const resumeHtml = _renderResumeSection();

  if (clean.length === 0 && !_historyQuery) {
    content.innerHTML = resumeHtml + `
      <div class="discovery-empty">
        <p>Your browsing trail will appear here.</p>
        <p style="color: var(--text-muted)">Pages you visit and videos you watch are logged privately on your device.</p>
      </div>
    `;
    _wireResumeCards(content);
    return;
  }

  let html = resumeHtml + `<input type="text" class="discovery-history-search"
    placeholder="Search history..." value="${escapeHtml(_historyQuery)}">`;

  if (clean.length === 0 && _historyQuery) {
    html += `<div class="discovery-empty"><p>No results for "${escapeHtml(_historyQuery)}"</p></div>`;
  }

  const days = _groupByDay(clean);
  for (const [label, items] of days) {
    html += `<div class="discovery-day-header">${escapeHtml(label)}</div>`;
    for (const item of items) {
      html += _renderHistoryItem(item);
    }
  }

  // Infinite scroll sentinel
  html += '<div class="discovery-scroll-sentinel"></div>';

  content.innerHTML = html;

  _wireResumeCards(content);

  // Wire search
  const search = content.querySelector('.discovery-history-search');
  if (search) {
    search.addEventListener('input', (e) => {
      clearTimeout(_searchDebounce);
      _searchDebounce = setTimeout(() => {
        _historyQuery = e.target.value;
        _historyPage = 1;
        _loadHistory();
      }, 300);
    });
    if (window.innerWidth >= 768) {
      search.focus();
      search.setSelectionRange(search.value.length, search.value.length);
    }
  }

  // Wire click to open (history revisit counts as engagement signal).
  // Media items (synthetic augm:media: URLs) route through the same
  // _openLibraryFile path that Discovery library cards use — opens the
  // audio player, comic reader, or video preview based on content_type.
  // Web URLs continue to dispatch discovery:open-url for the browse
  // panel reader.
  content.querySelectorAll('.discovery-history-item').forEach(el => {
    el.addEventListener('click', (e) => {
      if (e.target.closest('[data-action="delete"]')) return;
      const url = el.dataset.url;
      if (!url) return;
      const title = el.querySelector('.discovery-history-title')?.textContent || '';

      if (url.startsWith(_MEDIA_HISTORY_PREFIX)) {
        // Library media — route to the right player via the existing
        // openLibraryFile dispatcher. file_id + content_type land in
        // the data attrs from the renderer above.
        const fileId = el.dataset.fileId || url.slice(_MEDIA_HISTORY_PREFIX.length);
        const contentType = el.dataset.contentType || '';
        const openKind = _CONTENT_TYPE_TO_OPEN_KIND[contentType] || '';
        if (fileId && openKind) {
          // Log the engagement signal so the recommender can lift this
          // item's frecency, same as the web-history click below.
          _sendSignal('discovery_click', url, title);
          _openLibraryFile(fileId, openKind);
        }
        return;
      }
      _sendSignal('discovery_click', url, title);
      window.dispatchEvent(new CustomEvent('discovery:open-url', { detail: { url } }));
    });
  });

  // Wire delete with animation
  content.querySelectorAll('[data-action="delete"]').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      const row = btn.closest('.discovery-history-item');
      if (row) {
        row.classList.add('removing');
        setTimeout(() => row.remove(), 200);
      }
      try {
        await fetch(`/api/discovery/history/${id}`, { method: 'DELETE' });
        _historyItems = _historyItems.filter(i => i.id !== id);
      } catch { /* ignore */ }
    });
  });

  // Set up infinite scroll
  _setupInfiniteScroll(content);
}

// Library media items live under a synthetic URL prefix so they can ride
// in the same browse_history table as web pages without a schema change.
// The renderer below detects the prefix to swap favicon → cover, domain
// → kind label, and route clicks through the player rather than the URL
// opener. Keeping the prefix as a single source of truth here means
// future changes (e.g. swapping to a separate table) only touch one
// place in the frontend.
const _MEDIA_HISTORY_PREFIX = 'augm:media:';

const _MEDIA_KIND_LABELS = {
  audiobook: 'Audiobook',
  podcast:   'Podcast',
  comic:     'Comic',
  movie:     'Movie',
  show:      'Show',
};

// Map the history item's content_type back to the kind argument
// _openLibraryFile expects (audio / document / video). Without this
// the click handler would dispatch with content_type values that no
// listener recognises.
const _CONTENT_TYPE_TO_OPEN_KIND = {
  audiobook: 'audio',
  podcast:   'audio',
  comic:     'document',
  movie:     'video',
  show:      'video',
};

// Per-kind framing for the progress label. Audio + video like
// "47% · 2h 15m left"; comics like "Page 84 of 230". The framing
// matches what's culturally familiar for each medium — book/page
// for comics, time-remaining for audio/video. Returns '' when there's
// no usable progress data so the renderer falls back to relative-time.
function _mediaProgressLabel(contentType, meta) {
  const pct = Number(meta.current_progress_pct) || 0;
  if (meta.is_finished) return 'Finished';
  if (pct <= 0) return '';

  if (contentType === 'comic' && Number(meta.page_count) > 0) {
    const cur = Math.max(1, Number(meta.current_page) || 1);
    return `Page ${cur} of ${meta.page_count}`;
  }

  // Audio + video: "47%" prefix + remaining time when we have duration.
  const pctLabel = `${Math.round(pct * 100)}%`;
  const dur = Number(meta.duration_s) || 0;
  const cur = Number(meta.current_time_s) || 0;
  const remaining = Math.max(0, dur - cur);
  if (dur > 0 && remaining > 0) {
    return `${pctLabel} · ${_formatRemaining(remaining)} left`;
  }
  return pctLabel;
}

function _formatRemaining(seconds) {
  const total = Math.round(seconds);
  if (total < 60) return `${total}s`;
  const mins = Math.round(total / 60);
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  const rem = mins % 60;
  return rem ? `${hrs}h ${rem}m` : `${hrs}h`;
}

function _renderHistoryItem(item) {
  const meta = typeof item.metadata === 'string' ? JSON.parse(item.metadata || '{}') : (item.metadata || {});
  const isMedia = typeof item.url === 'string' && item.url.startsWith(_MEDIA_HISTORY_PREFIX);
  const isVideo = item.content_type === 'video';
  const domain = escapeHtml(item.domain || '');
  const faviconUrl = !isMedia && domain
    ? `/api/browse/image?url=${encodeURIComponent(`https://www.google.com/s2/favicons?domain=${item.domain}&sz=16`)}`
    : '';
  const title = escapeHtml(item.title || item.url);
  const id = escapeHtml(item.id);
  const url = escapeHtml(item.url);
  const time = _relativeTime(item.last_visited);

  if (isMedia) {
    // Library media item — cover + kind label + author/series subtitle.
    // Reuses .discovery-history-item--video layout so styling stays
    // consistent with YouTube history rows; differentiation is in
    // the meta line + click handler.
    //
    // Progress fields come from the server-side enrichment in
    // /api/discovery/history (which merges current source_metadata at
    // read-time). For audio: progress_pct + current_time_s + duration_s
    // give us a "47% · 2h 15m left" framing. For comics: page_count +
    // current_page give a "Page 84 of 230" framing. Either way the
    // progress bar uses current_progress_pct as the canonical 0-1
    // value so one bar style fits all kinds.
    const kindLabel = _MEDIA_KIND_LABELS[item.content_type] || 'Library';
    const fileId = String(meta.file_id || item.url.slice(_MEDIA_HISTORY_PREFIX.length) || '');
    const coverUrl = item.thumbnail || (fileId ? `/api/media/cover/${encodeURIComponent(fileId)}` : '');
    const subtitle = escapeHtml(meta.author || '');
    const progressPct = Math.max(0, Math.min(100, (Number(meta.current_progress_pct) || 0) * 100));
    const isFinished = !!meta.is_finished;
    const isInProgress = progressPct > 0 && progressPct < 100 && !isFinished;
    const progressLabel = _mediaProgressLabel(item.content_type, meta);
    const finishedBadge = isFinished
      ? '<span class="discovery-history-finished">Finished</span>'
      : '';
    return `
      <div class="discovery-history-item discovery-history-item--video discovery-history-item--media${isInProgress ? ' is-in-progress' : ''}${isFinished ? ' is-finished' : ''}"
           data-url="${url}" data-id="${id}"
           data-file-id="${escapeHtml(fileId)}"
           data-content-type="${escapeHtml(item.content_type || '')}">
        <div class="discovery-history-thumb">
          ${coverUrl ? `<img src="${escapeHtml(coverUrl)}" alt="" loading="lazy" decoding="async" onerror="this.parentElement.style.display='none'">` : ''}
          ${isInProgress ? `<div class="discovery-history-progress-bar" style="width:${progressPct.toFixed(1)}%"></div>` : ''}
          ${finishedBadge}
        </div>
        <div class="discovery-history-info">
          <div class="discovery-history-title">${title}</div>
          <div class="discovery-history-domain">
            <span class="discovery-history-kind">${escapeHtml(kindLabel)}</span>
            ${subtitle ? `<span>&middot;</span><span>${subtitle}</span>` : ''}
            ${progressLabel ? `<span>&middot;</span><span class="discovery-history-progress">${escapeHtml(progressLabel)}</span>` : ''}
            ${time && !progressLabel ? `<span>&middot;</span><span>${time}</span>` : ''}
          </div>
        </div>
        <div class="discovery-history-menu">
          <button title="Remove" data-action="delete" data-id="${id}">&times;</button>
        </div>
      </div>`;
  }

  if (isVideo && item.thumbnail) {
    // Rich video item with thumbnail
    const thumbUrl = `/api/browse/image?url=${encodeURIComponent(item.thumbnail)}`;
    const duration = meta.total_duration ? _formatTime(meta.total_duration) : '';
    const progress = meta.progress_seconds && meta.total_duration
      ? `<span class="discovery-history-progress">${_formatTime(meta.progress_seconds)} / ${_formatTime(meta.total_duration)}</span>`
      : '';
    const progressPct = (meta.progress_seconds && meta.total_duration)
      ? Math.min(100, (meta.progress_seconds / meta.total_duration) * 100)
      : 0;

    return `
      <div class="discovery-history-item discovery-history-item--video" data-url="${url}" data-id="${id}">
        <div class="discovery-history-thumb">
          <img src="${thumbUrl}" alt="" loading="lazy" decoding="async" onerror="this.parentElement.style.display='none'">
          ${duration ? `<span class="discovery-history-duration">${duration}</span>` : ''}
          ${progressPct > 0 ? `<div class="discovery-history-progress-bar" style="width:${progressPct.toFixed(1)}%"></div>` : ''}
          <div class="discovery-thumb-play"><svg viewBox="0 0 12 12" width="10" height="10" fill="#fff"><polygon points="3,1 10,6 3,11"/></svg></div>
        </div>
        <div class="discovery-history-info">
          <div class="discovery-history-title">${title}</div>
          <div class="discovery-history-domain">
            ${faviconUrl ? `<img class="discovery-history-favicon" src="${faviconUrl}" alt="" loading="lazy" decoding="async" onerror="this.style.display='none'">` : ''}
            <span>${domain}</span>
            ${progress ? `<span>&middot;</span>${progress}` : ''}
          </div>
        </div>
        <div class="discovery-history-menu">
          <button title="Remove" data-action="delete" data-id="${id}">&times;</button>
        </div>
      </div>`;
  }

  // Article item with favicon
  return `
    <div class="discovery-history-item" data-url="${url}" data-id="${id}">
      ${faviconUrl ? `<img class="discovery-history-favicon" src="${faviconUrl}" alt="" loading="lazy" decoding="async" onerror="this.style.display='none'">` : ''}
      <div class="discovery-history-info">
        <div class="discovery-history-title">${title}</div>
        <div class="discovery-history-domain">
          <span>${domain}</span>
          ${time ? `<span>&middot;</span><span>${time}</span>` : ''}
        </div>
      </div>
      <div class="discovery-history-menu">
        <button title="Remove" data-action="delete" data-id="${id}">&times;</button>
      </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Infinite Scroll
// ---------------------------------------------------------------------------

function _setupInfiniteScroll(scrollParent) {
  _disconnectObserver();
  if (!_hasMore) return;

  const sentinel = scrollParent.querySelector('.discovery-scroll-sentinel');
  if (!sentinel) return;

  _scrollObserver = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && _hasMore && !_loading) {
      _historyPage++;
      _loadHistory(true);
    }
  }, { root: scrollParent, rootMargin: '200px' });

  _scrollObserver.observe(sentinel);
}

function _disconnectObserver() {
  if (_scrollObserver) {
    _scrollObserver.disconnect();
    _scrollObserver = null;
  }
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

async function _loadHistory(append = false, { render = true } = {}) {
  if (_loading) return;
  _loading = true;
  // innerHTML reset in _renderHistory clamps scrollTop to 0 on each paint;
  // snapshot the prior offset on append so infinite scroll doesn't jump.
  const content = _container?.querySelector('.discovery-content');
  const savedScroll = (append && content) ? content.scrollTop : 0;
  try {
    const params = new URLSearchParams({ page: String(_historyPage) });
    if (_historyQuery) params.set('q', _historyQuery);
    const resp = await fetch(`/api/discovery/history?${params}`);
    if (!resp.ok) return;
    const data = await resp.json();
    _historyItems = append ? [..._historyItems, ...data.items] : data.items;

    // Hard cap on in-memory history. Infinite-scroll through a year of
    // browsing would otherwise pull thousands of items into memory and
    // quadratically slow _renderHistory (full innerHTML rebuild per
    // page). When the cap is hit, stop the observer from requesting more
    // pages — the user still sees everything loaded so far, and older
    // results remain reachable via the search box (which hits the server
    // with a fresh query). Trimming from the top would be incorrect:
    // dropped items shift the scroll anchor and we have no cheap way to
    // know their combined height, so the user would jump mid-list.
    if (_historyItems.length >= _HISTORY_MAX_ITEMS) {
      _historyItems = _historyItems.slice(0, _HISTORY_MAX_ITEMS);
      _hasMore = false;
    } else {
      _hasMore = data.has_more;
    }

    if (render) {
      _renderHistory();
      if (append && content) content.scrollTop = savedScroll;
    }
  } catch { /* ignore */ }
  finally { _loading = false; }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _groupByDay(items) {
  const groups = new Map();
  const now = new Date();
  const today = now.toDateString();
  const yesterday = new Date(now.getTime() - 86400000).toDateString();
  for (const item of items) {
    const date = new Date(item.last_visited || item.first_visited);
    const ds = date.toDateString();
    let label;
    if (ds === today) label = 'Today';
    else if (ds === yesterday) label = 'Yesterday';
    else label = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(item);
  }
  return groups;
}

function _formatTime(seconds) {
  if (!seconds || seconds <= 0) return '';
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

function _sendSignal(signalType, url, title) {
  try {
    fetch('/api/discovery/signal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        signal_type: signalType,
        source_url: url,
        source_title: title,
        content_type: 'article',
        weight: signalType === 'discovery_hide_url' ? -1.0 : 1.0,
      }),
    }).catch(() => {});
  } catch { /* ignore */ }
}

function _fadeOut(el) {
  el.style.transition = 'opacity 0.15s, transform 0.15s';
  el.style.opacity = '0';
  el.style.transform = 'translateX(-20px)';
  setTimeout(() => el.remove(), 150);
}

function _relativeTime(iso) {
  if (!iso) return '';
  try {
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return '';  // day headers handle older items
  } catch { return ''; }
}
