/**
 * discover.js — Unified Discover surface.
 *
 * Replaces the buried Settings → Marketplace button. Browses
 * provider services, titles, and (eventually) community items
 * from one grid. Backend: /api/discover/*.
 *
 * Spec: docs/superpowers/specs/2026-06-10-discover-surface-design.md
 */
import { escapeHtml } from '../app.js';

let _overlay = null;
let _busy = new Set();
let _activeCategory = '';    // empty = all
let _searchQuery = '';
let _categoryCounts = new Map();

const CATEGORY_LABELS = {
  '':              'All',
  'providers':     'Providers',
  // Add-ons sit next to Providers deliberately: both extend what this
  // instance can DO rather than adding an app you open. See
  // augmentum/addons/__init__.py for the category definition.
  'addons':        'Add-ons',
  'files':         'Files & Productivity',
  'media':         'Media',
  'networking':    'Networking',
  'automation':    'Automation',
  'developer':     'Developer Tools',
  'games':         'Games',
  'characters':    'Characters',
  'powers':        'Powers',
  'reasoning-flows': 'Reasoning Flows',
  'knowledge':     'Knowledge',
  'other':         'Other',
};

const CATEGORY_ORDER = [
  'providers', 'addons', 'files', 'media', 'networking', 'automation', 'developer',
  'games', 'characters', 'powers', 'reasoning-flows', 'knowledge', 'other',
];

const KIND_LABEL = {
  provider_service:  'Provider',
  media_server:      'Media server',
  streamed_game:     'Streamed game',
  js13k_game:        'js13k',
  web_app:           'Web app',
  character:         'Character',
  power:             'Power',
  reasoning_flow:    'Reasoning flow',
  knowledge_pack:    'Knowledge pack',
  service:           'App',
  addon:             'Add-on',
};


export async function openDiscover(initial = {}) {
  if (!_overlay) _buildPanel();
  _overlay.classList.add('visible');
  document.body.classList.add('discover-lock-scroll');
  if (initial.category) _activeCategory = initial.category;
  // Deep-link with a search term (e.g. from the calendar's "Connect a
  // calendar" affordance) so the relevant app is presented directly rather
  // than leaving the user to hunt a category.
  if (initial.search) {
    _searchQuery = String(initial.search);
    _activeCategory = '';
    const input = _overlay.querySelector('#discover-search-input');
    if (input) input.value = _searchQuery;
  }
  await _refresh();
}

export function closeDiscover() {
  if (!_overlay) return;
  _overlay.classList.remove('visible');
  document.body.classList.remove('discover-lock-scroll');
}


function _buildPanel() {
  _overlay = document.createElement('div');
  _overlay.id = 'discover-overlay';
  _overlay.className = 'discover-overlay';
  _overlay.innerHTML = `
    <div class="discover-panel">
      <header class="discover-header">
        <h2 class="discover-title">Discover</h2>
        <div class="discover-search">
          <input id="discover-search-input" type="search"
                 placeholder="Search providers, games, characters…"
                 autocomplete="off" />
        </div>
        <button class="discover-close" id="discover-close-btn"
                aria-label="Close Discover">×</button>
      </header>
      <nav class="discover-categories" id="discover-categories"></nav>
      <main class="discover-body">
        <section class="discover-home" id="discover-home" hidden>
          <section class="discover-installed-shelf" id="discover-installed-shelf"
                   aria-label="Your installed apps" hidden></section>
          <section class="discover-system-dashboard" id="discover-system-dashboard"
                   aria-label="System capabilities" hidden></section>
          <div class="discover-home-shell">
            <section class="discover-spotlight" id="discover-spotlight"></section>
            <section class="discover-paths" aria-label="Discovery paths">
              <div class="discover-path-grid" id="discover-path-grid"></div>
            </section>
          </div>
          <section class="discover-lanes" id="discover-lanes"></section>
        </section>
        <section class="discover-featured-rail" id="discover-featured-rail" hidden>
          <h3 class="discover-section-title">Featured</h3>
          <div class="discover-featured-grid" id="discover-featured-grid"></div>
        </section>
        <section class="discover-main-section">
          <h3 class="discover-section-title" id="discover-main-title">All</h3>
          <div class="discover-grid" id="discover-grid"></div>
        </section>
        <div class="discover-empty" id="discover-empty" hidden>
          <p>Nothing to show.</p>
        </div>
      </main>
    </div>
  `;
  document.body.appendChild(_overlay);

  _overlay.querySelector('#discover-close-btn').addEventListener('click', closeDiscover);
  _overlay.addEventListener('click', (e) => {
    if (e.target === _overlay) closeDiscover();
  });

  const searchInput = _overlay.querySelector('#discover-search-input');
  let searchTimer;
  searchInput.addEventListener('input', (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      _searchQuery = e.target.value.trim();
      _refresh();
    }, 200);
  });
}


async function _refresh() {
  _renderSkeleton();
  try {
    // Featured rail only renders on the unfiltered "All" view. Filters
    // are scoped — surfacing a featured rail INSIDE a category filter
    // would duplicate items the user is already scanning.
    const showFeatured = !_activeCategory && !_searchQuery;
    const fetches = [
      fetch('/api/discover/categories', { credentials: 'same-origin' }),
      _fetchListings(),
    ];
    if (showFeatured) {
      fetches.push(fetch('/api/discover/catalog?featured=true&limit=12',
                         { credentials: 'same-origin' }));
    }
    const [catRes, listingsRes, featuredRes] = await Promise.all(fetches);

    let categories = [];
    let listings = [];
    let featured = [];
    let listingsFailed = false;

    if (catRes.ok) {
      const data = await catRes.json();
      categories = data.categories || [];
      _renderCategories(categories);
    }
    if (listingsRes.ok) {
      const data = await listingsRes.json();
      listings = data.listings || [];
    } else if (listingsRes.status === 503) {
      listingsFailed = true;
      _renderError('Discover is disabled. Enable it in Settings.');
    } else {
      listingsFailed = true;
      _renderError(`Couldn't load catalog (${listingsRes.status}).`);
    }
    if (featuredRes && featuredRes.ok) {
      const data = await featuredRes.json();
      featured = data.listings || [];
    }

    if (listingsFailed) {
      _renderHome([], categories, []);
      _renderFeatured([]);
      _updateMainTitle();
      return;
    }

    _renderHome(listings, categories, featured);
    _renderListings(listings);
    _renderFeatured(featured);
    _updateMainTitle();
  } catch (err) {
    _renderError(`Couldn't reach the server: ${err.message || err}`);
  }
}


function _updateMainTitle() {
  const title = _overlay.querySelector('#discover-main-title');
  if (!title) return;
  if (_searchQuery) {
    title.textContent = `Search: "${_searchQuery}"`;
  } else if (_activeCategory) {
    title.textContent = CATEGORY_LABELS[_activeCategory] || _activeCategory;
  } else {
    title.textContent = 'All';
  }
}


function _renderFeatured(listings) {
  const rail = _overlay.querySelector('#discover-featured-rail');
  const grid = _overlay.querySelector('#discover-featured-grid');
  if (!listings.length) {
    rail.hidden = true;
    grid.innerHTML = '';
    return;
  }
  rail.hidden = false;
  // Featured cards reuse the same renderer + listing cache so install
  // works identically. Set lastListings union so _findListing can
  // locate either rail's items.
  _lastListings = _mergeListings(_lastListings, listings);
  grid.innerHTML = listings.map(_renderCard).join('');
  _wireListingActions(grid);
}

// Attach install handlers and category jumpers. Idempotent (guarded by
// data flags) so re-rendering one card — e.g. after an install flips
// its state — doesn't double-bind.
function _wireListingActions(root) {
  root.querySelectorAll('.discover-install-btn[data-id]').forEach((btn) => {
    if (!btn || btn.dataset.wired === '1') return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const listing = _findListing(btn.dataset.id);
      if (!listing) return;
      const res = await _confirmInstall(listing);
      if (res) _installListing(btn.dataset.id, res);
    });
  });

  // The whole card opens the app detail sheet (app-store interaction) —
  // same dialog the Install button uses, so Install-from-detail works
  // identically. Button clicks stopPropagation above, so no double-open.
  root.querySelectorAll('.discover-card[data-id]').forEach((card) => {
    if (!card || card.dataset.detailWired === '1') return;
    card.dataset.detailWired = '1';
    card.style.cursor = 'pointer';
    card.addEventListener('click', async () => {
      const listing = _findListing(card.dataset.id);
      if (!listing) return;
      const res = await _confirmInstall(listing);
      if (res && !listing.installed) _installListing(listing.id, res);
    });
  });

  // Installed media-server "Open ▸" → launch the HTTPS front door in a new
  // tab. (No data-id, so the install handler above skips it.)
  root.querySelectorAll('.discover-install-btn[data-open-url]').forEach((btn) => {
    if (!btn || btn.dataset.wired === '1') return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const url = btn.dataset.openUrl;
      if (url) window.open(url, '_blank', 'noopener');
    });
  });

  // Installed media-server "Manage" (⋯) → reopen the full management card.
  root.querySelectorAll('.discover-install-btn[data-manage-id]').forEach((btn) => {
    if (!btn || btn.dataset.wired === '1') return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const listing = _findListing(btn.dataset.manageId);
      if (listing) _openMediaServerManage(listing);
    });
  });

  // Installed manifest-service "Manage" (⋯) → the service card
  // (status, open, pause/start, uninstall).
  root.querySelectorAll('.discover-install-btn[data-service-manage-id]').forEach((btn) => {
    if (!btn || btn.dataset.wired === '1') return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const listing = _findListing(btn.dataset.serviceManageId);
      if (listing) _showServiceCard(listing, { manage: true });
    });
  });

  // Installed add-on "Open ▸" → the in-app surface the capability appears
  // on. An add-on has no URL of its own, so this navigates INSIDE Augmentum
  // (closing Discover first) instead of opening a tab.
  root.querySelectorAll('.discover-install-btn[data-addon-surface]').forEach((btn) => {
    if (!btn || btn.dataset.wired === '1') return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      _openAddonSurface(btn.dataset.addonSurface || '');
    });
  });

  // Installed add-on "Manage" (⋯) → the add-on sheet (what it provides,
  // disk it holds, Remove).
  root.querySelectorAll('.discover-install-btn[data-addon-manage-id]').forEach((btn) => {
    if (!btn || btn.dataset.wired === '1') return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const listing = _findListing(btn.dataset.addonManageId);
      if (listing) _showAddonCard(listing);
    });
  });

  root.querySelectorAll('[data-discover-category]').forEach((btn) => {
    if (btn.dataset.navWired === '1') return;
    btn.dataset.navWired = '1';
    btn.addEventListener('click', () => {
      _activeCategory = btn.dataset.discoverCategory || '';
      _searchQuery = '';
      const searchInput = _overlay?.querySelector('#discover-search-input');
      if (searchInput) searchInput.value = '';
      _refresh();
    });
  });
}


function _mergeListings(a, b) {
  const seen = new Set();
  const out = [];
  for (const l of [...a, ...b]) {
    if (!seen.has(l.id)) {
      seen.add(l.id);
      out.push(l);
    }
  }
  return out;
}


async function _fetchListings() {
  const params = new URLSearchParams();
  if (_activeCategory) params.set('category', _activeCategory);
  if (_searchQuery) params.set('q', _searchQuery);
  params.set('limit', '200');
  return fetch(`/api/discover/catalog?${params.toString()}`,
               { credentials: 'same-origin' });
}


function _renderCategories(cats) {
  const wrap = _overlay.querySelector('#discover-categories');
  // Sort by configured order, drop empties not in the order list.
  const totalCount = cats.reduce((sum, c) => sum + c.count, 0);
  const lookup = new Map(cats.map(c => [c.id || 'other', c.count]));
  _categoryCounts = lookup;
  const items = [
    { id: '', label: CATEGORY_LABELS[''], count: totalCount },
    ...CATEGORY_ORDER
      .filter(id => lookup.has(id))
      .map(id => ({
        id,
        label: CATEGORY_LABELS[id] || id,
        count: lookup.get(id) || 0,
      })),
  ];
  wrap.innerHTML = items.map(it => `
    <button class="discover-cat-pill ${it.id === _activeCategory ? 'active' : ''}"
            data-cat="${escapeHtml(it.id)}">
      <span class="discover-cat-label">${escapeHtml(it.label)}</span>
      <span class="discover-cat-count">${it.count}</span>
    </button>
  `).join('');

  wrap.querySelectorAll('.discover-cat-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      _activeCategory = btn.dataset.cat;
      _searchQuery = '';
      const searchInput = _overlay?.querySelector('#discover-search-input');
      if (searchInput) searchInput.value = '';
      _refresh();
    });
  });
}


function _renderListings(listings) {
  const grid = _overlay.querySelector('#discover-grid');
  const empty = _overlay.querySelector('#discover-empty');

  _lastListings = listings;

  if (!listings.length) {
    grid.innerHTML = '';
    _renderEmpty();
    return;
  }
  empty.hidden = true;

  grid.innerHTML = listings.map(_renderCard).join('');
  _wireListingActions(grid);
}

// Tiny in-memory cache so the confirm dialog can read the listing
// without re-fetching. Populated on every render.
let _lastListings = [];
function _findListing(id) {
  return _lastListings.find(l => l.id === id);
}

// Installed media-server rows fetched once and cached for the session. The
// Discover catalog only knows a media server is "installed"; the per-server
// management actions (sync, console credentials, uninstall) need the
// user_media_servers row id, which we map from the listing's provider.
// Pause/start work off the catalog service_id directly so they don't need
// this — but we reuse viewer_is_admin from the same payload.
let _mediaServersCache = null;  // { byProvider: Map, viewerIsAdmin: bool } | null

async function _loadMediaServers(force = false) {
  if (_mediaServersCache && !force) return _mediaServersCache;
  const byProvider = new Map();
  let viewerIsAdmin = false;
  try {
    const r = await fetch('/api/media/servers', { credentials: 'same-origin' });
    if (r.ok) {
      const data = await r.json();
      viewerIsAdmin = !!data.viewer_is_admin;
      for (const s of (data.servers || [])) {
        if (s && s.provider) byProvider.set(String(s.provider), s);
      }
    }
  } catch (_) { /* best-effort — management degrades to Open + Setup only */ }
  _mediaServersCache = { byProvider, viewerIsAdmin };
  return _mediaServersCache;
}

// Front-door URL for a media-server listing. Prefers the dedicated HTTPS
// front door (Caddy terminates TLS and proxies to the container) since the
// raw host port is plain HTTP and fails in a browser under HSTS.
function _mediaServerUrl(l) {
  const caps = (l && l.capabilities) || {};
  // Prefer the access-gate door (<svc>.<gate_domain>:6443) when configured —
  // it's the unbounded, signed-in entry point (no 6800-6809 port cap), so
  // every installed app is openable regardless of how many are running.
  const gateUrl = String(caps.gate_url || '');
  if (gateUrl) return gateUrl;
  const httpsPort = Number(caps.https_port || 0);
  if (httpsPort) return `https://${location.hostname}:${httpsPort}`;
  const hostPort = Number(caps.host_port || 0);
  if (hostPort) return `${location.protocol}//${location.hostname}:${hostPort}`;
  return '';
}

// The catalog service id (e.g. "suwayomi") drives pause/start, which act on
// the shared managed container. Both the install_payload service_id and
// provider carry it; prefer service_id.
function _mediaServiceId(l) {
  const p = (l && l.install_payload) || {};
  return String(p.service_id || p.provider || '');
}

// ── Manifest-service helpers (kind: "service") ─────────────────────

function _svcServiceId(l) {
  const caps = (l && l.capabilities) || {};
  const p = (l && l.install_payload) || {};
  const svc = p.service || {};
  // provider_service listings carry the id flat at install_payload.service_id
  // (see loaders/providers.py), not nested under .service — fall back to it so
  // the shared service-manage card can target audio providers too.
  return String(caps.service_id || svc.id || p.service_id || '');
}

// Front-door URL for an installed manifest service, honoring the
// manifest's browser.path (e.g. Radicale's /.web/). Same HTTPS-first
// contract as media servers — the raw host port is plain HTTP and
// fails in a browser under HSTS.
function _serviceUrl(l) {
  const base = _mediaServerUrl(l);   // reads capabilities.https_port/host_port
  if (!base) return '';
  const browser = ((l && l.install_payload) || {}).browser || {};
  const path = String(browser.path || '/');
  return base + (path.startsWith('/') ? path : `/${path}`);
}

// Human copy for the manifest's browser.after_install contract.
const AFTER_INSTALL_COPY = {
  setup_page: 'First visit opens its setup page — you create the account.',
  login: 'First visit opens its login page.',
  status: 'Opens ready to use — no account needed.',
};


// Per-kind glyphs for thumbnail-less listings. The catalog ships almost
// no cover art today (marketplace listings carry an empty thumbnail_url;
// providers carry none), so without this every card fell to a 2-letter
// monogram on one hardcoded gradient — a wall of identical dark squares
// that read as "unfinished." A kind icon + per-id hue makes an art-free
// grid look like a designed icon set instead.
const KIND_GLYPH = {
  provider_service: '<rect x="3" y="4" width="18" height="6" rx="1.5"/><rect x="3" y="14" width="18" height="6" rx="1.5"/><line x1="7" y1="7" x2="7.01" y2="7"/><line x1="7" y1="17" x2="7.01" y2="17"/>',
  streamed_game: '<rect x="2" y="6" width="20" height="12" rx="6"/><line x1="6" y1="12" x2="10" y2="12"/><line x1="8" y1="10" x2="8" y2="14"/><line x1="15" y1="11" x2="15.01" y2="11"/><line x1="18" y1="13" x2="18.01" y2="13"/>',
  web_app: '<rect x="3" y="4" width="18" height="16" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><circle cx="6" cy="6.5" r="0.5" fill="currentColor"/><circle cx="8.4" cy="6.5" r="0.5" fill="currentColor"/>',
  character: '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5"/>',
  power: '<polygon points="13 2 4 14 11 14 10 22 20 9 13 9 13 2"/>',
  reasoning_flow: '<circle cx="6" cy="6" r="2.4"/><circle cx="18" cy="6" r="2.4"/><circle cx="12" cy="18" r="2.4"/><path d="M6 8.4v3a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-3"/><line x1="12" y1="13.4" x2="12" y2="15.6"/>',
  knowledge_pack: '<path d="M5 4a2 2 0 0 1 2-2h12v18H7a2 2 0 0 0-2 2z"/><line x1="9" y1="7" x2="15" y2="7"/><line x1="9" y1="11" x2="14" y2="11"/>',
  media_server: '<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="2" y1="13" x2="22" y2="13"/><polygon points="10 6.5 14.5 9 10 11.5"/><line x1="8" y1="20" x2="16" y2="20"/><line x1="12" y1="17" x2="12" y2="20"/>',
};
KIND_GLYPH.js13k_game = KIND_GLYPH.streamed_game;
// Add-on: a puzzle-piece slotting into a frame — "extends what this
// instance can do", as opposed to the provider glyph's rack of services.
KIND_GLYPH.addon = '<path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H10a2 2 0 1 1 4 0h4.5A1.5 1.5 0 0 1 20 5.5V10a2 2 0 1 0 0 4v4.5a1.5 1.5 0 0 1-1.5 1.5H14a2 2 0 1 0-4 0H5.5A1.5 1.5 0 0 1 4 18.5V14a2 2 0 1 1 0-4z"/>';

const DISCOVER_PATHS = [
  {
    category: 'providers',
    title: 'Local engines',
    copy: 'Model, speech, and helper services for this instance.',
    iconKind: 'provider_service',
  },
  {
    category: 'media',
    title: 'Media',
    copy: 'Servers and apps for music, video, books, and downloads.',
    iconKind: 'media_server',
  },
  {
    category: 'files',
    title: 'Files & Productivity',
    copy: 'Sync, notes, documents, and everyday tools.',
    iconKind: 'web_app',
  },
  {
    category: 'networking',
    title: 'Networking',
    copy: 'Monitoring, notifications, and network utilities.',
    iconKind: 'provider_service',
  },
  {
    category: 'automation',
    title: 'Automation',
    copy: 'Workflows, watchers, and glue between your services.',
    iconKind: 'power',
  },
  {
    category: 'developer',
    title: 'Developer Tools',
    copy: 'Editors, backends, and tooling on your own box.',
    iconKind: 'reasoning_flow',
  },
  {
    category: 'games',
    title: 'Playable finds',
    copy: 'Browser and streamed games that land in the Library.',
    iconKind: 'streamed_game',
  },
  {
    category: 'characters',
    title: 'Characters',
    copy: 'Personas and story companions ready to add.',
    iconKind: 'character',
  },
  {
    category: 'powers',
    title: 'Powers',
    copy: 'New abilities and automations for Augmentum.',
    iconKind: 'power',
  },
  {
    category: 'reasoning-flows',
    title: 'Reasoning flows',
    copy: 'Structured thinking paths for harder work.',
    iconKind: 'reasoning_flow',
  },
  {
    category: 'knowledge',
    title: 'Knowledge',
    copy: 'Reference packs and domain memory.',
    iconKind: 'knowledge_pack',
  },
];

const DISCOVER_LANES = [
  {
    id: 'local-ai',
    title: 'Local AI stack',
    copy: 'Model runners, embeddings, and tool-capable engines.',
    category: 'providers',
    predicate: _isLocalAiListing,
  },
  {
    id: 'voice',
    title: 'Voice and listening',
    copy: 'Speech synthesis, voice cloning, and transcription.',
    category: 'providers',
    predicate: _isVoiceListing,
  },
  {
    id: 'play',
    title: 'Playable now',
    copy: 'Games and browser apps that become Library items.',
    category: 'games',
    predicate: _isGameListing,
  },
  {
    id: 'low-friction',
    title: 'Low-friction picks',
    copy: 'CPU-friendly services and browser-native installs.',
    category: '',
    predicate: (listing) => _isCpuFriendlyProvider(listing) || _isGameListing(listing),
  },
];

// ── "Your apps" launcher shelf ──────────────────────────────────────
// Umbrel's home-screen idea, adapted: installed browser-facing apps
// (services + media servers) live as an icon row at the TOP of the
// Discover home — one click opens the app's HTTPS front door in a new
// tab, the small ⋯ opens its manage card, and a live status dot keeps
// runtime honest. Hidden entirely when nothing is installed.

function _launchUrl(l) {
  if (l.kind === 'service') return _serviceUrl(l);
  if (l.kind === 'media_server') return _mediaServerUrl(l);
  return '';
}

function _renderInstalledShelf(allListings) {
  const shelf = _overlay?.querySelector('#discover-installed-shelf');
  if (!shelf) return;
  const apps = allListings.filter(l =>
    l.installed && (l.kind === 'service' || l.kind === 'media_server'));
  if (!apps.length) {
    shelf.hidden = true;
    shelf.innerHTML = '';
    return;
  }
  shelf.hidden = false;
  shelf.innerHTML = `
    <h3 class="discover-section-title">Your apps</h3>
    <div class="discover-shelf-row">
      ${apps.map(l => `
        <div class="discover-shelf-app" data-shelf-id="${escapeHtml(l.id)}"
             role="button" tabindex="0" title="Open ${escapeHtml(l.title || 'app')}">
          ${_thumbHtml(l, 'discover-shelf-icon')}
          <span class="discover-shelf-dot" data-status-for="${escapeHtml(l.id)}"
                aria-hidden="true"></span>
          <button class="discover-shelf-manage" data-shelf-manage="${escapeHtml(l.id)}"
                  title="Manage ${escapeHtml(l.title || 'app')}"
                  aria-label="Manage ${escapeHtml(l.title || 'app')}">&#8943;</button>
          <span class="discover-shelf-name">${escapeHtml(l.title || 'App')}</span>
        </div>
      `).join('')}
    </div>`;

  shelf.querySelectorAll('.discover-shelf-app').forEach((el) => {
    const open = () => {
      const listing = _findListing(el.dataset.shelfId);
      if (!listing) return;
      const url = _launchUrl(listing);
      if (url) window.open(url, '_blank', 'noopener');
      else if (listing.kind === 'service') _showServiceCard(listing, { manage: true });
      else _openMediaServerManage(listing);
    };
    el.addEventListener('click', open);
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
    });
  });
  shelf.querySelectorAll('.discover-shelf-manage').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const listing = _findListing(btn.dataset.shelfManage);
      if (!listing) return;
      if (listing.kind === 'media_server') _openMediaServerManage(listing);
      else _showServiceCard(listing, { manage: true });
    });
  });

  // Live status dots — one cheap probe per installed app, best-effort.
  apps.forEach(async (l) => {
    const sid = l.kind === 'service' ? _svcServiceId(l) : _mediaServiceId(l);
    if (!sid) return;
    try {
      const r = await fetch(
        `/api/marketplace/services/${encodeURIComponent(sid)}/status`,
        { credentials: 'same-origin' });
      if (!r.ok) return;
      const status = String((await r.json()).status || '');
      const dot = shelf.querySelector(
        `.discover-shelf-dot[data-status-for="${CSS.escape(l.id)}"]`);
      if (dot) dot.dataset.state =
        status === 'running' ? 'ok'
        : (status === 'pulling' || status === 'starting') ? 'warn'
        : 'down';
    } catch { /* dot stays neutral */ }
  });
}

function _renderSystemDashboard(allListings) {
  const dash = _overlay?.querySelector('#discover-system-dashboard');
  if (!dash) return;
  const caps = [];
  for (const l of allListings) {
    if (!l.installed) continue;
    const system = l.system;
    if (!system || !Array.isArray(system.capabilities) || !system.capabilities.length) continue;
    for (const cap of system.capabilities) {
      caps.push({ cap, listing: l });
    }
  }
  if (!caps.length) {
    dash.hidden = true;
    dash.innerHTML = '';
    return;
  }
  dash.hidden = false;
  dash.innerHTML = `
    ${caps.map(({ cap, listing }) => `
      <div class="discover-system-card" data-system-card="${escapeHtml(cap.hook)}"
           data-listing-id="${escapeHtml(listing.id)}" role="button" tabindex="0">
        <span class="discover-system-icon">${escapeHtml(cap.icon)}</span>
        <div class="discover-system-body">
          <div class="discover-system-header">
            <span class="discover-system-label">${escapeHtml(cap.label)}</span>
            <span class="discover-system-status">${escapeHtml(cap.status)}</span>
          </div>
          <div class="discover-system-sub">
            ${escapeHtml(listing.title || 'App')}
            ${cap.protocol ? ` · ${escapeHtml(cap.protocol)}` : ''}
          </div>
          ${cap.companion_hint ? `
            <div class="discover-system-hint">${escapeHtml(cap.companion_hint)}</div>
          ` : ''}
        </div>
      </div>
    `).join('')}`;

  dash.querySelectorAll('.discover-system-card').forEach((el) => {
    el.addEventListener('click', () => {
      const listing = _findListing(el.dataset.listingId);
      if (listing) _showServiceCard(listing, { manage: true });
    });
  });
}

function _renderHome(listings, categories, featured = []) {
  const home = _overlay?.querySelector('#discover-home');
  if (!home) return;

  const showHome = !_activeCategory && !_searchQuery;
  const allListings = _mergeListings(listings || [], featured || []);
  const total = _catalogTotal(categories);
  if (!showHome || (!allListings.length && !total)) {
    home.hidden = true;
    return;
  }

  _renderInstalledShelf(allListings);
  _renderSystemDashboard(allListings);

  const spotlight = _pickSpotlight(allListings, featured);
  const spotlightEl = home.querySelector('#discover-spotlight');
  if (spotlightEl) {
    if (spotlight) {
      spotlightEl.hidden = false;
      spotlightEl.dataset.id = spotlight.id || '';
      spotlightEl.innerHTML = _renderSpotlight(spotlight, {
        total,
        installed: allListings.filter(l => l.installed).length,
      });
    } else {
      spotlightEl.hidden = true;
      spotlightEl.innerHTML = '';
    }
  }

  const pathGrid = home.querySelector('#discover-path-grid');
  if (pathGrid) {
    const paths = DISCOVER_PATHS
      .map(def => ({ def, count: _categoryCount(categories, def.category) }))
      .filter(p => p.count > 0);
    pathGrid.innerHTML = paths.map(({ def, count }) =>
      _renderPathCard(def, count)).join('');
  }

  const lanesEl = home.querySelector('#discover-lanes');
  if (lanesEl) {
    const lanes = DISCOVER_LANES
      .map(def => _renderLane(def, allListings))
      .filter(Boolean);
    lanesEl.innerHTML = lanes.join('');
  }

  home.hidden = false;
  _wireListingActions(home);
}

function _catalogTotal(categories) {
  if (Array.isArray(categories) && categories.length) {
    return categories.reduce((sum, c) => sum + (Number(c.count) || 0), 0);
  }
  let total = 0;
  for (const count of _categoryCounts.values()) total += Number(count) || 0;
  return total;
}

function _categoryCount(categories, category) {
  if (Array.isArray(categories) && categories.length) {
    const row = categories.find(c => (c.id || 'other') === category);
    if (row) return Number(row.count) || 0;
  }
  return Number(_categoryCounts.get(category) || 0);
}

function _pickSpotlight(listings, featured) {
  const ordered = _mergeListings(featured || [], listings || []);
  return (
    ordered.find(l => l.featured && !l.installed)
    || ordered.find(l => !l.installed)
    || ordered[0]
    || null
  );
}

function _renderSpotlight(listing, stats) {
  const category = _listingCategory(listing);
  const categoryLabel = CATEGORY_LABELS[category] || 'Catalog';
  const tagline = listing.tagline || listing.description || '';
  const installed = Number(stats.installed || 0);
  const total = Number(stats.total || 0);
  const statHtml = [
    total ? ['Available', total] : null,
    installed ? ['Installed', installed] : null,
    [categoryLabel, _categoryCount([], category)],
  ].filter(Boolean).map(([label, value]) => `
    <span class="discover-spotlight-stat">
      <strong>${escapeHtml(String(value))}</strong>
      <span>${escapeHtml(label)}</span>
    </span>
  `).join('');

  const secondary = category
    ? `<button class="discover-secondary-action" type="button"
               data-discover-category="${escapeHtml(category)}">
         More ${escapeHtml(categoryLabel)}
       </button>`
    : '';

  return `
    <div class="discover-spotlight-art">${_thumbHtml(listing)}</div>
    <div class="discover-spotlight-copy">
      <div class="discover-spotlight-kicker">Spotlight</div>
      <h3 class="discover-spotlight-title">${escapeHtml(listing.title || 'Featured item')}</h3>
      <p class="discover-spotlight-text">${escapeHtml(tagline)}</p>
      <div class="discover-spotlight-stats">${statHtml}</div>
      <div class="discover-spotlight-actions">
        ${_renderInstallButton(listing, 'discover-spotlight-action')}
        ${secondary}
      </div>
    </div>
  `;
}

function _renderPathCard(def, count) {
  const glyph = KIND_GLYPH[def.iconKind] || KIND_GLYPH.web_app;
  return `
    <button class="discover-path-card" type="button"
            data-discover-category="${escapeHtml(def.category)}"
            aria-label="${escapeHtml(def.title)}: ${escapeHtml(String(count))} available">
      <span class="discover-path-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${glyph}</svg>
      </span>
      <span class="discover-path-copy">
        <strong>${escapeHtml(def.title)}</strong>
        <span>${escapeHtml(def.copy)}</span>
      </span>
      <span class="discover-path-count">${escapeHtml(String(count))}</span>
    </button>
  `;
}

function _renderLane(def, listings) {
  const items = listings
    .filter(def.predicate)
    .sort(_listingSort)
    .slice(0, 4);
  if (!items.length) return '';
  const action = def.category
    ? `<button class="discover-lane-more" type="button"
               data-discover-category="${escapeHtml(def.category)}">View all</button>`
    : '';
  return `
    <section class="discover-lane" data-lane="${escapeHtml(def.id)}">
      <div class="discover-lane-head">
        <div>
          <h3 class="discover-lane-title">${escapeHtml(def.title)}</h3>
          <p class="discover-lane-copy">${escapeHtml(def.copy)}</p>
        </div>
        ${action}
      </div>
      <div class="discover-lane-grid">
        ${items.map(_renderLaneCard).join('')}
      </div>
    </section>
  `;
}

function _renderLaneCard(listing) {
  const kindLabel = KIND_LABEL[listing.kind] || listing.kind || '';
  const tagline = listing.tagline || listing.description || '';
  return `
    <article class="discover-lane-card" data-id="${escapeHtml(listing.id)}">
      ${_renderListingIcon(listing, 'discover-lane-icon', 24)}
      <div class="discover-lane-card-copy">
        <h4>${escapeHtml(listing.title || 'Untitled')}</h4>
        <p>${escapeHtml(tagline)}</p>
        <span>${escapeHtml(kindLabel)}</span>
      </div>
      ${_renderInstallButton(listing, 'discover-mini-install')}
    </article>
  `;
}

function _renderListingIcon(listing, className, size = 22) {
  const glyph = KIND_GLYPH[listing.kind] || KIND_GLYPH.web_app;
  const hue = _hueFor(listing.id || listing.title);
  return `
    <span class="${className}" style="--ph-hue:${hue}" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"
           width="${size}" height="${size}">${glyph}</svg>
    </span>
  `;
}

function _listingSort(a, b) {
  if (!!a.installed !== !!b.installed) return a.installed ? 1 : -1;
  if (!!a.featured !== !!b.featured) return a.featured ? -1 : 1;
  return String(a.title || '').localeCompare(String(b.title || ''));
}

function _listingCategory(listing) {
  if (listing.category) return listing.category;
  if (_isGameListing(listing)) return 'games';
  if (listing.kind === 'provider_service') return 'providers';
  if (listing.kind === 'character') return 'characters';
  if (listing.kind === 'power') return 'powers';
  if (listing.kind === 'reasoning_flow') return 'reasoning-flows';
  if (listing.kind === 'knowledge_pack') return 'knowledge';
  return 'other';
}

function _listingTerms(listing) {
  const meta = listing.metadata || {};
  const caps = listing.capabilities || {};
  const tags = Array.isArray(listing.tags) ? listing.tags : [];
  const features = Array.isArray(meta.features) ? meta.features : [];
  return [
    listing.kind, listing.title, listing.tagline, listing.description,
    listing.runtime_preferred, caps.api_type, meta.service_category,
    ...tags, ...features,
  ].filter(Boolean).join(' ').toLowerCase();
}

function _isGameListing(listing) {
  return _listingCategoryShallow(listing) === 'games'
    || listing.kind === 'streamed_game'
    || listing.kind === 'js13k_game'
    || listing.kind === 'web_app';
}

function _listingCategoryShallow(listing) {
  return listing.category || '';
}

function _isLocalAiListing(listing) {
  if (listing.kind !== 'provider_service') return false;
  const terms = _listingTerms(listing);
  return /\b(llm|ollama|gguf|embedding|embeddings|tool_calling|model_pull|openai_llm)\b/.test(terms);
}

function _isVoiceListing(listing) {
  if (listing.kind !== 'provider_service') return false;
  const terms = _listingTerms(listing);
  return /\b(tts|stt|voice|speech|whisper|transcription|voice_cloning|multilingual)\b/.test(terms);
}

function _isCpuFriendlyProvider(listing) {
  return listing.kind === 'provider_service'
    && listing.capabilities
    && listing.capabilities.gpu_required === false;
}

// Cheap deterministic hue from the listing id so each placeholder lands
// on a distinct (but theme-dark) tint — breaks up the "identical squares"
// look without needing real artwork.
function _hueFor(seed) {
  let h = 0;
  const s = String(seed || '');
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return h;
}

// Route external artwork through the browse image proxy so third-party
// hosts (e.g. the Umbrel gallery CDN) never see the user's browser
// directly — privacy-first, plus server-side caching. Same-origin and
// data: URLs pass through untouched.
function _artUrl(url) {
  const u = String(url || '');
  if (!u) return '';
  if (u.startsWith('/') || u.startsWith('data:')) return u;
  return `/api/browse/image?url=${encodeURIComponent(u)}`;
}

// Umbrel-style app icon: a large rounded-square ("squircle") rendered
// from thumbnail_url when the listing ships art, or the per-kind glyph
// on a per-id tinted gradient when it doesn't. The squircle mask +
// shadow live in CSS so every icon lands visually uniform.
function _thumbHtml(l, extraClass = '') {
  if (l.thumbnail_url) {
    return `
      <span class="discover-app-icon ${extraClass}">
        <img src="${escapeHtml(_artUrl(l.thumbnail_url))}" alt="" loading="lazy">
      </span>`;
  }
  const glyph = KIND_GLYPH[l.kind] || KIND_GLYPH.web_app;
  const hue = _hueFor(l.id || l.title);
  return `
    <span class="discover-app-icon is-placeholder ${extraClass}" style="--ph-hue:${hue}" aria-hidden="true">
      <svg class="discover-card-glyph" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${glyph}</svg>
    </span>`;
}

function _installButtonState(l) {
  const busy = _busy.has(l.id);
  const installed = !!l.installed;
  if (busy) {
    return { label: 'Installing...', className: 'is-busy', disabled: true };
  }
  if (installed) {
    let label = 'Installed';
    if (l.kind === 'provider_service') label = 'Running';
    else if (l.kind === 'media_server') label = 'Connected';
    // An add-on is "Available", not "Running": nothing is running, and
    // that's the correct resting state for a capability.
    else if (l.kind === 'addon') label = 'Available';
    return { label, className: 'is-installed', disabled: true };
  }
  // Builds are long and the card should say so BEFORE the click, not after.
  if (l.kind === 'addon') {
    const mins = Number((l.capabilities || {}).build_minutes || 0);
    return { label: mins ? `Install · ~${mins} min` : 'Install', className: '', disabled: false };
  }
  return { label: 'Install', className: '', disabled: false };
}

function _renderInstallButton(l, extraClass = '') {
  // Installed ADD-ON: "Open ▸" goes to the in-app surface the capability
  // shows up on (the Library), never to an external URL — an add-on has no
  // front door of its own. "⋯" opens the manage sheet (what it provides,
  // disk reclaimed on removal, Remove).
  if (!_busy.has(l.id) && l.installed && l.kind === 'addon') {
    const caps = l.capabilities || {};
    const openCls = ['discover-install-btn', 'is-open', extraClass].filter(Boolean).join(' ');
    const mgCls = ['discover-install-btn', 'is-manage', extraClass].filter(Boolean).join(' ');
    const name = escapeHtml(l.title || 'add-on');
    const openBtn = caps.surface
      ? `<button class="${openCls}" data-addon-surface="${escapeHtml(String(caps.surface))}"
             title="Open ${name} in the Library">Open &#9656;</button>`
      : '';
    return `
    <span class="discover-ms-actions">
      ${openBtn}
      <button class="${mgCls}" data-addon-manage-id="${escapeHtml(l.id)}"
              title="Manage ${name}" aria-label="Manage ${name}">&#8943;</button>
    </span>
  `;
  }
  // Installed media server: a one-tap "Open ▸" to the HTTPS front door plus
  // a compact "Manage" (⋯) that reopens the full management card — instead
  // of a dead "Connected" label. Falls back to the normal (disabled
  // "Connected") button only if no URL resolves, so there's never a dead
  // link.
  if (!_busy.has(l.id) && l.installed
      && (l.kind === 'media_server' || l.kind === 'service'
          || l.kind === 'provider_service')) {
    // Manifest services and audio provider services (speaches/kokoro/…) both
    // open the shared service-manage card (status, start/stop, uninstall).
    // Provider services run with no front door (https_port=0), so they're
    // Manage-only — no Open ▸.
    const isSvc = l.kind === 'service' || l.kind === 'provider_service';
    const url = l.kind === 'service' ? _serviceUrl(l)
      : l.kind === 'media_server' ? _mediaServerUrl(l)
      : '';
    if (url || isSvc) {
      const openCls = ['discover-install-btn', 'is-open', extraClass]
        .filter(Boolean).join(' ');
      const mgCls = ['discover-install-btn', 'is-manage', extraClass]
        .filter(Boolean).join(' ');
      const name = escapeHtml(l.title || 'app');
      const manageAttr = isSvc ? 'data-service-manage-id' : 'data-manage-id';
      const openBtn = url
        ? `<button class="${openCls}" data-open-url="${escapeHtml(url)}"
              title="Open ${name} in a new tab">Open &#9656;</button>`
        : '';
      return `
    <span class="discover-ms-actions">
      ${openBtn}
      <button class="${mgCls}" ${manageAttr}="${escapeHtml(l.id)}"
              title="Manage ${name}" aria-label="Manage ${name}">&#8943;</button>
    </span>
  `;
    }
  }
  const state = _installButtonState(l);
  const classes = ['discover-install-btn', state.className, extraClass]
    .filter(Boolean).join(' ');
  return `
    <button class="${classes}"
            data-id="${escapeHtml(l.id)}"
            ${state.disabled ? 'disabled' : ''}>
      ${escapeHtml(state.label)}
    </button>
  `;
}

// Umbrel-style horizontal card: big squircle icon on the left, name +
// one-line tagline + a quiet category/publisher line on the right, and
// the install pill anchored bottom-right. The icon carries the color;
// the card itself stays quiet glass.
function _renderCard(l) {
  const category = _listingCategory(l);
  const catLabel = CATEGORY_LABELS[category] || KIND_LABEL[l.kind] || l.kind || '';
  const tagline = l.tagline || (l.description ? l.description.slice(0, 140) : '');
  const featured = l.featured ? '<span class="discover-card-featured">Featured</span>' : '';
  const publisherChip = l.publisher && l.publisher !== 'augmentum'
    ? `<span class="discover-card-publisher">${escapeHtml(l.publisher)}</span>`
    : '';
  const thumb = _thumbHtml(l, 'discover-card-icon');
  const installed = !!l.installed;

  const installedRibbon = installed
    ? '<span class="discover-card-installed-pip" title="Installed">&#10003;</span>'
    : '';

  return `
    <article class="discover-card ${installed ? 'is-installed' : ''}" data-id="${escapeHtml(l.id)}" data-kind="${escapeHtml(l.kind)}">
      ${thumb}
      ${installedRibbon}
      <div class="discover-card-body">
        <div class="discover-card-head">
          <h3 class="discover-card-title">${escapeHtml(l.title || 'Untitled')}</h3>
          ${featured}
        </div>
        <p class="discover-card-tagline">${escapeHtml(tagline)}</p>
        <div class="discover-card-foot">
          <div class="discover-card-meta">
            ${catLabel ? `<span class="discover-card-kind">${escapeHtml(catLabel)}</span>` : ''}
            ${publisherChip}
          </div>
          ${_renderInstallButton(l)}
        </div>
      </div>
    </article>
  `;
}


// ── Install confirmation dialog ───────────────────────────────────
//
// Provider services start a Docker container with real side effects
// (port allocation, GPU contention, multi-GB image pull). Content
// installs (characters, flows, knowledge packs, games) are lighter.
// Show a tailored confirm for each kind so a casual click on a
// provider doesn't silently spawn ollama.

async function _confirmInstall(listing) {
  // Enrich provider services that declare a gated token with live preflight
  // (is the token already set?) so the dialog only prompts when it must.
  let l = listing;
  const needsPreflight = listing.kind === 'provider_service'
    && listing.metadata && listing.metadata.requirements && listing.metadata.requirements.token;
  if (needsPreflight) {
    try {
      const r = await fetch(`/api/discover/${encodeURIComponent(listing.id)}`,
                            { credentials: 'same-origin' });
      if (r.ok) { const d = await r.json(); if (d && d.listing) l = { ...listing, ...d.listing }; }
    } catch { /* fall back to cached listing; the dispatcher 422 still guards */ }
  }
  return new Promise((resolve) => {
    const modal = document.createElement('div');
    modal.className = 'discover-confirm-backdrop';
    modal.innerHTML = _confirmDialogHtml(l);
    _overlay.appendChild(modal);

    const cleanup = (result) => {
      modal.remove();
      resolve(result);
    };
    // Proceed resolves with an options object (truthy) so install can
    // forward any collected inputs (e.g. the media-library host path).
    // Cancel/dismiss resolves false.
    const proceed = async () => {
      // Gated token (e.g. fish-tts HuggingFace): save to settings FIRST via
      // the canonical settings endpoint (encrypt-at-rest + env propagation),
      // so the install preflight passes. Block proceeding if it's empty or
      // the save fails, rather than launching a doomed install.
      const tokenInput = modal.querySelector('.discover-req-token-input');
      if (tokenInput) {
        const errEl = modal.querySelector('.discover-req-token-error');
        const showErr = (msg) => { if (errEl) { errEl.textContent = msg; errEl.style.display = 'block'; } };
        const val = tokenInput.value.trim();
        const key = tokenInput.getAttribute('data-setting-key') || '';
        if (!val || !key) { showErr('This is required to install.'); return; }
        try {
          const r = await fetch('/api/config/tools', {
            method: 'PUT', credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [key]: val }),
          });
          if (!r.ok) { showErr(r.status === 403 ? 'Admin only.' : `Couldn't save token (${r.status}).`); return; }
        } catch (e) { showErr('Network error saving token.'); return; }
      }
      const opts = {};
      // Add-on license acknowledgement. The user is the builder here, so
      // where a recipe compiles GPL software or fetches a proprietary
      // browser, the acceptance has to be theirs and explicit — the
      // dispatcher 422s without it, so block rather than launch a doomed
      // 25-minute build.
      const licenseBox = modal.querySelector('.discover-addon-license-input');
      if (licenseBox) {
        if (!licenseBox.checked) {
          const err = modal.querySelector('.discover-addon-license-error');
          if (err) { err.textContent = 'Please accept to continue.'; err.style.display = 'block'; }
          return;
        }
        opts.license_acknowledged = true;
      }
      // Chosen model (e.g. fish-tts OpenAudio S1-mini/S1) → install option.
      const modelSel = modal.querySelector('.discover-req-model-input');
      if (modelSel && modelSel.value) opts.model = modelSel.value;
      const pathInput = modal.querySelector('.discover-media-host-path');
      if (pathInput && pathInput.value.trim()) {
        opts.media_host_path = pathInput.value.trim();
      }
      // Service-manifest env prompts: each input carries its manifest
      // env key; answers travel as _install_options.env and the
      // dispatcher filters them back against the declared keys.
      const envInputs = modal.querySelectorAll('.discover-env-prompt-input');
      if (envInputs.length) {
        const env = {};
        envInputs.forEach((inp) => {
          const key = inp.getAttribute('data-env-key');
          if (key && inp.value.trim()) env[key] = inp.value.trim();
        });
        if (Object.keys(env).length) opts.env = env;
      }
      // Advanced → optional host-RAM ceiling. Blank means unbounded, which is
      // the default and what almost every install should use.
      const memInput = modal.querySelector('.discover-mem-limit');
      if (memInput && memInput.value.trim()) {
        opts.mem_limit = memInput.value.trim();
      }
      cleanup(opts);
    };
    modal.addEventListener('click', (e) => {
      if (e.target === modal) cleanup(false);
    });
    modal.querySelectorAll(
      '.discover-confirm-cancel:not(.discover-confirm-manage):not(.discover-confirm-addon-manage)')
      .forEach((b) => b.addEventListener('click', () => cleanup(false)));
    // Installed-service sheet actions: Open the front door / open the
    // management card. Both close the sheet without installing.
    modal.querySelector('.discover-confirm-open')
      ?.addEventListener('click', (e) => {
        const url = e.currentTarget.dataset.url;
        cleanup(false);
        if (url) window.open(url, '_blank', 'noopener');
      });
    modal.querySelector('.discover-confirm-manage')
      ?.addEventListener('click', () => {
        cleanup(false);
        _showServiceCard(listing, { manage: true });
      });
    // Advanced → memory limit. Show what's actually set before offering to
    // change it: an empty box next to a live limit would read as "no limit".
    const memSave = modal.querySelector('.discover-mem-limit-save');
    if (memSave) {
      const memField = modal.querySelector('.discover-mem-limit');
      const memStatus = modal.querySelector('.discover-mem-limit-status');
      const url = `/api/discover/${encodeURIComponent(listing.id)}/mem-limit`;
      fetch(url, { credentials: 'same-origin' })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (d && memField) memField.value = d.mem_limit || ''; })
        .catch(() => { /* leave blank; Save still works */ });
      memSave.addEventListener('click', async () => {
        const val = (memField?.value || '').trim();
        memSave.disabled = true;
        if (memStatus) memStatus.textContent = val ? 'Applying…' : 'Removing limit…';
        try {
          const r = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({ mem_limit: val }),
          });
          const d = await r.json().catch(() => ({}));
          if (!r.ok) {
            // Surface the server's reason (bad unit, too small, admin-only)
            // rather than a generic failure the user can't act on.
            if (memStatus) memStatus.textContent = d.error || `Failed (${r.status})`;
            memStatus?.classList.add('is-error');
          } else {
            memStatus?.classList.remove('is-error');
            if (memStatus) {
              memStatus.textContent = !val
                ? 'Limit removed.'
                : (d.recreated ? `Limit set to ${val}.` : `Saved — applies next start.`);
            }
          }
        } catch (err) {
          if (memStatus) memStatus.textContent = `Failed: ${err.message}`;
          memStatus?.classList.add('is-error');
        } finally {
          memSave.disabled = false;
        }
      });
    }
    // Add-on sheet actions: in-app Open + the add-on manage sheet.
    modal.querySelector('.discover-confirm-addon-open')
      ?.addEventListener('click', (e) => {
        const surface = e.currentTarget.dataset.surface || '';
        cleanup(false);
        _openAddonSurface(surface);
      });
    modal.querySelector('.discover-confirm-addon-manage')
      ?.addEventListener('click', () => {
        cleanup(false);
        _showAddonCard(listing);
      });
    modal.querySelector('.discover-confirm-install:not(.discover-confirm-open)')
      ?.addEventListener('click', proceed);

    // Esc dismisses
    const onKey = (e) => {
      if (e.key === 'Escape') {
        document.removeEventListener('keydown', onKey);
        cleanup(false);
      }
    };
    document.addEventListener('keydown', onKey);
  });
}


function _confirmDialogHtml(l) {
  const caps = l.capabilities || {};
  const meta = l.metadata || {};
  let warnings = '';
  let confirmLabel = 'Install';
  let mediaPathField = '';
  const capRows = [];

  if (l.kind === 'addon') {
    // Add-ons are BUILT on this machine, so the sheet's job is to set an
    // honest expectation before a click that costs 25 minutes and gigabytes:
    // how long, how much disk, what capability you gain, what you're
    // agreeing to build, and what removal gives back.
    const mins = Number(caps.build_minutes || 0);
    const gb = caps.disk_mb ? (Number(caps.disk_mb) / 1000).toFixed(1) : '';
    const deps = Array.isArray(caps.requires) ? caps.requires.length : 0;
    warnings = `
      <ul class="discover-confirm-warnings">
        <li>Builds on this machine — about <strong>${mins || '?'} minutes</strong>${
          deps ? ', longer the first time (it also builds the shared streaming runtime)' : ''
        }.</li>
        ${gb ? `<li>Uses roughly <strong>${escapeHtml(gb)} GB</strong> of disk. Removing the add-on gives it back.</li>` : ''}
        <li>Nothing runs afterwards — the capability is used on demand, so it costs no memory or CPU at rest.</li>
        <li>Requires admin — non-admins get a 403.</li>
      </ul>
    `;
    confirmLabel = 'Build & add';

    if (caps.provides) {
      capRows.push(['Adds', escapeHtml(String(caps.provides))]);
    }
    // Pinned build args ARE this category's version pin. Show them: the
    // user is the builder, so the user gets to see the recipe.
    const args = caps.build_args && typeof caps.build_args === 'object' ? caps.build_args : {};
    const argKeys = Object.keys(args);
    if (argKeys.length) {
      capRows.push(['Versions', argKeys.map((k) =>
        `<span class="discover-detail-chip">${escapeHtml(k.replace(/_VERSION$/, '').toLowerCase())} ${escapeHtml(String(args[k]))}</span>`,
      ).join(' ')]);
    }
    if (meta.license) capRows.push(['License', escapeHtml(String(meta.license))]);

    if (caps.license_notice && !l.installed) {
      mediaPathField = `
        <div class="discover-addon-license" style="margin-top:14px;padding:12px 14px;
             border-radius:10px;background:rgba(128,128,128,0.1);
             border:1px solid rgba(128,128,128,0.28);">
          <p style="margin:0 0 10px;font-size:0.85em;line-height:1.55;">
            ${escapeHtml(String(caps.license_notice))}
          </p>
          <label style="display:flex;gap:9px;align-items:flex-start;font-size:0.85em;cursor:pointer;">
            <input type="checkbox" class="discover-addon-license-input" style="margin-top:3px;">
            <span>I understand this software is built on my machine and I accept its terms.</span>
          </label>
          <div class="discover-addon-license-error" style="display:none;margin-top:8px;
               font-size:0.8em;color:var(--danger,#e5484d);"></div>
        </div>`;
    }
  } else if (l.kind === 'provider_service') {
    // Provider services have real install cost. List GPU/port/image.
    const vramMb = Number(caps.vram_mb || 0);
    const gpuLine = caps.gpu_required
      ? `<li><strong>Requires GPU</strong>${vramMb ? ` — ~${vramMb} MB VRAM` : ''}</li>`
      : '<li>CPU-only — no GPU required</li>';
    const portLine = caps.host_port
      ? `<li>Will bind host port <code>${escapeHtml(String(caps.host_port))}</code></li>`
      : '';
    const imageLine = meta.image
      ? `<li>Pulls Docker image <code>${escapeHtml(String(meta.image))}</code><br>
           <span class="discover-confirm-aside">First run can take several minutes if the image isn't cached locally.</span></li>`
      : '';
    warnings = `
      <ul class="discover-confirm-warnings">
        ${gpuLine}
        ${portLine}
        ${imageLine}
        <li>Requires admin — non-admins get a 403.</li>
      </ul>
    `;
    confirmLabel = 'Start service';

    // Capabilities table — feature flags from catalog.json
    if (Array.isArray(meta.features) && meta.features.length) {
      capRows.push(['Features', meta.features.map(f =>
        `<span class="discover-detail-chip">${escapeHtml(f)}</span>`).join(' ')]);
    }
    if (caps.api_type) {
      capRows.push(['API', `<code>${escapeHtml(caps.api_type)}</code>`]);
    }
    if (meta.service_category) {
      capRows.push(['Category', escapeHtml(meta.service_category)]);
    }
    // Install-time requirements (gated token / license), collected inline so
    // the install doesn't pull a multi-GB image then hang on a missing secret.
    const reqs = (meta.requirements && typeof meta.requirements === 'object') ? meta.requirements : {};
    if (reqs.license && (reqs.license.note || reqs.license.id)) {
      capRows.push(['License', escapeHtml(reqs.license.note || reqs.license.id)]);
    }
    // Model choice — default preselected, but the user picks (never a silent
    // auto-select). The note carries the first-boot download/compile warning.
    const modelReq = reqs.model;
    if (modelReq && Array.isArray(modelReq.choices) && modelReq.choices.length) {
      const def = modelReq.default || (modelReq.choices[0] && modelReq.choices[0].id);
      const optsHtml = modelReq.choices.map((c) => {
        const sz = c.size_note ? ` — ${escapeHtml(c.size_note)}` : '';
        return `<option value="${escapeHtml(c.id)}"${c.id === def ? ' selected' : ''}>${escapeHtml(c.label || c.id)}${sz}</option>`;
      }).join('');
      mediaPathField += `
        <label class="discover-media-path-field">
          <span class="discover-media-path-label">${escapeHtml(modelReq.label || 'Model')}</span>
          <select class="discover-req-model-input"
                  style="width:100%;padding:8px 10px;border-radius:8px;font:inherit;color:inherit;
                         background:transparent;border:1px solid rgba(128,128,128,0.3);">${optsHtml}</select>
          ${modelReq.note ? `<span class="discover-media-path-hint">⏳ ${escapeHtml(modelReq.note)}</span>` : ''}
        </label>`;
    }
    const tokenReq = reqs.token;
    const tokenSet = !!(l.preflight && l.preflight.token_set);
    if (tokenReq && !tokenSet) {
      const help = tokenReq.help_url
        ? ` <a href="${escapeHtml(tokenReq.help_url)}" target="_blank" rel="noopener noreferrer">Get one ↗</a>`
        : '';
      mediaPathField += `
        <label class="discover-media-path-field">
          <span class="discover-media-path-label">${escapeHtml(tokenReq.label || 'Access token')}${help}</span>
          <input type="password" class="discover-req-token-input"
                 data-setting-key="${escapeHtml(tokenReq.setting || '')}"
                 autocomplete="off" spellcheck="false" placeholder="Required to install" />
          ${tokenReq.reason ? `<span class="discover-media-path-hint">${escapeHtml(tokenReq.reason)}</span>` : ''}
          <span class="discover-req-token-error" style="display:none;color:var(--danger,#e5484d);font-size:0.82em;margin-top:4px;"></span>
        </label>`;
    }
  } else if (l.kind === 'media_server') {
    // Provisions a fresh content server on this box and auto-connects
    // it to Files — spell out the real side effects + the payoff.
    const portLine = caps.host_port
      ? `<li>Binds host port <code>${escapeHtml(String(caps.host_port))}</code></li>`
      : '';
    const imageLine = meta.image
      ? `<li>Pulls Docker image <code>${escapeHtml(String(meta.image))}</code><br>
           <span class="discover-confirm-aside">First run can take a few minutes while the image downloads.</span></li>`
      : '';
    const payoff = caps.files_payoff || meta.files_payoff || '';
    const payoffLine = payoff
      ? `<li><strong>${escapeHtml(payoff)}</strong></li>`
      : '';
    const contentNote = meta.content_note || '';
    const contentLine = contentNote
      ? `<li>${escapeHtml(contentNote)}</li>`
      : '';
    const authLine = (caps.managed_credentials || caps.managed_auth)
      ? (caps.first_run_wizard
        ? '<li>Runs first-time setup automatically with a managed Augmentum admin account so it isn’t left open.</li>'
        : '<li>Secured automatically — Augmentum sets a managed login so it isn’t left open.</li>')
      : (caps.no_auth
        ? '<li>Connects automatically — no account or credentials to set up.</li>'
        : '<li>First-run account setup runs automatically where supported.</li>');
    warnings = `
      <ul class="discover-confirm-warnings">
        <li>Starts a fresh <strong>${escapeHtml(l.title || 'media')}</strong> container on this box.</li>
        ${imageLine}
        ${portLine}
        ${authLine}
        ${contentLine}
        ${payoffLine}
        <li>Requires admin — non-admins get a 403.</li>
      </ul>
    `;
    confirmLabel = 'Install & connect';
    if (Array.isArray(meta.features) && meta.features.length) {
      capRows.push(['Features', meta.features.map(f =>
        `<span class="discover-detail-chip">${escapeHtml(f)}</span>`).join(' ')]);
    }
    // External library: let the user point the server at their OWN media
    // folder (a host path bind-mounted in) instead of opaque Docker
    // storage. Optional — blank uses a managed Docker volume.
    if (caps.needs_media_path) {
      mediaPathField = `
        <label class="discover-media-path-field">
          <span class="discover-media-path-label">Media folder on this server <span class="discover-media-path-opt">(optional)</span></span>
          <input type="text" class="discover-media-host-path" autocomplete="off" spellcheck="false"
                 placeholder="e.g. /mnt/media or C:\\Media" />
          <span class="discover-media-path-hint">A folder on the machine running Augmentum (the Docker host). Your library stays there — leave blank to use managed Docker storage you can fill later.</span>
        </label>`;
    }
  } else if (l.kind === 'service') {
    // Service app manifest (2026-07-18 service-OS design). Everything
    // shown here comes from the validated manifest — image, RAM, what
    // opens in the browser afterwards, and any typed setup questions.
    const manifest = l.install_payload || {};
    const svc = manifest.service || {};
    const browser = manifest.browser || {};
    const resources = manifest.resources || {};
    const imageLine = svc.image
      ? `<li>Pulls Docker image <code>${escapeHtml(String(svc.image))}</code><br>
           <span class="discover-confirm-aside">First run can take a few minutes while the image downloads.</span></li>`
      : '';
    // The manifest's ram_mb is a declared MINIMUM ("won't run below this"),
    // not a typical or peak figure — say so, so nobody reads it as a budget.
    const ramLine = resources.ram_mb
      ? `<li>Needs at least ${escapeHtml(String(resources.ram_mb))} MB RAM free to start</li>`
      : '';
    const afterLabel = {
      setup_page: 'Opens its setup page in your browser after install — you create the account.',
      login: 'Opens its login page in your browser after install.',
      status: 'Opens a status page in your browser after install.',
    }[browser.after_install] || '';
    const afterLine = afterLabel ? `<li><strong>${escapeHtml(afterLabel)}</strong></li>` : '';
    warnings = `
      <ul class="discover-confirm-warnings">
        <li>Starts a <strong>${escapeHtml(l.title || 'service')}</strong> container on this box, served over HTTPS.</li>
        ${imageLine}
        ${ramLine}
        ${afterLine}
        <li>Uninstalling keeps its data — nothing is deleted without asking.</li>
        <li>Requires admin — non-admins get a 403.</li>
      </ul>
    `;
    confirmLabel = 'Install';
    // Optional host folder (media_mount) — reuse the media path field.
    if (svc.media_mount) {
      mediaPathField = `
        <label class="discover-media-path-field">
          <span class="discover-media-path-label">Library folder on this server <span class="discover-media-path-opt">(optional)</span></span>
          <input type="text" class="discover-media-host-path" autocomplete="off" spellcheck="false"
                 placeholder="e.g. /mnt/media or C:\Media" />
          <span class="discover-media-path-hint">A folder on the machine running Augmentum. Leave blank to use managed storage you can fill later.</span>
        </label>`;
    }
    // Typed setup questions from the manifest.
    const prompts = Array.isArray(svc.env_prompts) ? svc.env_prompts : [];
    for (const pr of prompts) {
      if (!pr || !pr.key) continue;
      mediaPathField += `
        <label class="discover-media-path-field">
          <span class="discover-media-path-label">${escapeHtml(pr.label || pr.key)}</span>
          <input type="${pr.secret ? 'password' : 'text'}" class="discover-env-prompt-input"
                 data-env-key="${escapeHtml(pr.key)}" autocomplete="off" spellcheck="false"
                 value="${escapeHtml(pr.default || '')}" />
        </label>`;
    }
    // Advanced — collapsed, because the right answer for almost everyone is
    // "leave it alone". Augmentum sets no default ceiling on purpose: how much
    // an app wants depends on what you use it for and how many people use it,
    // which we can't know. Offered here so the choice is yours, not ours.
    //
    // Installed apps get a Save button (the value applies via a recreate);
    // pre-install it just rides along in the install options.
    const memHint = `A hard ceiling on how much RAM this app may use. Nothing is
      reserved, so a limit costs you nothing until it's reached &mdash; but an app
      that hits it is stopped by the system, which looks to you like a crash.
      Leave blank unless you've measured this app on your own setup.`;
    mediaPathField += `
      <details class="discover-advanced">
        <summary class="discover-advanced-summary">Advanced</summary>
        <label class="discover-media-path-field">
          <span class="discover-media-path-label">Memory limit <span class="discover-media-path-opt">(optional)</span></span>
          <input type="text" class="discover-mem-limit" autocomplete="off" spellcheck="false"
                 placeholder="e.g. 2g — leave blank for no limit" />
          <span class="discover-media-path-hint">${memHint}</span>
          ${l.installed ? `
            <div class="discover-mem-limit-row">
              <button class="discover-mem-limit-save" type="button" data-id="${escapeHtml(l.id)}">Save &amp; restart</button>
              <span class="discover-mem-limit-status" role="status"></span>
            </div>
            <span class="discover-media-path-hint">Changing this restarts the app. Its data is kept.</span>` : ''}
        </label>
      </details>`;
  } else if (l.kind === 'power' || l.kind === 'knowledge_pack') {
    warnings = `
      <ul class="discover-confirm-warnings">
        <li>Install-wide — every user on this Augmentum instance gets access.</li>
        <li>Admin only.</li>
      </ul>
    `;
  } else if (l.kind === 'character' || l.kind === 'reasoning_flow') {
    warnings = `
      <ul class="discover-confirm-warnings">
        <li>Installs to your account only — won't affect other users.</li>
      </ul>
    `;
  } else {
    // Games / web apps / titles
    warnings = `
      <ul class="discover-confirm-warnings">
        <li>Adds the item to your Library.</li>
      </ul>
    `;
    // Title-specific capability surface
    if (Array.isArray(caps.input_modes) && caps.input_modes.length) {
      capRows.push(['Input', caps.input_modes.map(m =>
        `<span class="discover-detail-chip">${escapeHtml(m)}</span>`).join(' ')]);
    }
    if (caps.multiplayer) {
      capRows.push(['Multiplayer', `up to ${escapeHtml(String(caps.multiplayer))} players`]);
    }
    if (typeof caps.save_states === 'boolean') {
      capRows.push(['Save states', caps.save_states ? 'Yes' : 'No']);
    }
    if (typeof caps.offline === 'boolean') {
      capRows.push(['Offline play', caps.offline ? 'Yes' : 'No']);
    }
    if (meta.engine) {
      capRows.push(['Engine', escapeHtml(meta.engine)]);
    }
    if (meta.year) {
      capRows.push(['Year', escapeHtml(String(meta.year))]);
    }
    if (meta.license) {
      capRows.push(['License', escapeHtml(meta.license)]);
    }
  }

  const tagChips = Array.isArray(l.tags) && l.tags.length
    ? `<div class="discover-detail-tags">${l.tags.map(t =>
        `<span class="discover-detail-chip">${escapeHtml(t)}</span>`).join(' ')}</div>`
    : '';

  // ── App-store detail header: squircle icon · name · tagline ──────
  const publisherName = l.publisher && l.publisher !== 'augmentum'
    ? l.publisher : (meta.developer || '');
  const headerSub = [
    l.tagline ? escapeHtml(l.tagline) : '',
    publisherName ? `<span class="discover-detail-dev">${escapeHtml(publisherName)}</span>` : '',
  ].filter(Boolean).join('');

  // Screenshot gallery (metadata.gallery = list of image URLs, proxied).
  const gallery = Array.isArray(meta.gallery) ? meta.gallery.filter(Boolean) : [];
  const galleryBlock = gallery.length
    ? `<div class="discover-detail-gallery">${gallery.map(u =>
        `<img src="${escapeHtml(_artUrl(u))}" alt="" loading="lazy">`).join('')}</div>`
    : '';

  // Full description (vs the 200-char tagline). Falls back to tagline
  // if the listing didn't provide a description.
  const desc = l.description || l.tagline || '';
  const descBlock = desc
    ? `<p class="discover-detail-desc">${escapeHtml(desc)}</p>`
    : '';

  // Metadata rows, Umbrel-style label/value grid. Listing metadata
  // first, then any kind-specific capability rows collected above.
  const infoRows = [];
  if (meta.version) infoRows.push(['Version', escapeHtml(String(meta.version))]);
  if (meta.developer) infoRows.push(['Developer', escapeHtml(String(meta.developer))]);
  if (meta.license) infoRows.push(['License', escapeHtml(String(meta.license))]);
  if (meta.website) {
    infoRows.push(['Website', `<a href="${escapeHtml(meta.website)}" target="_blank"
      rel="noopener noreferrer">${escapeHtml(meta.website.replace(/^https?:\/\//, ''))} ↗</a>`]);
  }
  if (l.source_url) {
    infoRows.push(['Source', `<a href="${escapeHtml(l.source_url)}" target="_blank"
      rel="noopener noreferrer">${escapeHtml(l.source_url.replace(/^https?:\/\//, ''))} ↗</a>`]);
  }
  const allRows = [...infoRows, ...capRows];
  const capTable = allRows.length
    ? `<dl class="discover-detail-caps">${allRows.map(([k, v]) =>
        `<dt>${escapeHtml(k)}</dt><dd>${v}</dd>`).join('')}</dl>`
    : '';

  return `
    <div class="discover-confirm-dialog discover-detail-dialog" role="dialog" aria-modal="true">
      <header class="discover-detail-header">
        ${_thumbHtml(l, 'discover-detail-icon')}
        <div class="discover-detail-headings">
          <h3 class="discover-confirm-title">${escapeHtml(l.title || 'Install')}</h3>
          ${headerSub ? `<p class="discover-detail-tagline">${headerSub}</p>` : ''}
        </div>
      </header>
      ${galleryBlock}
      ${descBlock}
      ${capTable}
      ${tagChips}
      ${warnings}
      ${mediaPathField}
      <div class="discover-confirm-actions">
        <button class="discover-confirm-cancel" type="button">${l.installed ? 'Close' : 'Cancel'}</button>
        ${_confirmActionButtons(l, confirmLabel)}
      </div>
    </div>
  `;
}

// Primary actions for the detail sheet. Installed manifest services get
// live actions (Open front door + Manage) — a disabled "Installed ✓"
// would repeat the dead-button aftercare defect at the sheet level.
function _confirmActionButtons(l, confirmLabel) {
  if (!l.installed) {
    return `<button class="discover-confirm-install" type="button">${escapeHtml(confirmLabel)}</button>`;
  }
  if (l.kind === 'service') {
    const url = _serviceUrl(l);
    const open = url
      ? `<button class="discover-confirm-install discover-confirm-open" type="button"
             data-url="${escapeHtml(url)}">Open &#9656;</button>`
      : '';
    return `
      <button class="discover-confirm-cancel discover-confirm-manage" type="button">Manage</button>
      ${open}`;
  }
  // Installed add-on: live actions, not a dead "Installed ✓" — same
  // aftercare rule the service sheet follows. Manage opens the add-on
  // sheet (disk held, re-anchor state, Remove).
  if (l.kind === 'addon') {
    const surface = String((l.capabilities || {}).surface || '');
    const open = surface
      ? `<button class="discover-confirm-install discover-confirm-addon-open" type="button"
             data-surface="${escapeHtml(surface)}">Open &#9656;</button>`
      : '';
    return `
      <button class="discover-confirm-cancel discover-confirm-addon-manage" type="button">Manage</button>
      ${open}`;
  }
  return '<button class="discover-confirm-install is-installed" type="button" disabled>Installed ✓</button>';
}


async function _installListing(listingId, options = {}) {
  if (_busy.has(listingId)) return;
  _busy.add(listingId);
  _rerenderCard(listingId);

  try {
    const resp = await fetch(`/api/discover/${encodeURIComponent(listingId)}/install`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ confirm: true, options: options || {} }),
    });
    if (resp.ok) {
      const data = await resp.json();
      // Flip the listing to its installed state so the card reflects
      // reality immediately — previously it reset to "Install" and the
      // user had to reopen Discover to see it had worked.
      const listing = _findListing(listingId);
      if (listing) listing.installed = true;
      // Provider services (STT/TTS/LLM/image) get a post-install card:
      // the install only starts the container — the card surfaces the
      // now-registered provider, its model, live readiness, and the
      // concrete next steps (set default / open UI / test) so pressing
      // Install actually completes the user's intent.
      if (data.staged && data.resource_id) {
        // Staged install: resource_id is a background-job id. Show the staged
        // progress card (Preparing → Downloading → Starting → Warming up →
        // Ready), then hand off to the right post-install card once healthy.
        // Used by engine (service_staged) AND provider services now, for a
        // uniform install experience.
        _showStagedInstallCard(String(data.resource_id), listing);
      } else if (data.kind === 'provider_service' && data.resource_id) {
        // Legacy non-staged provider path (job queue unavailable) — fall back
        // to the provider next-steps card.
        _showProviderNextSteps(String(data.resource_id), listing);
      } else if (data.kind === 'service') {
        // Manifest services keep the browser promise: the setup card
        // polls live status ("pulling → starting → running") and opens
        // the app's front door per the manifest's browser block.
        _showServiceCard(listing, { fresh: true });
      } else if (data.kind === 'media_server') {
        // Media servers install EMPTY (no bundled content). The setup
        // card sets that expectation and drives the one required step:
        // add a source/library in the server, then sync into Files.
        _showMediaServerNextSteps(String(data.resource_id || ''), listing);
      } else {
        _toast(`Installed: ${KIND_LABEL[data.kind] || data.kind || 'item'}`);
      }
    } else {
      const err = await resp.json().catch(() => ({}));
      _toast(`Install failed: ${err.error || err.detail || resp.status}`, true);
    }
  } catch (err) {
    _toast(`Install failed: ${err.message || err}`, true);
  } finally {
    _busy.delete(listingId);
    _rerenderCard(listingId);
  }
}


// ── Post-install provider next-steps card ───────────────────────────
// Self-contained modal (inline-styled so it needs no CSS file and stays
// theme-neutral). Calls the marketplace provider-status endpoint, which
// self-heals registration + live-probes the service, and renders the
// concrete next steps the bridge computed. Polls while the model is still
// downloading so "Starting…" flips to "Ready · serving <model>" on its own.

async function _providerStatus(serviceId) {
  const resp = await fetch(
    `/api/marketplace/services/${encodeURIComponent(serviceId)}/provider-status`,
    { credentials: 'same-origin' },
  );
  if (!resp.ok) throw new Error(`status ${resp.status}`);
  return resp.json();
}

function _showProviderNextSteps(serviceId, listing) {
  // One card at a time.
  _overlay?.querySelector('.discover-provider-card')?.closest('.discover-provider-scrim')?.remove();

  const scrim = document.createElement('div');
  scrim.className = 'discover-provider-scrim';
  scrim.style.cssText =
    'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;' +
    'justify-content:center;background:rgba(0,0,0,0.5);backdrop-filter:blur(2px);';
  scrim.innerHTML = `
    <div class="discover-provider-card" role="dialog" aria-label="Install next steps"
         style="background:var(--surface-1,var(--bg,#16161c));color:var(--text,inherit);
                border:1px solid rgba(128,128,128,0.28);border-radius:14px;
                padding:20px 22px;max-width:460px;width:90%;
                box-shadow:0 14px 48px rgba(0,0,0,0.45);font:inherit;">
      <div class="dp-body">Checking provider…</div>
    </div>`;
  _overlay.appendChild(scrim);

  const card = scrim.querySelector('.discover-provider-card');
  const body = card.querySelector('.dp-body');
  const name = (listing && (listing.title || listing.name)) || serviceId;
  const state = { cancelled: false, polls: 0 };

  scrim.addEventListener('click', (e) => {
    if (e.target === scrim) { state.cancelled = true; scrim.remove(); }
  });

  const render = (st) => {
    if (state.cancelled) return;
    const reachable = !!st.reachable;
    const model = st.expected_model || st.default_model || '';
    const isDefault = !!st.is_default;
    const ptype = st.provider_type || '';
    const ready = reachable
      ? `<span style="color:var(--ok,#5cb85c);">● Ready</span>`
      : `<span style="color:var(--warn,#e0a23a);">◴ Starting…</span>`;
    const modelLine = model
      ? `<div style="opacity:0.8;font-size:0.86em;margin-top:2px;">Serving <strong>${escapeHtml(model)}</strong>${reachable ? '' : ' · downloading on first use'}</div>`
      : '';
    const probeNote = (!reachable && st.probe_detail)
      ? `<div style="opacity:0.6;font-size:0.8em;margin-top:4px;">${escapeHtml(st.probe_detail)}</div>` : '';
    const regWarn = (st.registered === false)
      ? `<div style="color:var(--warn,#e0a23a);font-size:0.84em;margin-top:6px;">Registration incomplete — ${escapeHtml(st.detail || 'try again')}.</div>` : '';
    const steps = Array.isArray(st.next_steps) ? st.next_steps : [];
    const btns = steps.map((s, i) => {
      const label = `${escapeHtml(s.label || s.action || 'Action')}`;
      const sub = s.detail ? `<span style="display:block;opacity:0.6;font-size:0.78em;">${escapeHtml(s.detail)}</span>` : '';
      const primary = (s.action === 'set_default');
      return `<button type="button" class="dp-step" data-i="${i}"
        style="display:block;width:100%;text-align:left;margin:6px 0;padding:9px 12px;
               border-radius:9px;cursor:pointer;font:inherit;
               border:1px solid ${primary ? 'var(--accent,#6c8cff)' : 'rgba(128,128,128,0.3)'};
               background:${primary ? 'color-mix(in srgb,var(--accent,#6c8cff) 16%,transparent)' : 'transparent'};
               color:inherit;">${label}${sub}</button>`;
    }).join('');

    body.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
        <strong style="font-size:1.05em;">✓ ${escapeHtml(name)}</strong>
        <button type="button" class="dp-close" aria-label="Close"
          style="border:none;background:transparent;color:inherit;opacity:0.6;
                 font-size:1.3em;line-height:1;cursor:pointer;">×</button>
      </div>
      <div style="margin-top:6px;font-size:0.92em;">${ready}${ptype ? ` · ${escapeHtml(ptype.toUpperCase())}` : ''}${isDefault ? ' · default' : ''}</div>
      ${modelLine}${probeNote}${regWarn}
      <div class="dp-steps" style="margin-top:14px;">${btns || '<div style="opacity:0.6;">No actions available.</div>'}</div>
      <div class="dp-models" style="margin-top:6px;"></div>
      <div class="dp-feedback" style="margin-top:8px;font-size:0.84em;min-height:1.1em;opacity:0.85;"></div>`;

    card.querySelector('.dp-close').addEventListener('click', () => {
      state.cancelled = true; scrim.remove();
    });
    const feedback = card.querySelector('.dp-feedback');
    card.querySelectorAll('.dp-step').forEach((btn) => {
      btn.addEventListener('click', () => _runProviderAction(
        steps[Number(btn.dataset.i)], st, { feedback, refresh: poll, scrim },
      ));
    });
    // Manifest model picker — only when the service is up (so its model APIs
    // answer). Loaded once and re-injected on subsequent renders.
    if (reachable) _injectModelPicker(serviceId, card, state);
  };

  const poll = async () => {
    if (state.cancelled) return;
    try {
      const st = await _providerStatus(serviceId);
      render(st);
      // Keep polling a bounded number of times while the service is still
      // booting / pulling its model, then stop (manual Retry still works).
      if (!st.reachable && state.polls < 8) {
        state.polls += 1;
        setTimeout(poll, 3000);
      }
    } catch (err) {
      if (state.cancelled) return;
      body.innerHTML =
        `<div style="display:flex;justify-content:space-between;">
           <strong>✓ ${escapeHtml(name)}</strong>
           <button type="button" class="dp-close" style="border:none;background:transparent;color:inherit;opacity:0.6;font-size:1.3em;cursor:pointer;">×</button>
         </div>
         <div style="margin-top:8px;opacity:0.8;font-size:0.88em;">Installed, but couldn't read provider status (${escapeHtml(String(err.message || err))}).</div>
         <button type="button" class="dp-retry" style="margin-top:12px;padding:8px 12px;border-radius:9px;border:1px solid rgba(128,128,128,0.3);background:transparent;color:inherit;cursor:pointer;">Retry</button>`;
      card.querySelector('.dp-close')?.addEventListener('click', () => { state.cancelled = true; scrim.remove(); });
      card.querySelector('.dp-retry')?.addEventListener('click', poll);
    }
  };
  poll();
}

// Manifest-driven model picker for providers whose models download on demand
// (e.g. speaches). Lists installed + pullable models from the marketplace
// /models endpoint and lets the user pull any of them. Cached on `state` and
// re-injected on re-render; refreshes itself a few seconds after a pull.
async function _injectModelPicker(serviceId, card, state) {
  const box = card.querySelector('.dp-models');
  if (!box || state.cancelled) return;
  if (!state.modelsData) {
    if (state.modelsLoading) return;
    state.modelsLoading = true;
    box.innerHTML = `<div style="opacity:0.55;font-size:0.82em;margin-top:8px;">Loading models…</div>`;
    try {
      const r = await fetch(
        `/api/marketplace/services/${encodeURIComponent(serviceId)}/models`,
        { credentials: 'same-origin' });
      state.modelsData = r.ok ? await r.json() : { supported: false };
    } catch { state.modelsData = { supported: false }; }
    state.modelsLoading = false;
    if (state.cancelled) return;
  }
  const d = state.modelsData || {};
  if (!d.supported) { box.innerHTML = ''; return; }  // bundled/auto — no picker
  const installed = Array.isArray(d.installed) ? d.installed : [];
  const available = Array.isArray(d.available) ? d.available : [];
  const pulling = Array.isArray(d.pulling) ? d.pulling : [];
  const chip = (m, css) =>
    `<span style="display:inline-block;margin:2px 4px 2px 0;padding:2px 8px;border-radius:10px;` +
    `font-size:0.78em;border:1px solid rgba(128,128,128,0.3);${css}">${escapeHtml(m)}</span>`;
  const installedChips = installed.length
    ? installed.map(m => chip(m + ' ✓', 'background:color-mix(in srgb,var(--ok,#5cb85c) 14%,transparent);')).join('')
    : `<span style="opacity:0.5;font-size:0.8em;">none yet</span>`;
  const pullingChips = pulling.filter(m => !installed.includes(m))
    .map(m => chip('⟳ ' + m, 'opacity:0.8;')).join('');
  const listId = `dp-models-${serviceId.replace(/[^a-z0-9_-]/gi, '')}`;
  const opts = available.map(m => `<option value="${escapeHtml(m)}">`).join('');
  box.innerHTML = `
    <div style="border-top:1px solid rgba(128,128,128,0.2);margin-top:12px;padding-top:10px;">
      <div style="font-size:0.82em;opacity:0.7;margin-bottom:6px;">Models${available.length ? ` · ${available.length} available` : ''}</div>
      <div style="margin-bottom:8px;">${installedChips}${pullingChips}</div>
      <div style="display:flex;gap:6px;">
        <input list="${listId}" class="dp-model-input" placeholder="Search & pull a model…"
          style="flex:1;min-width:0;padding:7px 9px;border-radius:8px;font:inherit;color:inherit;
                 background:var(--surface-2,rgba(128,128,128,0.08));border:1px solid rgba(128,128,128,0.3);">
        <datalist id="${listId}">${opts}</datalist>
        <button type="button" class="dp-model-pull"
          style="padding:7px 14px;border-radius:8px;cursor:pointer;font:inherit;color:#fff;
                 border:1px solid var(--accent,#6c8cff);background:var(--accent,#6c8cff);">Pull</button>
      </div>
      <div class="dp-model-status" style="margin-top:6px;font-size:0.8em;opacity:0.8;min-height:1em;"></div>
    </div>`;
  const input = box.querySelector('.dp-model-input');
  const status = box.querySelector('.dp-model-status');
  box.querySelector('.dp-model-pull').addEventListener('click', async () => {
    const model = (input.value || '').trim();
    if (!model) { status.textContent = 'Type or pick a model first.'; return; }
    status.textContent = `Starting download of ${model}…`;
    try {
      const r = await fetch(
        `/api/marketplace/services/${encodeURIComponent(serviceId)}/pull-model`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin', body: JSON.stringify({ model }) });
      const dd = await r.json().catch(() => ({}));
      if (r.ok && dd.ok) {
        status.textContent = `⟳ Downloading ${model}… (continues in the background)`;
        input.value = '';
        setTimeout(() => {
          if (state.cancelled) return;
          state.modelsData = null;
          _injectModelPicker(serviceId, card, state);
        }, 4000);
      } else {
        status.textContent = `✗ ${dd.error || 'Pull failed'}`;
      }
    } catch (err) {
      status.textContent = `✗ ${String(err.message || err)}`;
    }
  });
}

async function _runProviderAction(step, st, ctx) {
  if (!step) return;
  const { feedback, refresh, scrim } = ctx;
  const setFb = (m) => { if (feedback) feedback.textContent = m; };
  const pid = encodeURIComponent(st.provider_id || st.service_id || '');
  const isAudio = st.provider_type === 'stt' || st.provider_type === 'tts';
  try {
    switch (step.action) {
      case 'open_webui':
        if (step.url || st.webui) window.open(step.url || st.webui, '_blank', 'noopener');
        break;
      case 'set_default':
        if (!isAudio) { setFb('Default applies to STT/TTS only.'); break; }
        setFb('Setting default…');
        await fetch(`/api/audio/providers/${pid}`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin', body: JSON.stringify({ is_default: true }),
        });
        setFb('✓ Set as default. Refreshing…');
        setTimeout(refresh, 300);
        break;
      case 'test':
        setFb('Testing…');
        if (isAudio) {
          const r = await fetch(`/api/audio/providers/${pid}/test`, {
            method: 'POST', credentials: 'same-origin',
          });
          const d = await r.json().catch(() => ({}));
          const models = Array.isArray(d.models) ? d.models : [];
          setFb(r.ok
            ? `✓ Reachable${models.length ? ` · ${models.length} model(s): ${models.slice(0, 3).join(', ')}` : ''}`
            : `✗ ${d.error || d.detail || 'unreachable'}`);
        } else {
          setFb('Re-probing…'); setTimeout(refresh, 200);
        }
        break;
      case 'pick_model':
        if (step.url || st.webui) window.open(step.url || st.webui, '_blank', 'noopener');
        else setFb('Open the web UI to manage models.');
        break;
      case 'view_logs':
        setFb('Logs available via the Marketplace service panel.');
        break;
      case 'retry':
      case 'wait':
        setFb('Rechecking…'); setTimeout(refresh, 200);
        break;
      default:
        break;
    }
  } catch (err) {
    setFb(`✗ ${String(err.message || err)}`);
  }
}

// ── Post-install media-server setup card ────────────────────────────
//
// Media servers install EMPTY — we ship the software, never content.
// This card sets that expectation honestly and drives the one required
// setup step: add a source/library inside the server, then sync it into
// Files. Self-contained (inline-styled) so it needs no CSS file and
// stays theme-neutral, mirroring the provider next-steps card.

// Resolve the listing's user_media_servers row + admin/owner flags, then
// open the management card. Pause/start act on the catalog service_id, so
// the card still offers them even when the row can't be resolved.
async function _openMediaServerManage(listing) {
  const reg = await _loadMediaServers();
  const srv = reg.byProvider.get(_mediaServiceId(listing));
  _showMediaServerNextSteps(srv ? String(srv.id) : '', listing, {
    manage: true,
    isAdmin: reg.viewerIsAdmin,
    isOwner: srv ? !!srv.is_owned_by_viewer : false,
  });
}

// Post-install + reusable "Manage" card for a media server. opts:
//   manage   — reopened from the Discover card's ⋯ button (vs fresh install)
//   isAdmin  — viewer can pause/start the shared container + reveal creds
//   isOwner  — viewer owns this server row (can uninstall)
function _showMediaServerNextSteps(serverId, listing, opts = {}) {
  _overlay?.querySelector('.discover-media-scrim')?.remove();

  const { manage = false, isAdmin = false, isOwner = false } = opts;
  const caps = (listing && listing.capabilities) || {};
  const meta = (listing && listing.metadata) || {};
  const name = (listing && listing.title) || 'Media server';
  const serviceId = _mediaServiceId(listing);
  const guideUrl = meta.setup_guide_url || '';
  const contentNote = meta.content_note ||
    'Installs empty — you connect your own sources or files.';
  const payoff = caps.files_payoff || meta.files_payoff || '';
  // Browser-reachable "open the server" link — the dedicated HTTPS front
  // door (Caddy terminates TLS and proxies to the container); the raw host
  // port is plain HTTP and fails in a browser under HSTS.
  const serverUrl = _mediaServerUrl(listing);

  const scrim = document.createElement('div');
  scrim.className = 'discover-media-scrim';
  scrim.style.cssText =
    'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;' +
    'justify-content:center;background:rgba(0,0,0,0.5);backdrop-filter:blur(2px);';

  const managedAuth = !!(caps.managed_credentials || caps.managed_auth);
  const openSub = managedAuth
    ? `Set up your library — log in with the credentials below`
    : `Add a source &amp; a few titles at ${escapeHtml(serverUrl)}`;
  const openBtn = serverUrl
    ? `<button type="button" class="dm-open" style="display:block;width:100%;text-align:left;margin:6px 0;padding:10px 12px;
         border-radius:9px;cursor:pointer;font:inherit;color:inherit;
         border:1px solid var(--accent,#6c8cff);
         background:color-mix(in srgb,var(--accent,#6c8cff) 16%,transparent);">
         Open ${escapeHtml(name)} ▸
         <span style="display:block;opacity:0.6;font-size:0.78em;">${openSub}</span>
       </button>`
    : '';
  const guideBtn = guideUrl
    ? `<button type="button" class="dm-guide" style="display:block;width:100%;text-align:left;margin:6px 0;padding:9px 12px;
         border-radius:9px;cursor:pointer;font:inherit;color:inherit;
         border:1px solid rgba(128,128,128,0.3);background:transparent;">
         Setup guide ▸
         <span style="display:block;opacity:0.6;font-size:0.78em;">Official docs &amp; community setup advice</span>
       </button>`
    : '';
  const syncBtn = serverId
    ? `<button type="button" class="dm-sync" style="display:block;width:100%;text-align:left;margin:6px 0;padding:9px 12px;
         border-radius:9px;cursor:pointer;font:inherit;color:inherit;
         border:1px solid rgba(128,128,128,0.3);background:transparent;">
         Sync to Files
         <span style="display:block;opacity:0.6;font-size:0.78em;">Pull your library in once you've added titles</span>
       </button>`
    : '';
  // Pause/Start the shared managed container — admin-only, since it affects
  // everyone on this install. Filled in async once we know the live status.
  const lifecycleSlot = (isAdmin && serviceId)
    ? `<div class="dm-lifecycle" style="margin:6px 0;"></div>`
    : '';
  // Uninstall — admin-only (symmetric with the admin-only install). The
  // backend DELETE /api/discover/{id}/install stops the shared container,
  // removes the connection + cached library, and clears the install record
  // (install-wide), so the card flips back to "Install" and stays that way
  // on reload. Two-step confirm wired below.
  const uninstallBtn = isAdmin
    ? `<button type="button" class="dm-uninstall" style="display:block;width:100%;text-align:left;margin:6px 0;padding:9px 12px;
         border-radius:9px;cursor:pointer;font:inherit;
         border:1px solid color-mix(in srgb,var(--danger,#e5484d) 40%,transparent);
         background:transparent;color:var(--danger,#e5484d);">
         Uninstall ${escapeHtml(name)}
         <span style="display:block;opacity:0.72;font-size:0.78em;">Stops the server for everyone &amp; removes it from Augmentum — your media files are untouched.</span>
       </button>`
    : '';
  void isOwner;

  const titleLine = manage
    ? escapeHtml(name)
    : `✓ ${escapeHtml(name)} is running`;

  scrim.innerHTML = `
    <div class="discover-media-card" role="dialog" aria-label="${escapeHtml(name)} ${manage ? 'management' : 'setup'}"
         style="background:var(--surface-1,var(--bg,#16161c));color:var(--text,inherit);
                border:1px solid rgba(128,128,128,0.28);border-radius:14px;
                padding:20px 22px;max-width:460px;width:90%;
                box-shadow:0 14px 48px rgba(0,0,0,0.45);font:inherit;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
        <strong style="font-size:1.05em;">${titleLine}</strong>
        <button type="button" class="dm-close" aria-label="Close"
          style="border:none;background:transparent;color:inherit;opacity:0.6;
                 font-size:1.3em;line-height:1;cursor:pointer;">×</button>
      </div>
      <div class="dm-status" style="margin-top:6px;font-size:0.92em;color:var(--ok,#5cb85c);">● Connected to Files</div>
      <p style="margin:10px 0 2px;font-size:0.9em;opacity:0.85;">${escapeHtml(contentNote)}</p>
      ${payoff ? `<p style="margin:6px 0 0;font-size:0.84em;opacity:0.65;">${escapeHtml(payoff)}</p>` : ''}
      <div class="dm-actions" style="margin-top:14px;">${openBtn}${guideBtn}${syncBtn}${lifecycleSlot}${uninstallBtn}</div>
      <div class="dm-creds" style="margin-top:10px;"></div>
      <div class="dm-feedback" style="margin-top:8px;font-size:0.84em;min-height:1.1em;opacity:0.85;"></div>
    </div>`;
  _overlay.appendChild(scrim);

  const card = scrim.querySelector('.discover-media-card');
  const feedback = card.querySelector('.dm-feedback');
  const close = () => scrim.remove();

  scrim.addEventListener('click', (e) => { if (e.target === scrim) close(); });
  card.querySelector('.dm-close')?.addEventListener('click', close);
  card.querySelector('.dm-open')?.addEventListener('click', () => {
    if (serverUrl) window.open(serverUrl, '_blank', 'noopener');
  });
  card.querySelector('.dm-guide')?.addEventListener('click', () => {
    if (guideUrl) window.open(guideUrl, '_blank', 'noopener,noreferrer');
  });
  card.querySelector('.dm-sync')?.addEventListener('click', async () => {
    if (!serverId) return;
    if (feedback) feedback.textContent = 'Queuing sync…';
    try {
      const r = await fetch(
        `/api/media/servers/${encodeURIComponent(serverId)}/sync`,
        { method: 'POST', credentials: 'same-origin' });
      if (r.ok) {
        if (feedback) feedback.textContent =
          '⟳ Syncing your library into Files (continues in the background).';
      } else {
        const d = await r.json().catch(() => ({}));
        if (feedback) feedback.textContent = `✗ ${d.error || d.detail || `Sync failed (${r.status})`}`;
      }
    } catch (err) {
      if (feedback) feedback.textContent = `✗ ${String(err.message || err)}`;
    }
  });

  // Managed-auth servers: the container was provisioned WITH a Basic-auth
  // credential (so it's not left open on the host). Fetch + show it so the
  // user can log into the server's own console. Admin-only endpoint; on a
  // no-auth / non-managed server it returns managed_auth:false and we
  // simply render nothing.
  if (serverId) {
    (async () => {
      try {
        const r = await fetch(
          `/api/media/servers/${encodeURIComponent(serverId)}/console-credentials`,
          { credentials: 'same-origin' });
        if (!r.ok) return;
        const c = await r.json();
        if (!c || !c.managed_auth) return;
        const box = card.querySelector('.dm-creds');
        if (!box) return;
        const codeStyle =
          'font-family:ui-monospace,monospace;padding:1px 6px;border-radius:6px;' +
          'background:var(--surface-2,rgba(128,128,128,0.12));user-select:all;';
        const copyStyle =
          'border:1px solid rgba(128,128,128,0.3);background:transparent;color:inherit;' +
          'border-radius:6px;font:inherit;font-size:0.78em;padding:1px 8px;cursor:pointer;';
        const row = (label, value) => `
          <div style="display:flex;align-items:center;gap:8px;font-size:0.85em;margin-top:4px;">
            <span style="opacity:0.6;width:34px;">${label}</span>
            <code style="${codeStyle}">${escapeHtml(value)}</code>
            <button type="button" class="dm-copy" data-v="${escapeHtml(value)}" aria-label="Copy ${escapeHtml(label)}" aria-live="polite" style="${copyStyle}">Copy</button>
          </div>`;
        box.innerHTML = `
          <div style="border-top:1px solid rgba(128,128,128,0.2);padding-top:10px;">
            <div style="font-size:0.8em;opacity:0.7;">🔒 Console login (managed by Augmentum)</div>
            ${row('User', c.username || '')}
            ${row('Pass', c.password || '')}
          </div>`;
        box.querySelectorAll('.dm-copy').forEach((b) => {
          b.addEventListener('click', () => {
            try { navigator.clipboard?.writeText(b.dataset.v || ''); } catch { /* no clipboard */ }
            const prev = b.textContent;
            b.textContent = 'Copied';
            setTimeout(() => { b.textContent = prev; }, 1200);
          });
        });
      } catch { /* credentials are best-effort; card still works without them */ }
    })();
  }

  // Pause/Start the shared managed container (admin-only). Acts on the
  // catalog service_id (the container is install-wide), so it doesn't need
  // the per-user row. Fetch live status, render the right toggle, flip it on
  // click — clearly flagged as affecting everyone on this install.
  const lifeBox = card.querySelector('.dm-lifecycle');
  if (lifeBox && serviceId) {
    const btnStyle =
      'display:block;width:100%;text-align:left;padding:9px 12px;border-radius:9px;' +
      'cursor:pointer;font:inherit;color:inherit;border:1px solid rgba(128,128,128,0.3);background:transparent;';
    const renderLifecycle = (status) => {
      const running = status === 'running';
      const label = running ? 'Pause server' : 'Start server';
      const sub = running
        ? 'Stops the container for everyone on this install'
        : 'Starts the container for everyone on this install';
      lifeBox.innerHTML =
        `<button type="button" class="dm-life-btn" data-running="${running ? '1' : '0'}" style="${btnStyle}">
           ${label}<span style="display:block;opacity:0.6;font-size:0.78em;">${sub}</span>
         </button>`;
      lifeBox.querySelector('.dm-life-btn').addEventListener('click', async () => {
        const wasRunning = lifeBox.querySelector('.dm-life-btn')?.dataset.running === '1';
        const action = wasRunning ? 'disable' : 'enable';
        if (feedback) feedback.textContent = wasRunning ? 'Pausing…' : 'Starting…';
        try {
          const r = await fetch(
            `/api/marketplace/services/${encodeURIComponent(serviceId)}/${action}`,
            { method: 'POST', credentials: 'same-origin' });
          const d = await r.json().catch(() => ({}));
          if (r.ok) {
            renderLifecycle(d.status || (wasRunning ? 'stopped' : 'running'));
            if (feedback) feedback.textContent = wasRunning ? '● Paused' : '● Started';
          } else if (feedback) {
            feedback.textContent = `✗ ${d.error || d.detail || `Failed (${r.status})`}`;
          }
        } catch (err) {
          if (feedback) feedback.textContent = `✗ ${String(err.message || err)}`;
        }
      });
    };
    (async () => {
      try {
        const r = await fetch(
          `/api/marketplace/services/${encodeURIComponent(serviceId)}/status`,
          { credentials: 'same-origin' });
        const d = r.ok ? await r.json() : {};
        renderLifecycle(d.status || 'running');
      } catch { renderLifecycle('running'); }
    })();
  }

  // Uninstall (admin) — two-step confirm, then DELETE the listing install.
  // On success the container is stopped + connection removed + install
  // record cleared, so flip the card back to "Install" and close.
  const uninstallEl = card.querySelector('.dm-uninstall');
  if (uninstallEl) {
    let armed = false;
    uninstallEl.addEventListener('click', async () => {
      if (!armed) {
        armed = true;
        uninstallEl.innerHTML =
          `Click again to confirm<span style="display:block;opacity:0.72;font-size:0.78em;">Stops the server for everyone and removes it. Re-install anytime.</span>`;
        return;
      }
      uninstallEl.disabled = true;
      if (feedback) feedback.textContent = 'Uninstalling…';
      try {
        const r = await fetch(
          `/api/discover/${encodeURIComponent(listing.id)}/install`,
          { method: 'DELETE', credentials: 'same-origin' });
        const d = await r.json().catch(() => ({}));
        if (r.ok) {
          _mediaServersCache = null;          // force re-resolve next time
          const l = _findListing(listing.id);
          if (l) l.installed = false;          // flip card back to "Install"
          _rerenderCard(listing.id);
          close();
          _toast(`Uninstalled ${name}`);
        } else {
          uninstallEl.disabled = false;
          if (feedback) feedback.textContent =
            `✗ ${d.error || d.detail || `Uninstall failed (${r.status})`}`;
        }
      } catch (err) {
        uninstallEl.disabled = false;
        if (feedback) feedback.textContent = `✗ ${String(err.message || err)}`;
      }
    });
  }

  // Esc dismisses.
  const onKey = (e) => {
    if (e.key === 'Escape') {
      document.removeEventListener('keydown', onKey);
      close();
    }
  };
  document.addEventListener('keydown', onKey);
}

// ── Manifest-service setup / manage card (kind: "service") ──────────
//
// The aftercare surface Discover services were missing: after install
// (fresh) or from the installed card's ⋯ (manage), one card shows LIVE
// container status, keeps the manifest's browser promise with an Open
// front-door button, offers Start when the container is stopped or was
// deleted out-of-band (docker rm), and a two-step uninstall. Status is
// probed from the real container, so state never lies about runtime.

const _SVC_STATUS_COPY = {
  pulling:   ['◴', 'Downloading image…', 'var(--warn,#e0a23a)'],
  starting:  ['◴', 'Starting…', 'var(--warn,#e0a23a)'],
  running:   ['●', 'Running', 'var(--ok,#5cb85c)'],
  unhealthy: ['◐', 'Running — health check failing', 'var(--warn,#e0a23a)'],
  stopped:   ['■', 'Stopped — container not running', 'var(--text-tertiary,rgba(255,255,255,0.55))'],
  error:     ['✗', 'Error', 'var(--danger,#e5484d)'],
};

function _renderIntegrationRows(el, listing) {
  const system = listing.system;
  const caps = (system && Array.isArray(system.capabilities)) ? system.capabilities : [];
  if (!el || !caps.length) {
    if (el) el.innerHTML = '';
    return;
  }
  el.innerHTML = `
    <div class="discover-integration-heading">What this adds</div>
    ${caps.map(cap => `
      <div class="discover-integration-row">
        <span class="discover-integration-icon">${escapeHtml(cap.icon)}</span>
        <span class="discover-integration-label">${escapeHtml(cap.label)}
          ${cap.protocol ? `<span class="discover-integration-protocol"> · ${escapeHtml(cap.protocol)}</span>` : ''}
        </span>
        <span class="discover-integration-status">${escapeHtml(cap.status)}</span>
        ${cap.toggleable === false ? `
        <span class="discover-integration-builtin" title="Wired automatically at install — no setup">Built in</span>
        ` : `
        <label class="discover-toggle" title="Toggle ${escapeHtml(cap.label)} connection">
          <input type="checkbox" class="discover-integration-toggle"
                 data-hook="${escapeHtml(cap.hook)}"
                 data-listing-id="${escapeHtml(listing.id)}"
                 ${cap.connected ? 'checked' : ''} />
          <span class="discover-toggle-track">
            <span class="discover-toggle-thumb"></span>
          </span>
          <span class="discover-toggle-label">Connected</span>
        </label>
        `}
      </div>
    `).join('')}`;
}

// ── Staged install card (the Service Install Standard) ──────────────
// For install_via: "service_staged" services (e.g. the vLLM engine) whose
// install is a background job with staged progress. Polls /api/jobs/{id},
// renders an honest stage label + progress bar, and hands off to the standard
// management card (_showServiceCard) once the job reports ready. Closing the
// card does NOT cancel the install — the job runs on server-side and is
// restart-safe; reopening Discover shows the finished service.

// Friendly copy per stage the service_install handler emits. Unknown stages
// fall back to a title-cased version of the raw stage — never a blank spinner.
const _STAGED_STAGE_COPY = {
  'preparing': ['◴', 'Preparing…'],
  'downloading image': ['⇩', 'Downloading image — this can take a few minutes the first time'],
  'starting': ['◴', 'Starting…'],
  'warming up': ['◴', 'Warming up (first boot may download a model — this can take several minutes)…'],
  'registering engine': ['◴', 'Registering engine…'],
  'registering provider': ['◴', 'Registering provider…'],
  // Add-on build stages. Per-step labels ("building Console emulation —
  // building (12/40)") arrive verbatim from the job and fall through to the
  // capitalizing default, which is what we want: the step counter is real
  // information straight from the daemon.
  'anchoring': ['◴', 'Holding the build so Docker cleanups can\'t remove it…'],
  'ready': ['●', 'Ready'],
};

// ── Add-ons ────────────────────────────────────────────────────────
//
// An add-on has no URL and no container to manage, so it gets its own
// small surface rather than reusing the service card (which is built
// around start/stop/front-door — all meaningless here).

function _openAddonSurface(surface) {
  // Surfaces are declared in augmentum/addons/catalog.py and are always
  // in-app. Unknown values fall back to opening the Library root rather
  // than doing nothing, so the button is never dead.
  // "library:browse-games:emulator" — the optional third segment names
  // the pane's own sub-selection (a games source tab). Without it every
  // add-on landed on whichever source sorted first (js13k), which is
  // never the one the user just installed.
  const [target, selection, subId] = String(surface || '').split(':');
  closeDiscover();
  import('../library.js')
    .then((m) => m.openLibrary(
      selection ? { initialSelection: { kind: selection, id: subId || '' } } : {},
    ))
    .catch(() => _toast('Could not open the Library.', true));
  if (target && target !== 'library') {
    // Only the Library is wired today; say so instead of pretending.
    _toast(`Opened the Library — ${target} has no deep link yet.`);
  }
}

async function _showAddonCard(listing, opts = {}) {
  const { fresh = false } = opts;
  _overlay?.querySelector('.discover-svc-scrim')?.remove();

  // Refetch so disk/anchor state is current rather than whatever the grid
  // cached before the build ran.
  let l = listing;
  try {
    const r = await fetch(`/api/discover/${encodeURIComponent(listing.id)}`,
                          { credentials: 'same-origin' });
    if (r.ok) {
      const d = await r.json();
      if (d && d.listing) l = { ...listing, ...d.listing };
    }
  } catch { /* cached copy is good enough for the sheet */ }

  const caps = l.capabilities || {};
  const name = l.title || 'Add-on';
  const gb = caps.disk_mb ? (Number(caps.disk_mb) / 1000).toFixed(1) : '';
  const scrim = document.createElement('div');
  scrim.className = 'discover-svc-scrim';
  scrim.style.cssText =
    'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;' +
    'justify-content:center;background:rgba(0,0,0,0.5);backdrop-filter:blur(2px);';

  // "At risk" = the image exists but nothing holds it, which is exactly the
  // state that preceded the 2026-07-25 sweep. Offer the fix here rather than
  // letting it surface as a launch failure weeks later.
  const atRisk = caps.at_risk
    ? `<div class="dac-risk" style="margin-top:12px;padding:10px 12px;border-radius:9px;
          background:rgba(242,163,92,0.12);border:1px solid rgba(242,163,92,0.4);font-size:0.85em;">
         Built, but not held against Docker cleanups — a host <code>docker image prune</code>
         would remove it. Reinstalling re-anchors it without rebuilding.
       </div>`
    : '';

  scrim.innerHTML = `
    <div class="discover-svc-card" role="dialog" aria-label="${escapeHtml(name)}"
         style="background:var(--surface-1,var(--bg,#16161c));color:var(--text,inherit);
                border:1px solid rgba(128,128,128,0.28);border-radius:14px;
                padding:20px 22px;max-width:480px;width:90%;
                box-shadow:0 14px 48px rgba(0,0,0,0.45);font:inherit;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
        <strong style="font-size:1.05em;">${fresh ? 'Added' : ''} ${escapeHtml(name)}</strong>
        <button type="button" class="ds-close" aria-label="Close"
          style="border:none;background:transparent;color:inherit;opacity:0.6;
                 font-size:1.3em;line-height:1;cursor:pointer;">×</button>
      </div>
      <p style="margin:10px 0 0;font-size:0.9em;opacity:0.85;">
        Augmentum can now ${escapeHtml(String(caps.provides || 'use this capability'))}.
        Nothing runs until you use it.
      </p>
      ${atRisk}
      <dl style="margin:14px 0 0;display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:0.85em;">
        <dt style="opacity:0.6;">Disk</dt><dd style="margin:0;">${gb ? `${escapeHtml(gb)} GB` : '—'}</dd>
        <dt style="opacity:0.6;">Held</dt><dd style="margin:0;">${caps.anchored ? 'Yes — safe from Docker cleanups' : 'No'}</dd>
      </dl>
      <div style="display:flex;gap:8px;margin-top:16px;">
        ${caps.surface ? `<button type="button" class="dac-open" style="flex:1;padding:10px 12px;
           border-radius:9px;cursor:pointer;font:inherit;color:inherit;
           border:1px solid var(--accent,#6c8cff);background:transparent;">Open &#9656;</button>` : ''}
        <button type="button" class="dac-remove" style="flex:1;padding:10px 12px;border-radius:9px;
           cursor:pointer;font:inherit;color:var(--danger,#e5484d);
           border:1px solid rgba(229,72,77,0.5);background:transparent;">Remove</button>
      </div>
      <p class="dac-note" style="margin:10px 0 0;font-size:0.78em;opacity:0.55;">
        Removing deletes the build${gb ? ` (reclaims ~${escapeHtml(gb)} GB)` : ''} and turns the
        capability off in Augmentum. Your saved data is kept.
      </p>
    </div>`;
  _overlay.appendChild(scrim);

  const card = scrim.querySelector('.discover-svc-card');
  const close = () => scrim.remove();
  scrim.addEventListener('click', (e) => { if (e.target === scrim) close(); });
  card.querySelector('.ds-close').addEventListener('click', close);
  card.querySelector('.dac-open')?.addEventListener('click', () => {
    close();
    _openAddonSurface(String(caps.surface || ''));
  });
  card.querySelector('.dac-remove')?.addEventListener('click', async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    btn.textContent = 'Removing…';
    try {
      const r = await fetch(`/api/discover/${encodeURIComponent(l.id)}/install`,
                            { method: 'DELETE', credentials: 'same-origin' });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.error || err.detail || `HTTP ${r.status}`);
      }
      const listingRef = _findListing(l.id);
      if (listingRef) listingRef.installed = false;
      close();
      _toast(`Removed ${name}.`);
      _rerenderCard(l.id);
    } catch (err) {
      btn.disabled = false;
      btn.textContent = 'Remove';
      _toast(`Couldn't remove ${name}: ${err.message || err}`, true);
    }
  });
}

function _showStagedInstallCard(jobId, listing) {
  _overlay?.querySelector('.discover-svc-scrim')?.remove();
  const name = listing.title || 'Engine';

  const scrim = document.createElement('div');
  scrim.className = 'discover-svc-scrim';
  scrim.style.cssText =
    'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;' +
    'justify-content:center;background:rgba(0,0,0,0.5);backdrop-filter:blur(2px);';
  scrim.innerHTML = `
    <div class="discover-svc-card" role="dialog" aria-label="${escapeHtml(name)} install"
         style="background:var(--surface-1,var(--bg,#16161c));color:var(--text,inherit);
                border:1px solid rgba(128,128,128,0.28);border-radius:14px;
                padding:20px 22px;max-width:460px;width:90%;
                box-shadow:0 14px 48px rgba(0,0,0,0.45);font:inherit;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
        <strong style="font-size:1.05em;">Installing ${escapeHtml(name)}</strong>
        <button type="button" class="ds-close" aria-label="Close"
          style="border:none;background:transparent;color:inherit;opacity:0.6;
                 font-size:1.3em;line-height:1;cursor:pointer;">×</button>
      </div>
      <div class="dsi-stage" style="margin-top:10px;font-size:0.92em;opacity:0.9;">◴ Preparing…</div>
      <div style="margin-top:10px;height:8px;border-radius:6px;overflow:hidden;
                  background:rgba(128,128,128,0.22);">
        <div class="dsi-fill" style="height:100%;width:2%;border-radius:6px;
             background:var(--accent,#6c8cff);transition:width 0.4s ease;"></div>
      </div>
      <div class="dsi-note" style="margin-top:10px;font-size:0.82em;opacity:0.6;">
        You can close this — the install keeps running and finishes on its own.
      </div>
      <div class="dsi-actions" style="margin-top:14px;"></div>
    </div>`;
  _overlay.appendChild(scrim);

  const card = scrim.querySelector('.discover-svc-card');
  const stageEl = card.querySelector('.dsi-stage');
  const fillEl = card.querySelector('.dsi-fill');
  const actionsEl = card.querySelector('.dsi-actions');
  const state = { cancelled: false };
  const close = () => { state.cancelled = true; scrim.remove(); };
  scrim.addEventListener('click', (e) => { if (e.target === scrim) close(); });
  card.querySelector('.ds-close').addEventListener('click', close);
  const onKey = (e) => {
    if (e.key === 'Escape') { document.removeEventListener('keydown', onKey); close(); }
  };
  document.addEventListener('keydown', onKey);

  const poll = async () => {
    if (state.cancelled) return;
    let job = null;
    try {
      const r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`,
                            { credentials: 'same-origin' });
      if (r.ok) job = await r.json();
    } catch { /* transient — next tick retries */ }
    if (state.cancelled) return;

    if (job) {
      const status = String(job.status || '');
      const stage = String(job.stage || '');
      const frac = Math.max(0, Math.min(1, Number(job.progress) || 0));
      const [dot, label] = _STAGED_STAGE_COPY[stage]
        || ['◴', stage ? stage.charAt(0).toUpperCase() + stage.slice(1) : 'Working…'];
      stageEl.innerHTML = `${dot} ${escapeHtml(label)}`;
      fillEl.style.width = `${Math.round(frac * 100) || 2}%`;

      if (status === 'completed') {
        stageEl.innerHTML = '● Ready';
        fillEl.style.width = '100%';
        document.removeEventListener('keydown', onKey);
        close();
        // Hand off to the right post-install card: provider services get the
        // provider next-steps card (set default / test / model readiness);
        // everything else gets the standard management card.
        if (listing.kind === 'addon') {
          if (listing) listing.installed = true;
          _rerenderCard(listing.id);
          _showAddonCard(listing, { fresh: true });
        } else if (listing.kind === 'provider_service') {
          const sid = (listing.install_payload || {}).service_id || '';
          _showProviderNextSteps(sid, listing);
        } else {
          _showServiceCard(listing, { fresh: true });
        }
        return;
      }
      if (status === 'failed' || status === 'cancelled') {
        stageEl.innerHTML = `<span style="color:var(--danger,#e5484d);">✗ ${escapeHtml(job.error || 'Install failed')}</span>`;
        fillEl.style.background = 'var(--danger,#e5484d)';
        actionsEl.innerHTML =
          `<button type="button" class="dsi-retry" style="display:block;width:100%;
             padding:10px 12px;border-radius:9px;cursor:pointer;font:inherit;color:inherit;
             border:1px solid var(--accent,#6c8cff);background:transparent;">Retry</button>`;
        actionsEl.querySelector('.dsi-retry')?.addEventListener('click', () => {
          close();
          _installListing(listing.id);
        });
        return;
      }
    }
    setTimeout(poll, 1500);
  };
  poll();
}

async function _showServiceCard(listing, opts = {}) {
  const { fresh = false } = opts;
  _overlay?.querySelector('.discover-svc-scrim')?.remove();

  // Refetch the enriched listing: post-install the catalog attaches the
  // allocated front-door port + truthful installed state, which the
  // pre-install cached copy doesn't have.
  let l = listing;
  try {
    const r = await fetch(`/api/discover/${encodeURIComponent(listing.id)}`,
                          { credentials: 'same-origin' });
    if (r.ok) {
      const d = await r.json();
      if (d.listing) {
        l = { ...listing, ...d.listing };
        const cached = _findListing(listing.id);
        if (cached) Object.assign(cached, d.listing);
        _rerenderCard(listing.id);
      }
    }
  } catch { /* card still renders from the cached listing */ }

  const svcId = _svcServiceId(l);
  const name = l.title || 'App';
  const browser = (l.install_payload || {}).browser || {};
  const afterCopy = AFTER_INSTALL_COPY[browser.after_install] || '';
  const url = _serviceUrl(l);

  const scrim = document.createElement('div');
  scrim.className = 'discover-svc-scrim';
  scrim.style.cssText =
    'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;' +
    'justify-content:center;background:rgba(0,0,0,0.5);backdrop-filter:blur(2px);';
  const btn = (cls, label, sub, style = '') =>
    `<button type="button" class="${cls}" style="display:block;width:100%;text-align:left;margin:6px 0;
       padding:10px 12px;border-radius:9px;cursor:pointer;font:inherit;color:inherit;
       border:1px solid rgba(128,128,128,0.3);background:transparent;${style}">
       ${label}${sub ? `<span style="display:block;opacity:0.6;font-size:0.78em;">${sub}</span>` : ''}
     </button>`;
  scrim.innerHTML = `
    <div class="discover-svc-card" role="dialog" aria-label="${escapeHtml(name)} management"
         style="background:var(--surface-1,var(--bg,#16161c));color:var(--text,inherit);
                border:1px solid rgba(128,128,128,0.28);border-radius:14px;
                padding:20px 22px;max-width:460px;width:90%;
                box-shadow:0 14px 48px rgba(0,0,0,0.45);font:inherit;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
        <strong style="font-size:1.05em;">${fresh ? '✓ ' : ''}${escapeHtml(name)}${fresh ? ' is installed' : ''}</strong>
        <button type="button" class="ds-close" aria-label="Close"
          style="border:none;background:transparent;color:inherit;opacity:0.6;
                 font-size:1.3em;line-height:1;cursor:pointer;">×</button>
      </div>
      <div class="ds-status" style="margin-top:6px;font-size:0.92em;opacity:0.85;">◴ Checking status…</div>
      ${afterCopy ? `<p style="margin:10px 0 0;font-size:0.9em;opacity:0.85;">${escapeHtml(afterCopy)}</p>` : ''}
      <div class="ds-integration" style="margin-top:12px;"></div>
      <div class="ds-actions" style="margin-top:14px;"></div>
      <div class="ds-feedback" style="margin-top:8px;font-size:0.84em;min-height:1.1em;opacity:0.85;"></div>
    </div>`;
  _overlay.appendChild(scrim);

  const card = scrim.querySelector('.discover-svc-card');
  const statusEl = card.querySelector('.ds-status');
  const actionsEl = card.querySelector('.ds-actions');
  const feedback = card.querySelector('.ds-feedback');
  const integrationEl = card.querySelector('.ds-integration');

  // Render integration capabilities from system block.
  _renderIntegrationRows(integrationEl, l);
  const state = { cancelled: false, polls: 0, status: '' };
  const close = () => { state.cancelled = true; scrim.remove(); };
  scrim.addEventListener('click', (e) => { if (e.target === scrim) close(); });
  card.querySelector('.ds-close').addEventListener('click', close);
  const onKey = (e) => {
    if (e.key === 'Escape') { document.removeEventListener('keydown', onKey); close(); }
  };
  document.addEventListener('keydown', onKey);

  const renderActions = () => {
    const stopped = ['stopped', 'error'].includes(state.status);
    const openBtn = url
      ? btn('ds-open', `Open ${escapeHtml(name)} ▸`,
            afterCopy || `Served over HTTPS at ${escapeHtml(url)}`,
            'border-color:var(--accent,#6c8cff);background:color-mix(in srgb,var(--accent,#6c8cff) 16%,transparent);')
      : l.kind === 'provider_service'
        ? `<div style="font-size:0.84em;opacity:0.65;margin:6px 0;">Runs headless — consumed internally by the voice pipeline (no browser page).</div>`
        : `<div style="font-size:0.84em;opacity:0.65;margin:6px 0;">No browser address available — the HTTPS front door wasn't allocated. Check server logs.</div>`;
    const startBtn = stopped
      ? btn('ds-start', 'Start', 'Recreates the container with its saved settings — your data volume is intact')
      : '';
    const caps = (l && l.capabilities) || {};
    const updateBtn = caps.update_available
      ? btn('ds-update', 'Update available ▸',
            'Recreates the container on the catalog’s newer image — your data volume is kept.',
            'border-color:color-mix(in srgb,var(--accent,#6c8cff) 40%,transparent);')
      : '';
    const uninstallBtn = btn('ds-uninstall', `Uninstall ${escapeHtml(name)}`,
        'Stops and removes the container for everyone. Its data volumes are kept.',
        'border-color:color-mix(in srgb,var(--danger,#e5484d) 40%,transparent);color:var(--danger,#e5484d);');
    actionsEl.innerHTML = `${openBtn}${startBtn}${updateBtn}${uninstallBtn}`;

    actionsEl.querySelector('.ds-update')?.addEventListener('click', async (e) => {
      const el = e.currentTarget;
      el.disabled = true;
      feedback.textContent = 'Updating… (pulling image, brief downtime)';
      try {
        const r = await fetch(`/api/discover/${encodeURIComponent(l.id)}/update`,
                              { method: 'POST', credentials: 'same-origin' });
        const d = await r.json().catch(() => ({}));
        if (r.ok) {
          const cached = _findListing(l.id);
          if (cached && cached.capabilities) {
            cached.capabilities.update_available = false;
            cached.capabilities.installed_image = cached.capabilities.target_image;
          }
          feedback.textContent = '✓ Updated';
          state.polls = 0; poll();
          renderActions();
        } else if (r.status === 403) {
          el.disabled = false;
          feedback.textContent = '✗ Admin only.';
        } else {
          el.disabled = false;
          feedback.textContent = `✗ ${d.error || d.detail || `Update failed (${r.status})`}`;
        }
      } catch (err) {
        el.disabled = false;
        feedback.textContent = `✗ ${String(err.message || err)}`;
      }
    });

    actionsEl.querySelector('.ds-open')?.addEventListener('click', () => {
      window.open(url, '_blank', 'noopener');
    });
    actionsEl.querySelector('.ds-start')?.addEventListener('click', async () => {
      feedback.textContent = 'Starting…';
      try {
        const r = await fetch(
          `/api/marketplace/services/${encodeURIComponent(svcId)}/enable`,
          { method: 'POST', credentials: 'same-origin' });
        if (r.ok) { feedback.textContent = '● Starting'; state.polls = 0; poll(); }
        else if (r.status === 403) feedback.textContent = '✗ Admin only.';
        else {
          const d = await r.json().catch(() => ({}));
          feedback.textContent = `✗ ${d.error || d.detail || `Failed (${r.status})`}`;
        }
      } catch (err) { feedback.textContent = `✗ ${String(err.message || err)}`; }
    });
    const un = actionsEl.querySelector('.ds-uninstall');
    let armed = false;
    un?.addEventListener('click', async () => {
      if (!armed) {
        armed = true;
        un.innerHTML = `Click again to confirm<span style="display:block;opacity:0.72;font-size:0.78em;">Removes the container; data volumes stay. Re-install anytime.</span>`;
        return;
      }
      un.disabled = true;
      feedback.textContent = 'Uninstalling…';
      try {
        const r = await fetch(`/api/discover/${encodeURIComponent(l.id)}/install`,
                              { method: 'DELETE', credentials: 'same-origin' });
        const d = await r.json().catch(() => ({}));
        if (r.ok) {
          const cached = _findListing(l.id);
          if (cached) cached.installed = false;
          _rerenderCard(l.id);
          close();
          _toast(`Uninstalled ${name}`);
        } else {
          un.disabled = false;
          feedback.textContent = `✗ ${d.error || d.detail || `Uninstall failed (${r.status})`}`;
        }
      } catch (err) {
        un.disabled = false;
        feedback.textContent = `✗ ${String(err.message || err)}`;
      }
    });
  };

  const poll = async () => {
    if (state.cancelled || !svcId) return;
    let status = '';
    try {
      const r = await fetch(
        `/api/marketplace/services/${encodeURIComponent(svcId)}/status`,
        { credentials: 'same-origin' });
      if (r.ok) status = String((await r.json()).status || '');
    } catch { /* transient — next poll retries */ }
    if (state.cancelled) return;
    const [dot, label, color] = _SVC_STATUS_COPY[status]
      || ['◌', status || 'Unknown', 'var(--text-tertiary,rgba(255,255,255,0.55))'];
    statusEl.innerHTML = `<span style="color:${color};">${dot} ${escapeHtml(label)}</span>`;
    if (status !== state.status) { state.status = status; renderActions(); }
    // Keep polling through transitional states (image pull can run
    // minutes) — the copy above IS the progress indicator.
    if (['', 'pulling', 'starting'].includes(status) && state.polls < 100) {
      state.polls += 1;
      setTimeout(poll, 3000);
    }
  };
  poll();
}

// Replace one card in place from its current listing state (busy /
// installed / default). Used after an install starts and finishes so the
// button + installed pip stay truthful without re-fetching the catalog.
function _rerenderCard(listingId) {
  const listing = _findListing(listingId);
  if (!listing) return;
  const selector = `.discover-card[data-id="${CSS.escape(listingId)}"]`;
  _overlay?.querySelectorAll(selector).forEach((card) => {
    const tmp = document.createElement('div');
    tmp.innerHTML = _renderCard(listing);
    const fresh = tmp.firstElementChild;
    if (!fresh) return;
    card.replaceWith(fresh);
    _wireListingActions(fresh);
  });
  _syncInstallButtons(listingId, listing);
}

function _syncInstallButtons(listingId, listing) {
  const selector = `.discover-install-btn[data-id="${CSS.escape(listingId)}"]`;
  _overlay?.querySelectorAll(selector).forEach((btn) => {
    const extras = [];
    if (btn.classList.contains('discover-mini-install')) extras.push('discover-mini-install');
    if (btn.classList.contains('discover-spotlight-action')) extras.push('discover-spotlight-action');
    const tmp = document.createElement('div');
    tmp.innerHTML = _renderInstallButton(listing, extras.join(' '));
    const fresh = tmp.firstElementChild;
    if (!fresh) return;
    btn.replaceWith(fresh);
  });
  if (_overlay) _wireListingActions(_overlay);
}


// Shimmer placeholders during the catalog fetch. Only paints on a cold
// grid (first open / hard refresh) — on filter or search changes the
// existing cards stay put rather than flashing to skeletons and back.
function _renderSkeleton() {
  const grid = _overlay?.querySelector('#discover-grid');
  if (!grid || grid.children.length) return;
  const empty = _overlay.querySelector('#discover-empty');
  if (empty) empty.hidden = true;
  let cells = '';
  for (let i = 0; i < 8; i++) {
    cells += `
      <div class="discover-card discover-card-skel" aria-hidden="true">
        <span class="skeleton discover-skel-icon"></span>
        <div class="discover-card-body">
          <span class="skeleton skel-line discover-skel-title"></span>
          <span class="skeleton skel-line discover-skel-tag"></span>
          <span class="skeleton discover-skel-btn"></span>
        </div>
      </div>`;
  }
  grid.innerHTML = cells;
}

function _renderEmpty() {
  const empty = _overlay.querySelector('#discover-empty');
  if (!empty) return;
  const scoped = _activeCategory || _searchQuery;
  const sub = scoped
    ? 'Try another category or a different search.'
    : 'Catalog items will appear here as they’re published.';
  empty.innerHTML = `
    <svg class="discover-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" width="40" height="40">
      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
    <p class="discover-empty-title">${scoped ? 'No matches' : 'Nothing here yet'}</p>
    <p class="discover-empty-sub">${escapeHtml(sub)}</p>`;
  empty.hidden = false;
}

function _renderError(msg) {
  const grid = _overlay.querySelector('#discover-grid');
  const empty = _overlay.querySelector('#discover-empty');
  grid.innerHTML = '';
  empty.innerHTML = `
    <svg class="discover-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" width="40" height="40">
      <circle cx="12" cy="12" r="9"/><line x1="12" y1="8" x2="12" y2="13"/><line x1="12" y1="16.5" x2="12.01" y2="16.5"/>
    </svg>
    <p class="discover-empty-title">Couldn’t load Discover</p>
    <p class="discover-empty-sub">${escapeHtml(msg)}</p>`;
  empty.hidden = false;
}


function _toast(msg, isError = false) {
  // Lightweight in-overlay toast. App's global toast helper would be
  // nicer but Discover may be the first surface a user sees so we
  // stay self-contained.
  const t = document.createElement('div');
  t.className = `discover-toast${isError ? ' is-error' : ''}`;
  t.textContent = msg;
  _overlay.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}
