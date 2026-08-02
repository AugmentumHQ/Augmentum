/* ==========================================================================
   Augmentum — API Cache
   Generic fetch-once cache with TTL, dedup, and change detection.
   All read-heavy endpoints go through here instead of scattered fetches.
   ========================================================================== */

// ---------------------------------------------------------------------------
// Generic cache entry
// ---------------------------------------------------------------------------

// Stable content fingerprint used for change detection. Order-sensitive
// on purpose — a reordered catalog is a meaningful change for the
// "recently used" grouping in the picker. Handles arrays of strings
// (voice names), arrays of objects (models), and plain objects (tool
// settings). Falls back to length if the shape is unexpected.
function _signature(items) {
  if (!Array.isArray(items)) {
    try { return JSON.stringify(items); } catch { return 'obj'; }
  }
  try {
    return items
      .map(i => (typeof i === 'string' ? i : (i && (i.name || i.model || i.id)) || ''))
      .join('');
  } catch {
    return String(items.length);
  }
}

function createCache(url, { ttl = 30_000, extract = r => r } = {}) {
  let data = null;
  let count = 0;
  let lastFetch = 0;
  let lastSig = null;
  let promise = null;
  const listeners = [];

  async function doFetch() {
    try {
      const resp = await fetch(url);
      if (!resp.ok) return;
      const raw = await resp.json();
      const items = extract(raw);
      // Change detection by content signature, NOT length. A model
      // renamed/swapped, or a provider's passthrough set changing, keeps
      // the same count but different contents — the old length-only test
      // silently pinned the stale list until a hard refresh (which the
      // installed PWA can't do). ``lastSig`` starts null so the FIRST
      // populate also notifies: that's the hook that heals a cold boot
      // where consumers rendered before the list arrived.
      const sig = _signature(items);
      const changed = sig !== lastSig;
      data = items;
      count = items.length;
      lastSig = sig;
      lastFetch = Date.now();
      // Phase 8 — stash the model list globally so the chat renderer
      // can do a synchronous peer-icon lookup at message-render time
      // without re-fetching. Only the /api/tags cache populates this
      // (the chat renderer never cares about the other cached endpoints).
      if (url === '/api/tags') {
        try { window.__augmentumCachedModels = items; } catch {}
      }
      if (changed) listeners.forEach(cb => { try { cb(items); } catch {} });
    } catch { /* endpoint unavailable */ }
  }

  function refresh() {
    if (promise) return promise;
    promise = doFetch().finally(() => { promise = null; });
    return promise;
  }

  return {
    async get(force = false) {
      if (data && !force && Date.now() - lastFetch < ttl) return data;
      await refresh();
      return data || [];
    },
    getSync() {
      if (data && Date.now() - lastFetch > ttl) refresh();
      return data;
    },
    invalidate() { return refresh(); },
    onChange(cb) { listeners.push(cb); },
    get cached() { return data; },
  };
}

// ---------------------------------------------------------------------------
// Cache instances
// ---------------------------------------------------------------------------

const _filterBase = models => models.filter(m => {
  const n = m.name || m.model || '';
  return n && !n.startsWith('a/') && !n.startsWith('n/') && !n.startsWith('p/');
});

// LLM models — /api/tags returns { models: [...] }
// We store the full list and expose a filtered getter
const _allModels = createCache('/api/tags', {
  ttl: 300_000,  // 5min — only changes on explicit model pull/delete
  extract: r => r.models || [],
});

// TTS voices — /api/audio/voices returns [...]
const _voices = createCache('/api/audio/voices', {
  ttl: 300_000,  // 5min — only changes on voice clone/delete
  extract: r => r,  // response is already an array
});

// Local image models — /api/image/models returns [...]
const _imageModels = createCache('/api/image/models', {
  ttl: 300_000,  // 5min — only changes on explicit download
  extract: r => Array.isArray(r) ? r : r.models || [],
});

// Cloud image models — /api/image/cloud/models returns [...]
const _cloudImageModels = createCache('/api/image/cloud/models', {
  ttl: 600_000,  // 10min — provider list is basically static
  extract: r => Array.isArray(r) ? r : r.models || r.providers || [],
});

// Tool settings — /api/config/tools returns {...}
const _toolSettings = createCache('/api/config/tools', {
  ttl: 60_000,  // 1min — refreshes after settings save via invalidate()
  extract: r => r,  // keep as-is (object, not array)
});

// ---------------------------------------------------------------------------
// Public API — named exports for each resource
// ---------------------------------------------------------------------------

// LLM Models (base only, no a/n/p prefixes)
export async function getModels(force = false) {
  const all = await _allModels.get(force);
  return _filterBase(all);
}
export function getModelsSync() {
  const all = _allModels.getSync();
  return all ? _filterBase(all) : null;
}

// LLM Models (full list including prefix variants)
export async function getAllModels(force = false) { return _allModels.get(force); }

// TTS Voices
export async function getVoices(force = false) { return _voices.get(force); }
export function getVoicesSync() { return _voices.getSync(); }

// Image Models (local)
export async function getImageModels(force = false) { return _imageModels.get(force); }
export function getImageModelsSync() { return _imageModels.getSync(); }

// Image Models (cloud)
export async function getCloudImageModels(force = false) { return _cloudImageModels.get(force); }

// Tool Settings
export async function getToolSettings(force = false) { return _toolSettings.get(force); }

// ---------------------------------------------------------------------------
// Invalidation
// ---------------------------------------------------------------------------

/** Invalidate specific cache by name, or all if no arg. */
export function invalidate(name) {
  const map = { models: _allModels, voices: _voices, imageModels: _imageModels, cloudImageModels: _cloudImageModels, tools: _toolSettings };
  if (name && map[name]) return map[name].invalidate();
  return Promise.all(Object.values(map).map(c => c.invalidate()));
}

// ---------------------------------------------------------------------------
// Change listeners
// ---------------------------------------------------------------------------

export function onChange(name, cb) {
  const map = { models: _allModels, voices: _voices, imageModels: _imageModels, cloudImageModels: _cloudImageModels, tools: _toolSettings };
  if (map[name]) map[name].onChange(cb);
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

/** Warm critical caches at app init. */
export function warmup() {
  _allModels.invalidate();
  _voices.invalidate();
  // Image + tools fetched lazily on first use

  // Refresh all caches when page regains focus (tab switch, phone unlock)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      invalidate();
    }
  });

  // Real-time invalidation off the /api/system/events SSE bus
  // (ui/scripts/system-events.js re-dispatches server topics as
  // ``system-event:<topic>`` window events). The server already knows the
  // moment a model finishes installing — or a provider is added/removed on
  // ANY client — so catch it and refetch instead of waiting out the 5-min
  // TTL. This is the keystone for the installed PWA, which has no manual
  // refresh: invalidate() refetches and, via the signature change-detection
  // above, fires onChange so the chat model dropdown + composer label
  // update live. Until now this SSE bus had zero subscribers.
  const refetchModels = () => { _allModels.invalidate(); };
  window.addEventListener('system-event:models.installed', refetchModels);
  window.addEventListener('system-event:models.install_failed', refetchModels);
  // The server's model-map prober fills in / verifies the catalog a few
  // seconds after a cold restart (cloud backends like deepseek/openrouter are
  // slow on the first probe). It emits models.changed when the map actually
  // changes — refetch so the picker shows the full list without the user
  // having to manually refresh twice. (provider_registry.refresh_model_map)
  window.addEventListener('system-event:models.changed', refetchModels);
  // A provider mutation changes which passthrough models /api/tags
  // returns, so the catalog must refetch on provider topics too.
  window.addEventListener('system-event:providers.added', refetchModels);
  window.addEventListener('system-event:providers.updated', refetchModels);
  window.addEventListener('system-event:providers.deleted', refetchModels);

  // Voices change when this user clones/mixes a voice or an admin adds/
  // removes an audio provider (audio_routes.py emits voices.changed). The
  // voice picker subscribes via onChange('voices') and re-renders live.
  window.addEventListener('system-event:voices.changed', () => { _voices.invalidate(); });

  // Local image models change on pull/import/upload/delete (image_routes.py
  // emits image.models.changed). Guard on .cached so we only refetch if the
  // image panel was actually used this session.
  window.addEventListener('system-event:image.models.changed', () => {
    if (_imageModels.cached) _imageModels.invalidate();
  });

  // Cloud image models follow their provider list.
  const refetchCloudImageModels = () => {
    if (_cloudImageModels.cached) _cloudImageModels.invalidate();
  };
  window.addEventListener('system-event:image_providers.added', refetchCloudImageModels);
  window.addEventListener('system-event:image_providers.updated', refetchCloudImageModels);
  window.addEventListener('system-event:image_providers.deleted', refetchCloudImageModels);
}
