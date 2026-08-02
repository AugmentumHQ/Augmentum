/**
 * consumption/library-client.js — thin wrapper around /api/cast/library/home.
 *
 * Used by Media (in-app) and (eventually, after extraction) cast-control
 * and cast-app. Tiny in-memory cache keyed by receiverId; 5-minute TTL
 * matches cast-control's polling cadence.
 */

const CACHE_TTL_MS = 5 * 60 * 1000;
const _cache = new Map();  // receiverId -> { at, payload }

export async function fetchHome({ receiverId = '', force = false } = {}) {
  const key = receiverId || '';
  const now = Date.now();
  if (!force) {
    const hit = _cache.get(key);
    if (hit && (now - hit.at) < CACHE_TTL_MS) {
      return hit.payload;
    }
  }
  const qs = receiverId ? `?trusted_id=${encodeURIComponent(receiverId)}` : '';
  const resp = await fetch(`/api/cast/library/home${qs}`, {
    credentials: 'same-origin',
    headers: { 'Accept': 'application/json' },
  });
  if (!resp.ok) {
    throw new Error(`library/home ${resp.status}`);
  }
  const payload = await resp.json();
  _cache.set(key, { at: now, payload });
  return payload;
}

export function clearHomeCache() {
  _cache.clear();
}
