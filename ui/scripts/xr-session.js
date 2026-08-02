/* ==========================================================================
   Augmentum XR session client
   Server-backed room/session spine for browser WebXR/PWA launches.
   ========================================================================== */

let _capabilitiesPromise = null;

function _jsonHeaders() {
  return { 'Content-Type': 'application/json' };
}

async function _readJson(resp) {
  const text = await resp.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

export function isPwaLaunchContext() {
  try {
    if (window.matchMedia?.('(display-mode: standalone)').matches) return true;
    if (window.navigator?.standalone) return true;
    // Meta documents Digital Goods availability as a practical PWA-scope
    // signal for WebXR PWA auto-launch checks.
    if (window.getDigitalGoodsService !== undefined) return true;
  } catch {}
  return false;
}

export function buildDeviceHint() {
  const nav = window.navigator || {};
  return {
    userAgent: nav.userAgent || '',
    platform: nav.platform || '',
    language: nav.language || '',
    standalone: isPwaLaunchContext(),
    secureContext: !!window.isSecureContext,
    viewport: {
      width: window.innerWidth || 0,
      height: window.innerHeight || 0,
      devicePixelRatio: window.devicePixelRatio || 1,
    },
  };
}

export async function getCapabilities({ refresh = false, signal } = {}) {
  if (!_capabilitiesPromise || refresh) {
    _capabilitiesPromise = fetch('/api/xr/capabilities', { signal })
      .then(async (resp) => {
        const body = await _readJson(resp);
        if (!resp.ok) {
          throw new Error(body.error || `XR capabilities failed (${resp.status})`);
        }
        return body;
      });
  }
  return _capabilitiesPromise;
}

export async function createSession(payload = {}, { signal } = {}) {
  const resp = await fetch('/api/xr/sessions', {
    method: 'POST',
    headers: _jsonHeaders(),
    signal,
    body: JSON.stringify({
      surface: 'voice',
      room_id: 'modern-room',
      seat_id: 'default',
      device_hint: buildDeviceHint(),
      pwa: isPwaLaunchContext(),
      ...payload,
    }),
  });
  const body = await _readJson(resp);
  if (!resp.ok) throw new Error(body.error || `XR session failed (${resp.status})`);
  return body.session;
}

export async function patchSession(sessionId, patch = {}) {
  if (!sessionId) return null;
  try {
    const resp = await fetch(`/api/xr/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'PATCH',
      headers: _jsonHeaders(),
      body: JSON.stringify(patch),
    });
    const body = await _readJson(resp);
    if (!resp.ok) return null;
    return body.session || null;
  } catch {
    return null;
  }
}

export async function recordEvent(sessionId, type, payload = {}) {
  if (!sessionId || !type) return false;
  try {
    const resp = await fetch(`/api/xr/sessions/${encodeURIComponent(sessionId)}/events`, {
      method: 'POST',
      headers: _jsonHeaders(),
      body: JSON.stringify({ type, payload }),
    });
    return resp.ok;
  } catch {
    return false;
  }
}

export async function saveSeat(seatId = 'default', seat = {}) {
  const resp = await fetch(`/api/xr/seats/${encodeURIComponent(seatId)}`, {
    method: 'PUT',
    headers: _jsonHeaders(),
    body: JSON.stringify(seat),
  });
  const body = await _readJson(resp);
  if (!resp.ok) throw new Error(body.error || `XR seat save failed (${resp.status})`);
  return body.seat;
}
