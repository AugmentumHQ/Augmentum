/* notifications/push-subscribe.js — Web Push subscription client.
 *
 * Three jobs:
 *
 *   1. Register the dedicated SW at /notification-sw.js (NOT
 *      registering at "/" — we don't want this SW intercepting any
 *      fetches; it's purely a push handler).
 *   2. Request Notification permission + subscribe with the
 *      server's VAPID public key.
 *   3. POST the subscription endpoint + keys to
 *      /api/notify/subscriptions so the server can target this
 *      browser when a notification fans out and the user has no
 *      live WS attached.
 *
 * The page-side also listens for postMessage from the SW when the
 * user clicks a notification, so it can focus/route to the
 * relevant panel (Connect thread, coder run, etc.).
 */

const SW_URL = '/notification-sw.js';
const SW_SCOPE = '/notification-sw.js';  // Narrow scope = no fetch interception.

// ── Public ──────────────────────────────────────────────────────

/**
 * Returns the current permission + subscription state:
 *
 *   { supported, permission, subscribed, endpoint }
 *
 * Suitable for rendering a Settings toggle.
 */
export async function getPushState() {
  if (!_supported()) {
    return { supported: false, permission: 'unsupported', subscribed: false, endpoint: '' };
  }
  const permission = Notification.permission;
  let subscribed = false;
  let endpoint = '';
  try {
    const reg = await navigator.serviceWorker.getRegistration(SW_SCOPE);
    if (reg) {
      const sub = await reg.pushManager.getSubscription();
      if (sub) {
        subscribed = true;
        endpoint = sub.endpoint || '';
      }
    }
  } catch (_) { /* private mode or perm dialog dismissed — treat as unsubscribed */ }
  return { supported: true, permission, subscribed, endpoint };
}

/**
 * Run the full subscribe flow:
 *   - install/refresh the SW
 *   - request Notification.requestPermission()
 *   - pushManager.subscribe with the server's VAPID public key
 *   - POST to /api/notify/subscriptions
 *
 * Returns the resulting state object (same shape as getPushState).
 * Rejects with a coded Error on failure so the UI can surface a
 * specific reason rather than a generic "subscribe failed".
 */
export async function enablePush({ channelPattern = '*', importanceFloor = 0 } = {}) {
  if (!_supported()) {
    throw new Error('push_unsupported');
  }

  // 1. Get VAPID key first — without it, subscribe can't proceed
  //    and we shouldn't even prompt for permission.
  const vapidResp = await fetch('/api/notify/vapid-public-key', {
    credentials: 'same-origin',
  });
  if (!vapidResp.ok) {
    throw new Error(`vapid_fetch_${vapidResp.status}`);
  }
  const { public_key: publicKey } = await vapidResp.json();
  if (!publicKey) throw new Error('vapid_missing');

  // 2. Permission.
  const perm = await Notification.requestPermission();
  if (perm !== 'granted') {
    throw new Error(`permission_${perm}`);
  }

  // 3. SW registration. Narrow scope so the SW doesn't intercept
  //    fetches across the whole origin.
  const reg = await navigator.serviceWorker.register(SW_URL, {
    scope: SW_SCOPE,
  });
  // Wait for activation so pushManager.subscribe doesn't race the
  // SW lifecycle.
  if (reg.installing) {
    await new Promise((resolve) => {
      const w = reg.installing;
      w.addEventListener('statechange', () => {
        if (w.state === 'activated' || w.state === 'redundant') resolve();
      });
    });
  }

  // 4. Subscribe (or re-use an existing subscription).
  let subscription = await reg.pushManager.getSubscription();
  if (!subscription) {
    subscription = await reg.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: _urlBase64ToUint8Array(publicKey),
    });
  }

  // 5. POST to server. Subscription contains binary keys we need to
  //    base64url-encode for the wire.
  const subJson = subscription.toJSON ? subscription.toJSON() : {
    endpoint: subscription.endpoint,
    keys: {
      p256dh: _arrayBufferToB64Url(subscription.getKey('p256dh')),
      auth: _arrayBufferToB64Url(subscription.getKey('auth')),
    },
  };
  const postResp = await fetch('/api/notify/subscriptions', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      endpoint: subJson.endpoint,
      p256dh: subJson.keys.p256dh,
      auth: subJson.keys.auth,
      channel_pattern: channelPattern,
      importance_floor: importanceFloor,
    }),
  });
  if (!postResp.ok) {
    // Roll back the browser-side subscription so the user isn't in
    // an "I subscribed but server doesn't know" state.
    try { await subscription.unsubscribe(); } catch (_) {}
    throw new Error(`subscribe_post_${postResp.status}`);
  }

  return getPushState();
}

/**
 * Tear down the local subscription and notify the server. Best-effort
 * on both sides — orphan rows on the server are harmless (Web Push
 * sends will 410 and the dispatcher prunes them).
 */
export async function disablePush() {
  if (!_supported()) return { supported: false };

  const reg = await navigator.serviceWorker.getRegistration(SW_SCOPE);
  let endpoint = '';
  if (reg) {
    const sub = await reg.pushManager.getSubscription();
    if (sub) {
      endpoint = sub.endpoint;
      try { await sub.unsubscribe(); } catch (_) { /* already gone */ }
    }
  }

  if (endpoint) {
    try {
      // Look up our row server-side, then delete it. List endpoint
      // returns minimal metadata (no secrets).
      const listResp = await fetch('/api/notify/subscriptions', {
        credentials: 'same-origin',
      });
      if (listResp.ok) {
        const { subscriptions = [] } = await listResp.json();
        const match = subscriptions.find((s) => s.endpoint === endpoint);
        if (match) {
          await fetch(
            `/api/notify/subscriptions/${encodeURIComponent(match.subscription_id)}`,
            { method: 'DELETE', credentials: 'same-origin' },
          );
        }
      }
    } catch (_) { /* server cleanup is opportunistic */ }
  }

  return getPushState();
}

/**
 * Wire the SW's postMessage handler so notification clicks land back
 * in the page. The handler fires an ``augmentum:notification-click``
 * window event with the same shape thread-panel.js already listens
 * for via ``augmentum:notification-action`` — the two paths converge.
 */
export function installClickListener() {
  if (!_supported()) return;
  navigator.serviceWorker.addEventListener('message', (ev) => {
    const data = ev.data || {};
    if (data.type !== 'augmentum:notification-click') return;
    window.dispatchEvent(new CustomEvent('augmentum:notification-click', {
      detail: data,
    }));
  });
  _consumeColdStartClick();
}

/**
 * Cold-start push tap: with no app tab open, the service worker can't
 * postMessage a page that doesn't exist yet — it opens the app with a
 * ``?notify_open=…`` param instead. Consume it here (installClickListener
 * runs at boot), scrub the URL, and re-dispatch the SAME window event the
 * warm path uses so routing stays one code path. Deferred a beat so the
 * drawer / media surfaces have registered their listeners.
 */
function _consumeColdStartClick() {
  let data = null;
  try {
    const qs = new URLSearchParams(window.location.search);
    const raw = qs.get('notify_open');
    if (!raw) return;
    data = JSON.parse(raw);
    qs.delete('notify_open');
    const clean = window.location.pathname
      + (qs.toString() ? `?${qs}` : '') + window.location.hash;
    window.history.replaceState(null, '', clean);
  } catch (_) {
    return; // malformed param — scrubbed or ignored, never fatal
  }
  if (!data || typeof data !== 'object') return;
  setTimeout(() => {
    window.dispatchEvent(new CustomEvent('augmentum:notification-click', {
      detail: {
        type: 'augmentum:notification-click',
        action: '',
        notification_id: '',
        channel_id: data.channel_id || '',
        thread_id: '',
        payload: data.payload || {},
      },
    }));
  }, 1200);
}

// ── Internals ───────────────────────────────────────────────────

function _supported() {
  return (
    'serviceWorker' in navigator
    && 'PushManager' in window
    && 'Notification' in window
  );
}

/**
 * Convert a VAPID public key (base64url string from the server) to
 * the Uint8Array form pushManager.subscribe expects.
 */
function _urlBase64ToUint8Array(b64url) {
  const padding = '='.repeat((4 - b64url.length % 4) % 4);
  const base64 = (b64url + padding).replace(/-/g, '+').replace(/_/g, '/');
  const raw = atob(base64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) out[i] = raw.charCodeAt(i);
  return out;
}

function _arrayBufferToB64Url(buf) {
  if (!buf) return '';
  const bytes = new Uint8Array(buf);
  let str = '';
  for (let i = 0; i < bytes.length; i += 1) str += String.fromCharCode(bytes[i]);
  return btoa(str).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
