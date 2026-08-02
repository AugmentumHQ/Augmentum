/* notification-sw.js — Service worker for Augmentum Web Push.
 *
 * Lives at /notification-sw.js (NOT /sw.js — that path is a
 * tombstone that auto-unregisters; see ui/sw.js header). This SW
 * is scoped narrowly via the registration call so it doesn't
 * intercept fetches; it exists only to handle push + notification
 * click events.
 *
 * Three handlers:
 *
 *   - install   → skipWaiting so a freshly-published SW takes
 *                  over without forcing a tab reload.
 *   - activate  → claim existing clients so already-open tabs can
 *                  subscribe without a reload.
 *   - push      → render the notification with the payload the
 *                  server sent; click routes back into the open
 *                  app via clients.openWindow / focus.
 *
 * Payload shape (set by augmentum/notifications/hub.py::_dispatch_webpush):
 *   { notification_id, channel_id, title, body, icon, importance,
 *     thread_id, actions: [{action, title}], payload: {...} }
 *
 * The SW is intentionally tiny — anything that needs app state
 * (like opening a specific thread on click) happens via the page
 * once focus / openWindow lands.
 */

self.addEventListener('install', (event) => {
  // Take over from any older waiting SW so the user doesn't need to
  // reload twice to get push working.
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('push', (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch (_) {
    // The push server might send a non-JSON body during testing.
    // Fall back to a generic surface so the user still sees
    // SOMETHING.
    data = { title: 'Augmentum', body: event.data ? event.data.text() : '' };
  }

  const title = String(data.title || 'Augmentum');
  const opts = {
    body: String(data.body || ''),
    icon: data.icon ? String(data.icon) : '/ui/favicon.ico',
    badge: '/ui/favicon.ico',
    // Dedupe across rapid sends: a second push with the same tag
    // replaces (rather than stacks) the prior notification. Connect
    // uses dedupe_key per thread so chatty senders don't pile up
    // banners; the tag mirrors that.
    tag: data.thread_id ? `connect:${data.thread_id}` : (data.notification_id || ''),
    renotify: !!data.thread_id,
    data: {
      notification_id: data.notification_id || '',
      channel_id: data.channel_id || '',
      thread_id: data.thread_id || '',
      payload: data.payload || {},
    },
    // Up to 2 inline action buttons (browser-enforced cap).
    actions: Array.isArray(data.actions) ? data.actions.slice(0, 2) : [],
  };

  event.waitUntil(self.registration.showNotification(title, opts));
});

self.addEventListener('notificationclick', (event) => {
  const n = event.notification;
  const data = n.data || {};
  n.close();

  // Intent: focus an open Augmentum tab; if none, open a fresh one.
  // The page-side listener (push-subscribe.js) reads the message
  // payload and routes to the appropriate panel (thread, call,
  // coder run, etc.).
  event.waitUntil((async () => {
    const all = await self.clients.matchAll({
      type: 'window', includeUncontrolled: true,
    });
    const payload = {
      type: 'augmentum:notification-click',
      action: event.action || '',
      notification_id: data.notification_id || '',
      channel_id: data.channel_id || '',
      thread_id: data.thread_id || '',
      payload: data.payload || {},
    };

    for (const client of all) {
      try {
        client.postMessage(payload);
        if ('focus' in client) {
          await client.focus();
          return;
        }
      } catch (_) { /* try the next one */ }
    }
    // No tabs open — fall back to opening the root with a routing
    // param the page-side handler consumes on load. Task fires carry
    // their deep-link target (payload.open_url) so a cold-start tap
    // still lands ON the item, not just on the app.
    if (self.clients.openWindow) {
      let url = '/';
      if (data.thread_id) {
        url = `/?connect_thread=${encodeURIComponent(data.thread_id)}`;
      } else if (
        (data.payload && data.payload.open_url)
        || data.channel_id === 'companion.tasks'
      ) {
        const p = data.payload || {};
        url = `/?notify_open=${encodeURIComponent(JSON.stringify({
          channel_id: data.channel_id || '',
          payload: {
            open_url: p.open_url || '',
            open_kind: p.open_kind || '',
            open_title: p.open_title || '',
          },
        }))}`;
      }
      await self.clients.openWindow(url);
    }
  })());
});
