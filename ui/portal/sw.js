/* Minimal service worker — exists so the portal is installable (add to
 * home screen) and the shell loads instantly on return. Network-first so
 * we never serve a stale build; the cache is just a fast/offline fallback.
 */
const SHELL = 'portal-shell-v2';
const ASSETS = ['./', 'index.html', 'portal.js', 'portal.css', 'manifest.json', 'icon.svg',
  'env.js', '../lib/noble/index.js'];
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== SHELL).map((k) => caches.delete(k)))));
});
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.pathname.includes('/api/')) return; // never cache API
  e.respondWith(
    fetch(e.request).then((r) => {
      const copy = r.clone();
      caches.open(SHELL).then((c) => c.put(e.request, copy)).catch(() => {});
      return r;
    }).catch(() => caches.match(e.request).then((m) => m || caches.match('index.html')))
  );
});
