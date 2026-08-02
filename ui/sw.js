// Service worker tombstone.
//
// Keep this file so already-registered browsers can update to it, then
// unregister cleanly and clear Augmentum UI caches. The live app no longer
// registers a service worker while local development is changing quickly.
self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (event) => {
  // Clean up caches + unregister, then let the user's NEXT natural navigation
  // pick up the SW-less state. The previous version called
  // ``client.navigate(client.url)`` here, which forced an immediate reload of
  // every open tab — that aborted any in-flight ``fetch()`` on those tabs,
  // including the user's chat POST. On phones especially the activate timing
  // (~1s after page load) consistently cancelled chat streams. Don't reload.
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((key) => key.startsWith('augmentum-'))
        .map((key) => caches.delete(key)),
    );
    await self.registration.unregister();
  })());
});
