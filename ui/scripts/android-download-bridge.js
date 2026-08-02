// Android download bridge — makes blob:/data: downloads work inside the
// Augmentum Android app's WebView.
//
// In a normal browser, clicking an `<a download>` whose href is a blob: or
// data: URL is handled by the browser's own download machinery. Android
// WebView instead hands the URL to the app's DownloadListener, whose
// DownloadManager only accepts http(s) — so every client-generated export
// (chat JSON, artifact/image saves, flow exports, project zips, …) silently
// did nothing in the app. This shim intercepts those downloads at the
// prototype level (most export paths click a DETACHED anchor, which no
// document-level listener can observe) and streams the bytes to the native
// side in base64 chunks; native writes them into the system Downloads
// collection and toasts the result.
//
// Deliberately NOT intercepted: http(s) hrefs with a download attribute —
// the WebView's DownloadListener already handles those correctly (with the
// session cookie attached), including Content-Disposition names.
//
// Inert everywhere else: feature-detects window.AugmentumAndroid AND the
// saveFile methods, so browsers and older APKs keep their default behavior.
(function () {
  'use strict';

  const bridge = window.AugmentumAndroid;
  if (
    !bridge ||
    typeof bridge.beginSaveFile !== 'function' ||
    typeof bridge.appendSaveFile !== 'function' ||
    typeof bridge.endSaveFile !== 'function'
  ) {
    return;
  }

  // ── Mirror the document's blob-URL store ─────────────────────────────
  // Several export paths revoke the object URL SYNCHRONOUSLY after click()
  // (`a.click(); URL.revokeObjectURL(url)`), so an async fetch of the URL
  // after interception would find it already dead. Instead, keep url → Blob
  // for as long as the URL is registered. This costs no extra memory: the
  // browser itself pins the blob's bytes until revokeObjectURL — we only
  // hold a reference to what's already alive, and drop it exactly when the
  // browser does.
  const blobsByUrl = new Map();
  const origCreateObjectURL = URL.createObjectURL.bind(URL);
  const origRevokeObjectURL = URL.revokeObjectURL.bind(URL);
  URL.createObjectURL = function (obj) {
    const url = origCreateObjectURL(obj);
    try {
      if (obj instanceof Blob) blobsByUrl.set(url, obj);
    } catch (_) { /* MediaSource etc. — not downloadable, ignore */ }
    return url;
  };
  URL.revokeObjectURL = function (url) {
    try { blobsByUrl.delete(url); } catch (_) { /* ignore */ }
    return origRevokeObjectURL(url);
  };

  /** href (blob:/data:) → Blob. Map hit is the reliable path for blob: */
  function resolveBlob(href) {
    const mapped = blobsByUrl.get(href);
    if (mapped) return Promise.resolve(mapped);
    // data: URLs, or a blob: URL minted outside this document (worker /
    // other realm) — fetch while it's (hopefully) still registered.
    return fetch(href).then((r) => r.blob());
  }

  // Binary chunk size per bridge call. Kept well under the native-side
  // sanity cap; a multiple of 3 so each chunk base64-encodes cleanly.
  const CHUNK_BYTES = 768 * 1024;

  /** Base64-encode one blob slice via FileReader (no giant btoa strings). */
  function chunkToBase64(slice) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => {
        const s = String(fr.result || '');
        resolve(s.slice(s.indexOf(',') + 1));
      };
      fr.onerror = () => reject(fr.error || new Error('blob read failed'));
      fr.readAsDataURL(slice);
    });
  }

  /** Stream a Blob to the native Downloads writer. Never truncates: every
   *  byte is sent across sequential chunks; any failure aborts the whole
   *  save (native deletes the partial file) rather than keeping a short one. */
  async function saveBlob(blob, name) {
    const id = bridge.beginSaveFile(name || '', blob.type || '', String(blob.size));
    if (!id) throw new Error('native save rejected');
    try {
      for (let off = 0; off < blob.size; off += CHUNK_BYTES) {
        const b64 = await chunkToBase64(blob.slice(off, off + CHUNK_BYTES));
        if (!bridge.appendSaveFile(id, b64)) throw new Error('native write failed');
      }
      if (!bridge.endSaveFile(id)) throw new Error('native finalize failed');
    } catch (err) {
      try {
        if (typeof bridge.abortSaveFile === 'function') bridge.abortSaveFile(id);
      } catch (_) { /* ignore */ }
      throw err;
    }
  }

  /**
   * If [anchor] is a download the WebView can't handle natively, start the
   * bridged save and return true (caller suppresses the default action).
   */
  function interceptDownload(anchor) {
    if (!(anchor instanceof HTMLAnchorElement)) return false;
    if (!anchor.hasAttribute('download')) return false;
    const href = anchor.href || '';
    if (!/^(blob:|data:)/i.test(href)) return false;
    const name = anchor.getAttribute('download') || '';
    resolveBlob(href)
      .then((blob) => saveBlob(blob, name))
      .catch((err) => {
        // Native side toasts its own failures exactly once; this covers
        // JS-side ones (blob already revoked, read error).
        console.warn('[android-download] save failed:', err);
      });
    return true;
  }

  // Programmatic clicks — the dominant export pattern is a detached
  // `document.createElement('a')` + `.click()`, which never dispatches
  // through the document, so the prototype is the only choke point.
  const origClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () {
    if (interceptDownload(this)) return;
    return origClick.apply(this, arguments);
  };

  // Real user taps on rendered <a download> links (attached anchors).
  // Mutually exclusive with the prototype patch: page-code .click() calls
  // return above and never dispatch an event, so nothing saves twice.
  document.addEventListener(
    'click',
    (e) => {
      const el = e.target;
      const a = el && el.closest ? el.closest('a[download]') : null;
      if (a && interceptDownload(a)) e.preventDefault();
    },
    true,
  );
})();
