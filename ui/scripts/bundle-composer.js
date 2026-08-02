/* ==========================================================================
   Bundle composer + save-state bridge.

   Two cooperating pieces used by game-surface.js (Library "Play" path) and
   cast-app.js (TV cast surface) so a local-mode bundle renders identically
   in both surfaces:

     composeBundle(html, files, entryPath, initialSave)
       Pure function. Takes the entry HTML + sibling files, returns a self-
       contained document string ready for iframe.srcdoc:
         - Inlines <script src>, <link rel=stylesheet href>, <img src> when
           the target is a bundled file
         - Injects a head shim that patches fetch / XMLHttpRequest so
           relative reads at runtime resolve against the bundle
         - Replaces localStorage / sessionStorage with a shim pre-seeded
           with the user's persisted save state; subsequent writes
           postMessage back to the parent

     installSaveBridge({iframe, artifactId, onStatus})
       Wires the parent-side half of the save bridge. Pre-fetches
       /api/games/saves/{id}, listens for storage-init / storage-set /
       storage-remove / storage-clear messages from the iframe, debounces
       PUTs back to /api/games/saves/{id}. Returns { uninstall, flush,
       getInitialSave } so the caller can tear down + flush on close.
   ========================================================================== */

import { agentBridgeShim } from './agent/agent-bridge.js';


/* ── Bundle composer ─────────────────────────────────────────────── */

const _IMG_EXTENSIONS = {
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  gif: 'image/gif',
  webp: 'image/webp',
  svg: 'image/svg+xml',
};

export function composeBundle(html, files, entryPath, initialSave, agentBridge) {
  const byPath = new Map();
  for (const f of files) {
    if (!f.path || f.path === entryPath || f.path.startsWith('.')) continue;
    byPath.set(f.path, f);
  }
  const hasSiblings = byPath.size > 0;

  if (hasSiblings) {
    html = html.replace(/<script\b([^>]*)\bsrc\s*=\s*["']([^"']+)["']([^>]*)><\/script>/gi,
      (match, pre, src, post) => {
        const f = byPath.get(src);
        if (!f || f.encoding !== 'text') return match;
        return `<script${pre}${post}>\n/* inlined from ${src} */\n${f.content.replace(/<\/script>/gi, '<\\/script>')}\n</script>`;
      });

    html = html.replace(
      /<link\b[^>]*\brel\s*=\s*["']stylesheet["'][^>]*\bhref\s*=\s*["']([^"']+)["'][^>]*\/?>/gi,
      (match, href) => {
        const f = byPath.get(href);
        if (!f || f.encoding !== 'text') return match;
        return `<style>/* inlined from ${href} */\n${f.content}\n</style>`;
      });

    html = html.replace(/<img\b([^>]*)\bsrc\s*=\s*["']([^"']+)["']([^>]*)>/gi,
      (match, pre, src, post) => {
        const f = byPath.get(src);
        if (!f) return match;
        const ext = (src.split('.').pop() || '').toLowerCase();
        const mime = _IMG_EXTENSIONS[ext] || 'application/octet-stream';
        const b64 = f.encoding === 'base64' ? f.content : btoa(unescape(encodeURIComponent(f.content)));
        return `<img${pre} src="data:${mime};base64,${b64}"${post}>`;
      });
  }

  const bundleLiteral = JSON.stringify(Object.fromEntries(
    Array.from(byPath.entries()).map(([path, f]) => [path, { e: f.encoding, c: f.content }])
  ));
  const initialSaveLiteral = JSON.stringify(initialSave || {});
  const fetchXhrShim = hasSiblings ? `
  const B = ${bundleLiteral};
  function bytes(p) {
    const f = B[p]; if (!f) return null;
    if (f.e === 'base64') {
      const s = atob(f.c); const u = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
      return u;
    }
    return new TextEncoder().encode(f.c);
  }
  function resolve(url) {
    if (!url || typeof url !== 'string') return null;
    let k = url.replace(/^\\.\\//, '').replace(/^\\//, '');
    if (B[k]) return k;
    const tail = k.split('/').pop();
    if (tail && B[tail]) return tail;
    return null;
  }
  const _fetch = window.fetch;
  window.fetch = function(input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    const k = resolve(url);
    if (k) {
      const u = bytes(k);
      return Promise.resolve(new Response(u, { status: 200, headers: { 'content-type': 'application/octet-stream' } }));
    }
    return _fetch.call(this, input, init);
  };
  const _X = window.XMLHttpRequest;
  window.XMLHttpRequest = function() {
    const x = new _X();
    const _open = x.open.bind(x);
    let _k = null;
    x.open = function(method, url) {
      _k = (String(method || '').toUpperCase() === 'GET') ? resolve(url) : null;
      if (_k) { return; }
      return _open.apply(x, arguments);
    };
    const _send = x.send.bind(x);
    x.send = function() {
      if (!_k) return _send.apply(x, arguments);
      queueMicrotask(() => {
        const u = bytes(_k);
        try { Object.defineProperty(x, 'readyState', { value: 4, configurable: true }); } catch {}
        try { Object.defineProperty(x, 'status', { value: 200, configurable: true }); } catch {}
        try { Object.defineProperty(x, 'response', { value: u.buffer, configurable: true }); } catch {}
        try { Object.defineProperty(x, 'responseText', { value: new TextDecoder().decode(u), configurable: true }); } catch {}
        try { Object.defineProperty(x, 'responseType', { value: 'arraybuffer', writable: true, configurable: true }); } catch {}
        x.dispatchEvent(new Event('readystatechange'));
        x.dispatchEvent(new Event('load'));
        x.dispatchEvent(new Event('loadend'));
      });
    };
    return x;
  };
` : '';

  const shim = `<script>(() => {${fetchXhrShim}
  const _store = ${initialSaveLiteral};
  const _session = {};
  function makeLS(dict, syncToParent) {
    return {
      getItem(k) { return Object.prototype.hasOwnProperty.call(dict, k) ? dict[k] : null; },
      setItem(k, v) {
        const val = String(v);
        dict[k] = val;
        if (syncToParent) parent.postMessage({ type: 'storage-set', key: String(k), value: val }, '*');
      },
      removeItem(k) {
        delete dict[k];
        if (syncToParent) parent.postMessage({ type: 'storage-remove', key: String(k) }, '*');
      },
      clear() {
        for (const k of Object.keys(dict)) delete dict[k];
        if (syncToParent) parent.postMessage({ type: 'storage-clear' }, '*');
      },
      get length() { return Object.keys(dict).length; },
      key(i) { return Object.keys(dict)[i] || null; },
    };
  }
  const ls = makeLS(_store, true);
  const ss = makeLS(_session, false);

  try {
    Object.defineProperty(window, 'localStorage', { value: ls, configurable: true, writable: true });
    Object.defineProperty(window, 'sessionStorage', { value: ss, configurable: true, writable: true });
  } catch {
    try { globalThis.localStorage = ls; } catch {}
    try { globalThis.sessionStorage = ss; } catch {}
  }

  parent.postMessage({ type: 'storage-init' }, '*');
})();</script>`;

  if (/<head[^>]*>/i.test(html)) {
    html = html.replace(/<head[^>]*>/i, (m) => m + '\n' + shim);
  } else if (/<html[^>]*>/i.test(html)) {
    html = html.replace(/<html[^>]*>/i, (m) => m + '\n<head>' + shim + '</head>');
  } else {
    html = shim + html;
  }

  // Agent-play mode: append the agent-bridge shim so the game_agent can
  // observe + drive this game over the session bridge WS. Omitted for the
  // normal human-play path (agentBridge undefined) — that path is unchanged.
  // Appended at the END so it runs after the game has defined
  // window.AUGMENTUM_GAME and created its <canvas>.
  if (agentBridge && agentBridge.wsUrl && agentBridge.sessionId) {
    const bridge = agentBridgeShim(agentBridge);
    if (/<\/body>/i.test(html)) {
      html = html.replace(/<\/body>/i, bridge + '\n</body>');
    } else {
      html = html + '\n' + bridge;
    }
  }
  return html;
}


/* ── Save-state bridge ───────────────────────────────────────────── */

const SAVE_DEBOUNCE_MS = 1500;

async function _loadSaveState(artifactId) {
  try {
    const resp = await fetch(`/api/games/saves/${encodeURIComponent(artifactId)}`);
    if (!resp.ok) return {};
    const body = await resp.json().catch(() => ({}));
    return body?.data && typeof body.data === 'object' ? body.data : {};
  } catch {
    return {};
  }
}

/**
 * Install the parent-side save bridge for a local-mode bundle.
 *
 * @param {object} opts
 * @param {HTMLIFrameElement} opts.iframe   The iframe the bundle mounts in
 * @param {string} opts.artifactId          Artifact id for /api/games/saves
 * @param {(status: string) => void} [opts.onStatus]  Status callback —
 *   'idle' | 'syncing' | 'saved' | 'error'. Survey only; presentation is the
 *   caller's responsibility (Library shows a pill, cast surface ignores it).
 *
 * @returns {{ uninstall(): void, flush(): Promise<void>, getInitialSave(): Promise<object> }}
 */
export function installSaveBridge({ iframe, artifactId, onStatus }) {
  const state = {
    active: true,
    data: null,
    inflight: false,
    dirty: false,
    flushTimer: null,
    onMessage: null,
    initialPromise: null,
  };
  const setStatus = (s) => { try { onStatus?.(s); } catch { /* ignore */ } };

  state.initialPromise = _loadSaveState(artifactId).then((data) => {
    state.data = data;
    return data;
  });

  async function flush() {
    if (!state.active) return;
    if (state.inflight) {
      state.dirty = true;
      return;
    }
    state.inflight = true;
    state.dirty = false;
    setStatus('syncing');
    try {
      const resp = await fetch(`/api/games/saves/${encodeURIComponent(artifactId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: state.data || {} }),
      });
      setStatus(resp.ok ? 'saved' : 'error');
    } catch {
      setStatus('error');
    } finally {
      state.inflight = false;
      if (state.dirty) {
        clearTimeout(state.flushTimer);
        state.flushTimer = setTimeout(flush, SAVE_DEBOUNCE_MS);
      }
    }
  }

  function scheduleFlush() {
    clearTimeout(state.flushTimer);
    state.flushTimer = setTimeout(flush, SAVE_DEBOUNCE_MS);
  }

  state.onMessage = (e) => {
    if (!e.data?.type || !state.active) return;
    if (iframe && e.source !== iframe.contentWindow) return;
    switch (e.data.type) {
      case 'storage-init': {
        try {
          e.source?.postMessage({ type: 'storage-init-response', data: state.data || {} }, '*');
        } catch { /* ignore */ }
        break;
      }
      case 'storage-set': {
        if (!state.data) state.data = {};
        state.data[e.data.key] = String(e.data.value);
        scheduleFlush();
        break;
      }
      case 'storage-remove': {
        if (state.data) delete state.data[e.data.key];
        scheduleFlush();
        break;
      }
      case 'storage-clear': {
        state.data = {};
        scheduleFlush();
        break;
      }
    }
  };
  window.addEventListener('message', state.onMessage);
  setStatus('idle');

  return {
    uninstall() {
      if (!state.active) return;
      if (state.onMessage) {
        window.removeEventListener('message', state.onMessage);
        state.onMessage = null;
      }
      if (state.dirty || state.flushTimer) {
        clearTimeout(state.flushTimer);
        state.flushTimer = null;
        flush().catch(() => {});
      }
      state.active = false;
      state.data = null;
    },
    flush,
    getInitialSave() { return state.initialPromise; },
  };
}
