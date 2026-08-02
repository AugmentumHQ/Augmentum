/**
 * Static-file preview helper for coder mode.
 *
 * Sits alongside the dev-server reverse-proxy preview (which requires
 * a published port + a live HTTP server inside the container). This
 * module renders workspace files directly via /api/coder/preview-file
 * — useful for AI-generated harness.html, README.md, design.svg,
 * screenshot.png, spec.pdf, etc.
 *
 * Single source of truth: the previewable extension list lives in
 * augmentum/coder/preview_types.py. This module fetches it once from
 * /api/coder/preview-types and caches in-memory. No parallel list to
 * keep in sync.
 *
 * Public API:
 *
 *   await loadPreviewableExtensions()    — preload the registry (idempotent)
 *   isPreviewable(filenameOrPath)        — bool; false until loadPreviewableExtensions resolves
 *   buildPreviewUrl(workspaceId, path)   — string; the route URL
 *   getExtension(filenameOrPath)         — '.html' / '' helper
 */

const STATE = {
  loaded: false,
  loading: null,
  extensions: new Set(),  // {'.html', '.md', ...}
  byKind: {},             // {html: ['.html', '.htm'], markdown: [...], ...}
};

/**
 * Return the lowercase, leading-dot extension of a path/filename.
 * Returns '' when there isn't one. Mirrors preview_types.extension_for_path.
 */
export function getExtension(pathOrFilename) {
  if (!pathOrFilename || typeof pathOrFilename !== 'string') return '';
  const idx = pathOrFilename.lastIndexOf('.');
  if (idx < 0) return '';
  const tail = pathOrFilename.slice(idx + 1).toLowerCase();
  if (!tail || tail.includes('/')) return '';
  return '.' + tail;
}

/**
 * Fetch /api/coder/preview-types once and cache. Subsequent calls return
 * the cached promise so callers from different sites don't re-fetch.
 *
 * Failure modes are silent: a 401 (not logged in) / 404 / network error
 * leaves the cache empty, and isPreviewable() returns false everywhere.
 * That's safe — the worst outcome is the Preview context-menu item never
 * appears, not that broken URLs get generated.
 */
export function loadPreviewableExtensions() {
  if (STATE.loaded) return Promise.resolve();
  if (STATE.loading) return STATE.loading;
  STATE.loading = (async () => {
    try {
      const resp = await fetch('/api/coder/preview-types', { credentials: 'include' });
      if (!resp.ok) return;
      const data = await resp.json();
      const exts = Array.isArray(data?.extensions) ? data.extensions : [];
      STATE.extensions = new Set(exts.map(e => String(e).toLowerCase()));
      STATE.byKind = data?.by_kind && typeof data.by_kind === 'object' ? data.by_kind : {};
      STATE.loaded = true;
    } catch {
      // Silent failure — see docstring rationale.
    } finally {
      STATE.loading = null;
    }
  })();
  return STATE.loading;
}

/**
 * True iff the filename's extension is registered as previewable.
 *
 * Returns false until loadPreviewableExtensions() has resolved. Callers
 * should await the load before relying on this for visibility decisions;
 * if you call it without awaiting (e.g. in a synchronous render path),
 * the gate fails closed and the menu item is hidden until next render —
 * acceptable, since the registry preloads at coder-mode init.
 */
export function isPreviewable(pathOrFilename) {
  const ext = getExtension(pathOrFilename);
  if (!ext) return false;
  return STATE.extensions.has(ext);
}

/**
 * Build the route URL for a file preview. Doesn't open it — the caller
 * (file-tree context menu) hands this to the existing iframe + token
 * machinery in coder.js so origin isolation + expiry handling are
 * preserved without duplication.
 *
 * ``path`` is the absolute container path (``/workspace/foo/bar.html``).
 * The route's ``{path:path}`` capture eats the leading slash, but FastAPI
 * still routes correctly because the prefix terminates with a slash and
 * the remaining segments are captured verbatim. We encode each segment
 * (not the slashes) so a name like ``my report.pdf`` survives.
 */
export function buildPreviewUrl(workspaceId, path) {
  if (!workspaceId || !path) return '';
  // Split on '/' so we can encode each segment individually — encodeURI
  // alone wouldn't escape '?' or '#' that a malicious filename might
  // include, and encodeURIComponent would eat the path separators.
  const safeSegments = path.split('/').map(seg => encodeURIComponent(seg));
  // Re-join. path starts with '/' so safeSegments[0] is '' — preserves leading /.
  const safePath = safeSegments.join('/');
  return `/api/coder/preview-file/${encodeURIComponent(workspaceId)}${safePath}`;
}

/**
 * Returns the file's preview "kind" (html, markdown, image, pdf, json,
 * text, audio, video, svg, code) or '' when not previewable. Useful for
 * the UI to pick an icon or heading without a parallel mapping.
 */
export function previewKind(pathOrFilename) {
  const ext = getExtension(pathOrFilename);
  if (!ext) return '';
  for (const [kind, exts] of Object.entries(STATE.byKind)) {
    if (Array.isArray(exts) && exts.includes(ext)) return kind;
  }
  return '';
}
