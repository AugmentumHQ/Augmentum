/* ==========================================================================
   Studio Image-tool API client
   --------------------------------------------------------------------------
   Thin wrapper around /api/studio/{id}/{search-images|generate-image|
   staging/...} so the Image tool (and any future caller) talks to one
   surface. Keeps fetch-shape consistent — JSON body, parsed JSON response,
   HTTP error → thrown.
   ========================================================================== */

export function createStudioImageApi({ artifactId }) {
  if (!artifactId) throw new Error('createStudioImageApi requires artifactId');
  const base = `/api/studio/${encodeURIComponent(artifactId)}`;

  async function _post(path, body) {
    const resp = await fetch(`${base}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
      credentials: 'same-origin',
    });
    return _parse(resp);
  }

  async function _del(path) {
    const resp = await fetch(`${base}${path}`, {
      method: 'DELETE',
      credentials: 'same-origin',
    });
    return _parse(resp);
  }

  async function _parse(resp) {
    let data = {};
    try { data = await resp.json(); }
    catch { data = {}; }
    if (!resp.ok) {
      const msg = data?.error || `${resp.status} ${resp.statusText}`;
      throw new Error(msg);
    }
    return data;
  }

  return {
    searchImages: (params) => _post('/search-images', params),
    generateImage: (params) => _post('/generate-image', params),
    commitStaged: (genId) => _post(`/staging/${encodeURIComponent(genId)}/commit`),
    discardStaged: (genId) => _del(`/staging/${encodeURIComponent(genId)}`),

    // Library listing reuses the existing /api/artifacts endpoint and
    // filters client-side to image-type artifacts. The Image tool's
    // Library tab calls this so the picker stays in one module.
    async listLibrary() {
      try {
        const resp = await fetch('/api/artifacts', { credentials: 'same-origin' });
        if (!resp.ok) return [];
        const data = await resp.json();
        const list = Array.isArray(data) ? data : (data.artifacts || data.items || []);
        return list.filter((a) => {
          const fmt = (a.format || '').toLowerCase();
          const mime = a.mime_type || '';
          return fmt === 'png' || fmt === 'jpg' || fmt === 'jpeg' || fmt === 'webp' || /^image\//.test(mime);
        });
      } catch (err) {
        console.warn('listLibrary failed', err);
        return [];
      }
    },
  };
}
