/* ui/lib/silero-vad/loader.js
 *
 * Loads `@ricky0123/vad-web` (Silero VAD running in the browser via
 * ONNX Runtime Web) from same-origin vendored assets. Same-origin is
 * mandatory: the bundle's internal ORT uses `import('./<file>.mjs')`
 * to load its WASM provider, and that dynamic import resolves
 * against the script's own URL — which fails with "base URL is
 * about:blank" when the script was loaded as a CORS-cross-origin
 * resource (jsdelivr/unpkg).
 *
 * The vendored assets live in ui/lib/silero-vad/ — see README.md for
 * version pins and how to upgrade.
 *
 * Public API:
 *   loadSileroVAD()  → Promise<{ vad: VadModule }>
 *   smokeTest()      → Promise<boolean>  (true if a silent 1s buffer
 *                                          produces zero speech segments)
 */

// Absolute paths so MicVAD's internal default fetchers resolve correctly
// regardless of where the calling script is hosted.
const ASSET_BASE = '/ui/lib/silero-vad/';

const ASSETS = Object.freeze({
  bundle:  `${ASSET_BASE}bundle.min.js`,
  worklet: `${ASSET_BASE}vad.worklet.bundle.min.js`,
  model:   `${ASSET_BASE}silero_vad_legacy.onnx`,
  // ORT's wasm loader expects its .mjs + .wasm side-by-side at this
  // base. `onnxWASMBasePath` is the option vad-web exposes to override
  // ORT's lookup root.
  ortWasmBase: ASSET_BASE,
});

const VERSIONS = Object.freeze({
  vadWeb: '0.0.30',
  ort: '1.20.0',
  sileroModel: 'legacy',
});

let _modulePromise = null;

function injectScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-silero-src="${src}"]`);
    if (existing) {
      if (existing.dataset.sileroLoaded === '1') {
        resolve();
      } else {
        existing.addEventListener('load', () => resolve());
        existing.addEventListener('error', () => reject(new Error(`script load failed: ${src}`)));
      }
      return;
    }
    const tag = document.createElement('script');
    tag.src = src;
    tag.dataset.sileroSrc = src;
    tag.addEventListener('load', () => {
      tag.dataset.sileroLoaded = '1';
      resolve();
    });
    tag.addEventListener('error', () => reject(new Error(`script load failed: ${src}`)));
    document.head.appendChild(tag);
  });
}

/**
 * Idempotent loader. First call injects the vad-web bundle from
 * same-origin; subsequent calls await the same promise.
 *
 * vad-web bundles its own ORT — we don't pre-inject onnxruntime-web.
 *
 * On failure the cached promise is cleared so the next call retries.
 */
export function loadSileroVAD() {
  if (_modulePromise) return _modulePromise;
  _modulePromise = (async () => {
    await injectScript(ASSETS.bundle);
    if (typeof window.vad === 'undefined') {
      throw new Error('@ricky0123/vad-web loaded but window.vad is undefined');
    }
    return { vad: window.vad };
  })().catch((err) => {
    _modulePromise = null;
    throw err;
  });
  return _modulePromise;
}

/**
 * Verify the library loads + can run the model over 1 second of
 * silence. Resolves true on success; throws on failure so callers can
 * surface the underlying error.
 */
export async function smokeTest() {
  const { vad } = await loadSileroVAD();
  const silence = new Float32Array(16000);
  const session = await vad.NonRealTimeVAD.new({
    baseAssetPath: ASSET_BASE,
    onnxWASMBasePath: ASSETS.ortWasmBase,
    model: VERSIONS.sileroModel,
  });
  const segments = [];
  for await (const seg of session.run(silence, 16000)) {
    segments.push(seg);
  }
  session.destroy?.();
  return segments.length === 0;
}

export const META = Object.freeze({
  versions: VERSIONS,
  assets: ASSETS,
  assetBase: ASSET_BASE,
});
