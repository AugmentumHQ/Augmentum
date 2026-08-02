/**
 * Math renderer — wraps KaTeX's auto-render extension.
 *
 * Policy: KaTeX is loaded lazily from `/ui/lib/katex/` only when a caller
 * actually detects math in content. The vendored files may not be
 * present yet — we HEAD-probe the CSS first so a missing optional asset
 * fails silently (no noisy "Refused to apply style ... MIME type" warning
 * from the browser when FastAPI's default 404 returns JSON).
 *
 * To enable math rendering, vendor these files into ui/lib/katex/:
 *   - katex.min.js
 *   - katex.min.css
 *   - contrib/auto-render.min.js
 * (Matching versions — KaTeX won't auto-render without the contrib script.)
 */

// Quick regex test — cheaper than KaTeX's full DOM walk. Only trigger
// the lazy-load when a document actually has math delimiters.
const _MATH_PATTERNS = [
  /\\\(.*?\\\)/s,           // \( ... \)
  /\\\[.*?\\\]/s,           // \[ ... \]
  /\$\$[\s\S]+?\$\$/,       // $$ ... $$
  /<math[\s>]/i,            // inline MathML
];

export function containsMath(text) {
  if (!text) return false;
  for (const re of _MATH_PATTERNS) if (re.test(text)) return true;
  return false;
}

let _loadState = 'idle'; // idle | loading | ready | failed
let _loadPromise = null;

function _injectStylesheet(href) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`link[href="${href}"]`);
    if (existing) {
      if (existing.sheet) resolve();
      else existing.addEventListener('load', () => resolve(), { once: true });
      return;
    }
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    link.addEventListener('load', () => resolve(), { once: true });
    link.addEventListener('error', () => reject(new Error(`failed to load ${href}`)), { once: true });
    document.head.appendChild(link);
  });
}

function _injectScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === '1') resolve();
      else existing.addEventListener('load', () => resolve(), { once: true });
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.addEventListener('load', () => { script.dataset.loaded = '1'; resolve(); }, { once: true });
    script.addEventListener('error', () => reject(new Error(`failed to load ${src}`)), { once: true });
    document.head.appendChild(script);
  });
}

async function _loadKatex() {
  if (_loadState === 'ready') return true;
  if (_loadState === 'failed') return false;
  if (_loadState === 'loading') return _loadPromise;

  _loadState = 'loading';
  _loadPromise = (async () => {
    try {
      // HEAD-probe first — if the optional vendor bundle isn't present,
      // the request returns FastAPI's default JSON 404 and the browser
      // logs a loud "Refused to apply style" warning if we let a <link>
      // see it. Bail quietly instead.
      const probe = await fetch('/ui/lib/katex/katex.min.css', { method: 'HEAD' });
      const ct = probe.headers.get('content-type') || '';
      if (!probe.ok || !ct.includes('css')) {
        throw new Error(`KaTeX not vendored (probe ${probe.status} ${ct})`);
      }
      // CSS first so glyphs render correctly once the JS finishes.
      await _injectStylesheet('/ui/lib/katex/katex.min.css');
      await _injectScript('/ui/lib/katex/katex.min.js');
      await _injectScript('/ui/lib/katex/contrib/auto-render.min.js');
      if (typeof window.renderMathInElement !== 'function') {
        throw new Error('KaTeX auto-render not exposed on window');
      }
      _loadState = 'ready';
      return true;
    } catch (err) {
      // Vendored files missing or corrupt — never retry this session.
      // One console note tells the developer what to do; end-users see
      // unrendered LaTeX source, which is still readable.
      console.debug('[augmentum] KaTeX not available — math will render as source.', err.message);
      _loadState = 'failed';
      return false;
    }
  })();

  return _loadPromise;
}

/**
 * Render math inside `container`. Fast-paths: no container, no math
 * detected, KaTeX unavailable. Safe to call indiscriminately — the
 * containsMath check guards against wasted work.
 */
export async function renderMathIn(container) {
  if (!container) return;
  const text = container.textContent || '';
  if (!containsMath(text)) return;

  const ready = await _loadKatex();
  if (!ready) return;

  try {
    window.renderMathInElement(container, {
      // Delimiters sorted longest-first so $$ doesn't get split by $.
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
      ],
      // Don't touch code blocks — hljs already owns them and math inside
      // a code sample is almost always meant as source, not rendered.
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
      // KaTeX throws loudly on malformed input by default; swallow so
      // one bad equation doesn't prevent the rest from rendering.
      throwOnError: false,
      errorColor: 'var(--error, #ef4444)',
    });
  } catch (err) {
    console.warn('[augmentum] KaTeX render error:', err);
  }
}
