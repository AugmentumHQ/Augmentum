/**
 * /ui/play-web/ — standalone URL-addressable player for js13k and
 * marketplace (curated) web games.
 *
 * Query params:
 *   embed_url  (required)  the game URL — must be http(s); page CSP
 *                          frame-src controls whether the browser
 *                          actually permits the load. Self-contained
 *                          URL means cast senders pass everything in
 *                          one shot, no metadata lookup at the receiver.
 *   title      (optional)  display name shown in the chrome bar
 *   kiosk      (optional)  '1' to hide the chrome bar — used by
 *                          cast-receiver embeds where the receiver
 *                          shell owns close affordances
 *
 * Security:
 *   The iframe is sandboxed (allow-scripts allow-forms allow-popups
 *   allow-pointer-lock allow-same-origin) — same flags as legacy
 *   game-surface. CSP frame-src is the actual origin allowlist; this
 *   page just refuses non-http(s) inputs so an attacker can't inject
 *   javascript: or data: URLs via a crafted cast message.
 */

const _fallback = document.getElementById('play-web-fallback');
const _fallbackDetail = document.getElementById('play-web-fallback-detail');

function _setFallback(title, detail) {
  if (!_fallback) return;
  const t = _fallback.querySelector('.play-web-fallback-title');
  if (t) t.textContent = title;
  if (_fallbackDetail) _fallbackDetail.textContent = detail || '';
}

function _clearFallback() {
  if (_fallback && _fallback.parentNode) _fallback.parentNode.removeChild(_fallback);
}

function _validateEmbedUrl(raw) {
  // Allow http/https only. Block javascript:, data:, file:, blob:,
  // about: — none of these should reach an iframe from a cast message.
  if (!raw) return null;
  let url;
  try {
    url = new URL(raw);
  } catch {
    return null;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
  return url.toString();
}

function _emitSurfaceReady(payload) {
  // `type:` (not `kind:`) — matches the receiver's iframe-message
  // forwarder, which converts these to generic surface_event WS frames
  // that land in cast_event_store. The controller phone can poll for
  // them to confirm the iframe loaded.
  try {
    window.parent?.postMessage({
      type: 'augmentum.surface_ready',
      surface: 'play.web',
      ...payload,
    }, '*');
  } catch (_) { /* not embedded — ignore */ }
}

function _bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const embedUrl = _validateEmbedUrl(params.get('embed_url'));
  const title = params.get('title') || 'Web game';
  const kiosk = params.get('kiosk') === '1';

  if (!embedUrl) {
    _setFallback('Invalid embed_url',
      'embed_url must be a valid http(s) URL. Got: '
      + (params.get('embed_url') || '(empty)'));
    return;
  }

  const overlay = document.createElement('div');
  overlay.className = 'play-web-overlay';
  if (kiosk) overlay.classList.add('is-kiosk');

  // Chrome bar — shown unless kiosk. Just a title + close, mirroring
  // emulator-stage's minimal-chrome register.
  const bar = document.createElement('div');
  bar.className = 'play-web-bar';
  const titleEl = document.createElement('span');
  titleEl.className = 'play-web-title';
  titleEl.textContent = title;
  const spacer = document.createElement('span');
  spacer.className = 'play-web-spacer';
  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.className = 'play-web-close';
  closeBtn.textContent = '✕';
  closeBtn.title = 'Close (Esc)';
  closeBtn.addEventListener('click', () => {
    try { window.parent?.postMessage({
      type: 'augmentum.surface_closed',
      surface: 'play.web',
    }, '*'); } catch (_) {}
    if (window.parent === window) window.close();
  });
  bar.appendChild(titleEl);
  bar.appendChild(spacer);
  bar.appendChild(closeBtn);
  overlay.appendChild(bar);

  const iframe = document.createElement('iframe');
  iframe.className = 'play-web-iframe';
  iframe.setAttribute(
    'sandbox',
    'allow-scripts allow-forms allow-modals allow-popups allow-pointer-lock allow-same-origin',
  );
  iframe.setAttribute('allow', 'fullscreen; gamepad; autoplay');
  iframe.referrerPolicy = 'no-referrer';
  iframe.title = title;
  iframe.src = embedUrl;
  iframe.addEventListener('load', () => {
    _emitSurfaceReady({ embed_url: embedUrl, title });
  }, { once: true });
  overlay.appendChild(iframe);

  // ESC closes — keyboard fallback for TV setups with attached keyboards.
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !document.fullscreenElement) {
      closeBtn.click();
    }
  });

  document.body.appendChild(overlay);
  _clearFallback();
}

_bootstrap();
