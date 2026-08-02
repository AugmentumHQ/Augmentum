/**
 * cast-picker.js — Reusable popover for picking a target device.
 *
 * Any surface that can produce castable content (the audio mini-player,
 * the detail panel, future image cards) opens this picker anchored to a
 * Cast button. The picker fetches saved devices, filters by the target
 * capability, lets the user pick one, dispatches the cast, and closes.
 *
 * Public API:
 *   openCastPicker({ anchor, capability, content, onCast, onError })
 *
 * Args:
 *   anchor      — HTMLElement the popover positions itself against
 *   capability  — capability ID to filter devices by (e.g. 'media.audio_play@1')
 *   content     — { contentUrl, title, posterUrl?, startTimeS?, contentType?,
 *                   contentKey?, fileId?, streamUrl?, author?, source?,
 *                   metadata? }
 *                 streamUrl + author + source apply to media.audio_play@1
 *                 when there's no fileId — route an arbitrary HTTP audio
 *                 URL (e.g. internet radio) through cast-audio with
 *                 override metadata for the TV chrome.
 *   onCast      — (device, result) => void  fired after a successful cast
 *   onError     — (err) => void              fired if the cast fails
 *
 * The picker dedicates one popover element to the document at a time;
 * opening a new picker closes any prior one.
 */


import { escapeHtml, showToast } from './app.js';
import { notifyCastStarted } from './cast-shelf.js';
import { openMediaServers } from './media-servers.js';


let _activeOverlay = null;
let _outsideHandler = null;
let _escHandler = null;

// Per-open toggle: by default the picker shows only paired Augmentum
// receivers (TVs running the cast-receiver shell — APK or browser tab).
// The footer disclosure flips this to fall back to the legacy
// /api/devices source, which surfaces every SSDP/DLNA renderer on the
// LAN — useful when casting to a non-Augmentum-aware DLNA speaker, but
// noisy because Emby's mobile app advertises every phone as a renderer.
// Resets to `false` whenever the picker is reopened from scratch.
let _showDiscovered = false;


/* ------------------------------------------------------------------ *\
   Public entry
\* ------------------------------------------------------------------ */


export async function openCastPicker({
  anchor,
  capability,
  content = {},
  onCast = null,
  onError = null,
} = {}) {
  if (!anchor) return;
  closeCastPicker();

  // Always start in trusted-only mode — the footer disclosure can flip
  // it for this session but each fresh open returns to the clean view.
  _showDiscovered = false;

  const overlay = document.createElement('div');
  overlay.className = 'cast-picker';
  overlay.innerHTML = _shellHtml();
  document.body.appendChild(overlay);
  _positionAnchored(overlay, anchor);
  _activeOverlay = overlay;

  // Outside-click + Escape close.
  _outsideHandler = (e) => {
    if (overlay.contains(e.target)) return;
    if (anchor.contains(e.target)) return;
    closeCastPicker();
  };
  _escHandler = (e) => {
    if (e.key === 'Escape') closeCastPicker();
  };
  // Defer attaching outside handler so the click that opened the picker
  // doesn't also dismiss it.
  setTimeout(() => {
    if (!_activeOverlay) return;
    document.addEventListener('click', _outsideHandler, true);
    document.addEventListener('keydown', _escHandler);
  }, 0);

  // Reposition on viewport changes so the popover stays anchored.
  const reposition = () => _positionAnchored(overlay, anchor);
  window.addEventListener('resize', reposition);
  window.addEventListener('scroll', reposition, true);
  overlay._cleanup = () => {
    window.removeEventListener('resize', reposition);
    window.removeEventListener('scroll', reposition, true);
  };

  // Load devices, filter, render. The loader is re-entrant — flipping
  // the discovery disclosure calls it again to repaint without closing
  // the popover.
  const list = overlay.querySelector('[data-cp-list]');

  async function reload() {
    list.innerHTML = `<div class="cp-loading">${
      _showDiscovered ? 'Looking for nearby devices…' : 'Looking for your paired devices…'
    }</div>`;
    try {
      const devices = _showDiscovered
        ? await _loadDiscovered(capability)
        : await _loadTrusted();
      _renderDeviceList(overlay, devices, {
        capability,
        content,
        onCast,
        onError,
        mode: _showDiscovered ? 'discovered' : 'trusted',
      });
    } catch (err) {
      list.innerHTML = `
        <div class="cp-empty">
          <div class="cp-empty-title">Couldn't load devices</div>
          <div class="cp-empty-sub">${escapeHtml(String(err?.message || err))}</div>
        </div>
      `;
    }
    _wireFooter(overlay, reload);
  }

  await reload();
}


async function _loadTrusted() {
  // Primary source: receivers the user has paired through the cast
  // pair flow (Augmentum APK on a TV, or any browser running the
  // cast-receiver shell). These are the durable "my devices" set.
  const resp = await fetch('/api/cast/trusted-receivers');
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const body = await resp.json();
  return (body.receivers || []).map(_trustedToDevice);
}


async function _loadDiscovered(capability) {
  // Disclosure source: every SSDP/DLNA renderer + any Saved Device the
  // user added by hand. Filtered by capability the same way the legacy
  // picker always did, so the experience is unchanged for users who
  // genuinely cast to a non-Augmentum-aware speaker.
  const resp = await fetch('/api/devices');
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const body = await resp.json();
  return (body.devices || []).filter(d => _supportsCapability(d, capability));
}


function _trustedToDevice(receiver) {
  // Adapt the trusted-receiver shape to the picker's device shape so
  // the existing _renderDeviceList works unchanged. The ``__trusted``
  // marker tells _dispatchCast to route via /api/cast/send instead of
  // the legacy /api/devices/{id}/{cap}/play path. ``registration_id``
  // is the runtime WS connection id the dispatch endpoint actually
  // takes — only present when the receiver is currently connected
  // (offline receivers have no live connection to send to).
  return {
    id: receiver.id,
    label: receiver.label || 'Augmentum receiver',
    driver: receiver.platform || 'augmentum',
    status: receiver.connected ? 'online' : 'offline',
    // The cast-receiver shell mounts all surface kinds uniformly, so
    // every paired device supports every capability the picker knows
    // about. _supportsCapability is bypassed entirely for trusted
    // entries (no filter on the trusted-load path).
    capabilities: [
      'media.audio_play@1',
      'media.video_play@1',
      'display.image_show@1',
      'display.web_show@1',
    ],
    bindings: [],
    __trusted: true,
    __trusted_id: receiver.id,
    __registration_id: receiver.registration_id || '',
    __wol_ready: !!receiver.wol_ready,
  };
}


function _wireFooter(overlay, reload) {
  const toggle = overlay.querySelector('[data-cp-toggle-source]');
  if (toggle) {
    toggle.textContent = _showDiscovered
      ? 'Show paired devices only'
      : 'Show all discovered devices';
    toggle.onclick = (e) => {
      e.preventDefault();
      _showDiscovered = !_showDiscovered;
      reload();
    };
  }
  const pair = overlay.querySelector('[data-cp-pair]');
  if (pair) {
    pair.onclick = (e) => {
      e.preventDefault();
      // The cast-control surface has the canonical pair UX — full
      // receiver list, QR pair instructions, prefs editing. Better
      // than minting a parallel inline pair flow.
      closeCastPicker();
      window.open('/ui/cast-control/', '_blank', 'noopener');
    };
  }
}


export function closeCastPicker() {
  if (!_activeOverlay) return;
  if (_activeOverlay._cleanup) _activeOverlay._cleanup();
  document.removeEventListener('click', _outsideHandler, true);
  document.removeEventListener('keydown', _escHandler);
  _outsideHandler = null;
  _escHandler = null;
  _activeOverlay.remove();
  _activeOverlay = null;
}


/* ------------------------------------------------------------------ *\
   Rendering
\* ------------------------------------------------------------------ */


function _shellHtml() {
  return `
    <div class="cp-popover" role="menu" aria-label="Cast to a device">
      <div class="cp-header">
        <span>Cast to…</span>
        <button class="cp-manage" data-cp-manage type="button">Manage</button>
      </div>
      <div class="cp-list" data-cp-list></div>
      <div class="cp-foot">
        <a href="#" data-cp-pair>Pair a new device</a>
        <span class="cp-foot-sep">·</span>
        <a href="#" data-cp-toggle-source>Show all discovered devices</a>
      </div>
    </div>
  `;
}


function _renderDeviceList(overlay, devices, { capability, content, onCast, onError, mode = 'trusted' }) {
  const list = overlay.querySelector('[data-cp-list]');
  if (!devices.length) {
    const emptyCopy = mode === 'trusted'
      ? {
          title: 'No paired devices',
          sub: 'Install the Augmentum receiver on a TV and tap <strong>Pair a new device</strong> below.',
        }
      : {
          title: 'No devices ready',
          sub: 'Add a TV or speaker in <strong>Connected Devices</strong>, then come back.',
        };
    list.innerHTML = `
      <div class="cp-empty">
        <div class="cp-empty-title">${emptyCopy.title}</div>
        <div class="cp-empty-sub">${emptyCopy.sub}</div>
      </div>
    `;
    overlay.querySelector('[data-cp-manage]')?.addEventListener('click', () => {
      closeCastPicker();
      openMediaServers();
    });
    return;
  }

  list.innerHTML = devices.map((d, i) => `
    <button class="cp-item" data-cp-device-idx="${i}" type="button"
            ${d.status !== 'online' ? 'data-cp-offline="1"' : ''}>
      <div class="cp-item-icon">${_iconForDriver(d.driver)}</div>
      <div class="cp-item-meta">
        <div class="cp-item-label">${escapeHtml(d.label)}</div>
        <div class="cp-item-sub">
          ${escapeHtml(_humanDriver(d.driver))}
          ${d.status === 'online' ? '' : ' &middot; <span class="cp-offline-tag">offline</span>'}
        </div>
      </div>
      <div class="cp-item-arrow" aria-hidden="true">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="9 18 15 12 9 6"/></svg>
      </div>
    </button>
  `).join('');

  list.querySelectorAll('[data-cp-device-idx]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const idx = Number(btn.dataset.cpDeviceIdx);
      const device = devices[idx];
      if (!device) return;
      btn.classList.add('cp-item-loading');
      btn.disabled = true;
      try {
        const result = await _dispatchCast(device, capability, content);
        if (!result.ok) throw new Error(result.message || result.code || 'Cast failed');
        showToast(`Casting to ${device.label}`, 'success', 2500);
        notifyCastStarted();
        if (typeof onCast === 'function') onCast(device, result);
        closeCastPicker();
      } catch (err) {
        btn.classList.remove('cp-item-loading');
        btn.disabled = false;
        showToast(`Cast failed: ${err.message || err}`, 'error', 4000);
        if (typeof onError === 'function') onError(err);
      }
    });
  });

  overlay.querySelector('[data-cp-manage]')?.addEventListener('click', () => {
    closeCastPicker();
    openMediaServers();
  });
}


/* ------------------------------------------------------------------ *\
   Cast dispatch
\* ------------------------------------------------------------------ */


async function _dispatchCast(device, capability, content) {
  // Trusted receivers (Augmentum APK / browser cast-receiver) take the
  // unified cast-surface path. Legacy /api/devices entries keep their
  // capability/action routing — the device substrate translates those
  // into driver-specific protocol calls (DLNA SetAVTransportURI etc.).
  if (device.__trusted) {
    if (device.status !== 'online') {
      return {
        ok: false,
        message: 'Receiver is offline — open it on the TV or wake it from Cast control.',
        code: 'offline',
      };
    }
    return _dispatchToTrusted(device, capability, content);
  }

  // Map our normalized content shape onto the action args expected by
  // the substrate's media/display capabilities. Missing fields stay
  // missing — the driver and capability know what's required.
  const args = {};
  if (content.contentUrl)  args.content_url  = content.contentUrl;
  if (content.contentType) args.content_type = content.contentType;
  if (content.title)       args.title        = content.title;
  if (content.posterUrl)   args.poster_url   = content.posterUrl;
  if (content.startTimeS)  args.start_time_s = content.startTimeS;
  if (content.contentKey)  args.content_key  = content.contentKey;
  if (content.fileId)      args.file_id      = content.fileId;
  if (content.author)      args.author       = content.author;
  if (content.artist)      args.artist       = content.artist;
  if (content.album)       args.album        = content.album;
  if (content.imageUrl && capability === 'display.image_show@1') {
    args.image_url = content.imageUrl;
  }

  // Same-origin URLs are auth-protected on our side. Flag them so the
  // server tokenizes the URL before handing it to the TV — the TV
  // doesn't carry our auth cookie and would otherwise hit a 401. The
  // server-side handler swaps content_url/image_url for a short-lived
  // public /api/cast/blob/{token} URL.
  const sourceUrl = content.contentUrl || content.imageUrl || '';
  if (sourceUrl && _isSameOrigin(sourceUrl)) {
    args.requires_auth = true;
  }

  // For display.* capabilities the action is "show" or "load_url"; for
  // media.* it's "play". Other capabilities go through their own path.
  const action = _defaultActionForCapability(capability);
  const url = `/api/devices/${encodeURIComponent(device.id)}/${encodeURIComponent(capability)}/${encodeURIComponent(action)}`;
  let resp;
  try {
    resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ args }),
    });
  } catch (err) {
    return { ok: false, message: 'network unreachable', code: 'NETWORK_ERROR' };
  }
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    return { ok: false, message: body.message || body.detail || `HTTP ${resp.status}`, code: body.code || '' };
  }
  return body;
}


async function _dispatchToTrusted(device, capability, content) {
  // Build the surface descriptor. Prefer the dedicated /ui/cast-{kind}/
  // surfaces when we have a fileId — those iframes carry the full
  // chrome (chapter list, subtitles, comic page-fit, etc.). Fall back
  // to the receiver's native media.* shortcut for direct URLs, which
  // is enough for casting a bare audio/video/image stream.
  const state = {};
  if (content.title)       state.title         = content.title;
  if (content.posterUrl)   state.poster_url    = content.posterUrl;
  if (content.contentType) state.content_type  = content.contentType;
  if (content.startTimeS)  state.start_time_s  = content.startTimeS;
  if (content.author)      state.author        = content.author;
  if (content.artist)      state.artist        = content.artist;
  if (content.album)       state.album         = content.album;
  if (content.contentKey)  state.content_key   = content.contentKey;
  if (content.fileId)      state.file_id       = content.fileId;

  let surface_kind = 'html.generic';
  let surface_url = content.contentUrl || '';

  if (capability === 'media.audio_play@1') {
    if (content.fileId) {
      surface_kind = 'html.generic';
      surface_url = `/ui/cast-audio/?id=${encodeURIComponent(content.fileId)}`;
    } else if (content.streamUrl) {
      // Arbitrary HTTP audio stream (e.g. Grove internet radio) —
      // route through cast-audio so the TV still shows cover, title,
      // and the VU meter instead of dropping to the receiver's silent
      // native media.audio shortcut.
      const qp = new URLSearchParams();
      qp.set('streamUrl', content.streamUrl);
      if (content.title)     qp.set('title',  content.title);
      if (content.author)    qp.set('author', content.author);
      if (content.posterUrl) qp.set('cover',  content.posterUrl);
      if (content.source)    qp.set('source', content.source);
      surface_kind = 'html.generic';
      surface_url = `/ui/cast-audio/?${qp.toString()}`;
    } else {
      surface_kind = 'media.audio';
      surface_url = content.contentUrl || '';
    }
  } else if (capability === 'media.video_play@1') {
    // Live TV: routed through /ui/cast-livetv/ so the receiver
    // surface can mint its own play session (Emby's HLS URL carries
    // the user's api_key — we never send that to the receiver any
    // more than to the browser). Content shape: contentKey starts
    // with "livetv:" + extra carries server_id / channel_id /
    // number / logo_url / now.
    if (content.contentKey && content.contentKey.startsWith('livetv:')) {
      const meta = content.metadata || {};
      const qp = new URLSearchParams();
      qp.set('server_id',  String(meta.server_id  || ''));
      qp.set('channel_id', String(meta.channel_id || ''));
      if (content.title)    qp.set('title',    content.title);
      if (meta.number)      qp.set('number',   String(meta.number));
      if (meta.logo_url)    qp.set('logo_url', String(meta.logo_url));
      if (meta.now)         qp.set('now',      String(meta.now));
      surface_kind = 'html.generic';
      surface_url  = `/ui/cast-livetv/?${qp.toString()}`;
    } else if (content.fileId) {
      surface_kind = 'html.generic';
      surface_url = `/ui/cast-video/?id=${encodeURIComponent(content.fileId)}`;
    } else {
      surface_kind = 'media.video';
      surface_url = content.contentUrl || '';
    }
  } else if (capability === 'display.image_show@1') {
    surface_kind = 'media.image';
    surface_url = content.imageUrl || content.contentUrl || '';
  } else if (capability === 'display.web_show@1') {
    surface_kind = 'html.generic';
    surface_url = content.contentUrl || '';
  }

  if (!surface_url) {
    return { ok: false, message: 'Nothing to cast — missing content URL.', code: 'no_url' };
  }

  // /api/cast/send takes the runtime WS registration_id, not the
  // durable trusted_id. The trusted-receivers list only includes a
  // registration_id when the receiver is currently connected; we
  // already guarded for offline above, so absence here is a server-
  // side race (paired, listed as connected, but dropped the WS
  // mid-render) — surface a clear message rather than 404 on send.
  if (!device.__registration_id) {
    return {
      ok: false,
      message: 'Receiver just disconnected — refresh and try again.',
      code: 'no_registration',
    };
  }
  let resp;
  try {
    resp = await fetch('/api/cast/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        receiver_id: device.__registration_id,
        surface_kind,
        surface_url,
        state,
        slot: 'main',
      }),
    });
  } catch (err) {
    return { ok: false, message: 'network unreachable', code: 'NETWORK_ERROR' };
  }
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    return {
      ok: false,
      message: body.detail || body.message || `HTTP ${resp.status}`,
      code: body.code || '',
    };
  }
  return { ok: true, ...body };
}


function _defaultActionForCapability(capability) {
  if (capability === 'display.image_show@1' || capability === 'display.text_show@1') return 'show';
  if (capability === 'display.web_show@1') return 'load_url';
  if (capability === 'audio.tts_say@1') return 'say';
  if (capability === 'lighting.set_state@1') return 'on';
  return 'play';
}


/* ------------------------------------------------------------------ *\
   Helpers
\* ------------------------------------------------------------------ */


function _isSameOrigin(url) {
  if (!url) return false;
  try {
    const u = new URL(url, window.location.origin);
    return u.origin === window.location.origin;
  } catch {
    // Treat relative URLs (no protocol) as same-origin.
    return !/^[a-z][a-z0-9+\-.]*:\/\//i.test(url);
  }
}


function _supportsCapability(device, capability) {
  const caps = device.capabilities || [];
  if (caps.includes(capability)) return true;
  // Bindings (cross-driver) — check those too.
  for (const b of device.bindings || []) {
    if ((b.capabilities || []).includes(capability)) return true;
  }
  return false;
}


function _humanDriver(driver) {
  const map = {
    dlna: 'DLNA / UPnP',
    cast: 'Google Cast',
    airplay: 'AirPlay',
    sonos: 'Sonos',
    hue: 'Hue',
    matter: 'Matter',
    augmentum_surface: 'In-app',
    augmentum: 'Augmentum',
    'android-tv': 'Augmentum on Android TV',
    browser: 'Augmentum in a browser',
    phone: 'Phone',
  };
  return map[driver] || driver;
}


function _iconForDriver(driver) {
  if (driver === 'augmentum_surface' || driver === 'augmentum'
      || driver === 'android-tv' || driver === 'browser') {
    return `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M9 9h6M9 13h6M9 17h4"/></svg>`;
  }
  if (driver === 'hue' || driver === 'lighting') {
    return `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a7 7 0 0 0-4 12.7c.7.5 1 1.3 1 2.1V19a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1v-2.2c0-.8.3-1.6 1-2.1A7 7 0 0 0 12 2z"/><path d="M9 22h6"/></svg>`;
  }
  return `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></svg>`;
}


function _positionAnchored(overlay, anchor) {
  const rect = anchor.getBoundingClientRect();
  const popover = overlay.querySelector('.cp-popover');
  if (!popover) return;
  // Default: anchor below + right-aligned to the button.
  const width = 280;
  const margin = 8;
  let left = rect.right - width;
  let top = rect.bottom + margin;
  // Keep within viewport
  if (left < 8) left = 8;
  if (left + width > window.innerWidth - 8) {
    left = window.innerWidth - width - 8;
  }
  // If there isn't room below, flip above
  const expectedHeight = 320;
  if (top + expectedHeight > window.innerHeight - 8 && rect.top > expectedHeight + margin) {
    top = rect.top - expectedHeight - margin;
  }
  Object.assign(popover.style, {
    position: 'fixed',
    width: `${width}px`,
    left: `${left}px`,
    top: `${top}px`,
  });
}
