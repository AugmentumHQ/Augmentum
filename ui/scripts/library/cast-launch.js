/**
 * library/cast-launch.js — "cast this game to a TV" flow.
 *
 * Public:
 *   castGame(item) — opens the device picker modal, then dispatches a
 *                    CMD_SURFACE_OPEN at /ui/play/ or /ui/play-web/
 *                    depending on the artifact kind. Returns a promise
 *                    that resolves with {receiver_id, surface_id} on
 *                    success or rejects on cancel / failure.
 *
 * Internal:
 *   _resolvePlayUrl(item)  - maps artifact kind → play URL
 *   _listReceivers()       - GET /api/cast/receivers
 *   _pickReceiver(list)    - DOM modal with a radio-list, resolves to id
 *   _dispatch(receiverId, item, url)  - POST /api/cast/send
 *
 * Why a separate module: the cast flow is also reachable from the
 * games-browse tab (browse a curated web game → cast straight to TV
 * without pinning first). Keeping it free of DetailPane state means
 * other surfaces can import castGame() directly.
 */

import { escapeHtml, showToast } from '../app.js';

const _CAST_SUPPORTED_KINDS = new Set([
  'emulator_rom', 'streamed_game', 'js13k_game', 'web_app',
]);

// Canonical kind for a library item. Library items don't carry a top-level
// ``kind`` — the value lives under ``metadata.kind`` for artifacts (games
// stamp "game", ROMs "emulator_rom", app builds "application") and is absent
// for publications (whose kind is projected into ``format`` by the union).
// Reading ``item.kind`` alone is why cast was dead for EVERY item; resolve
// from all three sources so games/ROMs classify correctly.
export function resolveKind(item) {
  const explicit = String(item?.kind || item?.metadata?.kind || '').toLowerCase();
  if (explicit) return explicit;
  return String(item?.format || '').toLowerCase();
}

// Framable URL for embed-style games. Pinned games store the resolved,
// CSP-safe variant under ``metadata.embed_src`` (the raw source page under
// ``metadata.embed_url``); browse items carry a top-level ``embed_url``.
function resolveEmbed(item) {
  return item?.metadata?.embed_src
    || item?.metadata?.embed_url
    || item?.embed_url
    || '';
}

function _resolvePlayUrl(item) {
  // Local fallback when the /classify endpoint isn't reachable (older
  // server, profile registry unavailable). Returns the surface URL to
  // send to the receiver — relative is fine because the receiver loads
  // it from the same origin as the server that minted the cast send.
  const kind = resolveKind(item);
  if (kind === 'emulator_rom' || kind === 'streamed_game') {
    return `/ui/play/?title_id=${encodeURIComponent(item.id)}&kiosk=1`;
  }
  // metadata.kind on a pinned game is the bare "game"; treat it as an
  // embed-mode web game alongside the explicit js13k/web_app kinds.
  if (kind === 'js13k_game' || kind === 'web_app' || kind === 'game') {
    const embedUrl = resolveEmbed(item);
    if (!embedUrl) return '';
    const title = item.display_name || item.title || item.name || 'Web game';
    return `/ui/play-web/?embed_url=${encodeURIComponent(embedUrl)}`
      + `&title=${encodeURIComponent(title)}&kiosk=1`;
  }
  return '';
}

async function _classify(item) {
  // Ask the server which CastStrategy + input chain to use. Returns
  // { surface_url, input_chain, keymap } on success, or null when the
  // endpoint is unavailable — callers fall back to _resolvePlayUrl.
  // Idempotent + cheap; no persistence side-effects on the server.
  if (!item?.id) return null;
  try {
    const r = await fetch(
      `/api/cast/games/${encodeURIComponent(item.id)}/classify`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          title_id: String(item.id),
          kind: resolveKind(item),
          display_name: item.display_name || item.title || item.name || '',
          embed_url: resolveEmbed(item),
          metadata: item.metadata || {},
        }),
      },
    );
    if (!r.ok) return null;
    const body = await r.json();
    const prepared = body?.prepared;
    if (!prepared?.surface_url) return null;
    return {
      surface_url: String(prepared.surface_url),
      input_chain: Array.isArray(prepared.input_chain)
        ? prepared.input_chain.slice()
        : ['gamepad_api'],
      keymap: prepared.keymap || null,
      strategy: String(prepared.strategy || 'shim'),
    };
  } catch (_) {
    return null;
  }
}

// Cast-everything: any library item with an id can be surfaced on a TV —
// games via their play URL, publications via the sandboxed launcher, and
// every other artifact (pdf/pptx/xlsx/docs/images/single-file apps) via the
// server-rendered preview. So the only hard gate is "does it have an id".
export function isCastable(item) {
  return !!(item && item.id);
}

// True for the game kinds that get the input-chain / classifier treatment.
function _isGameKind(item) {
  const kind = resolveKind(item);
  if (_CAST_SUPPORTED_KINDS.has(kind)) return true;
  // A pinned game carries metadata.kind="game" + an embed URL.
  return kind === 'game' && !!resolveEmbed(item);
}

function _isPublication(item) {
  return typeof item?.id === 'string' && item.id.startsWith('pub_');
}

async function _listReceivers() {
  const r = await fetch('/api/cast/receivers', { credentials: 'same-origin' });
  if (!r.ok) throw new Error(`receivers HTTP ${r.status}`);
  const body = await r.json();
  return Array.isArray(body.receivers) ? body.receivers : [];
}

function _pickReceiver(receivers) {
  // Single-instance modal — strip any earlier device picker that's
  // still up before mounting a new one.
  document.querySelectorAll('.cast-launch-picker').forEach(el => el.remove());

  return new Promise((resolve, reject) => {
    const overlay = document.createElement('div');
    overlay.className = 'cast-launch-picker';
    overlay.style.cssText = `
      position: fixed; inset: 0; background: rgba(0, 0, 0, 0.6);
      display: flex; align-items: center; justify-content: center;
      z-index: 10000; padding: 24px;
    `;
    const card = document.createElement('div');
    card.style.cssText = `
      background: #1c1c1f; border: 1px solid #2c2c2c; border-radius: 8px;
      max-width: 420px; width: 100%; padding: 20px; color: #e7e9ee;
      font: 13px/1.4 "Source Sans 3", "Inter", system-ui, sans-serif;
    `;
    const head = document.createElement('div');
    head.style.cssText = 'font-size: 15px; font-weight: 600; margin-bottom: 8px;';
    head.textContent = receivers.length
      ? 'Cast to TV'
      : 'No TVs connected';
    card.appendChild(head);

    const sub = document.createElement('div');
    sub.style.cssText = 'font-size: 12px; opacity: 0.6; margin-bottom: 16px;';
    sub.textContent = receivers.length
      ? 'Pick a receiver — the game opens on it immediately.'
      : 'Open Augmentum on your TV and pair it first.';
    card.appendChild(sub);

    if (receivers.length) {
      const list = document.createElement('div');
      list.style.cssText = 'display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px;';
      for (const r of receivers) {
        const row = document.createElement('button');
        row.type = 'button';
        row.style.cssText = `
          background: #232328; border: 1px solid #2c2c2c; border-radius: 6px;
          padding: 10px 12px; color: #e7e9ee; text-align: left; cursor: pointer;
          display: flex; align-items: center; gap: 10px;
        `;
        row.innerHTML = `
          <span style="font-size: 16px;">📺</span>
          <span style="flex: 1;">
            <div style="font-weight: 500;">${escapeHtml(r.label || 'Untitled TV')}</div>
            <div style="font-size: 11px; opacity: 0.55;">${escapeHtml(r.platform || 'receiver')}</div>
          </span>
        `;
        row.addEventListener('mouseenter', () => row.style.borderColor = '#6ea2ef');
        row.addEventListener('mouseleave', () => row.style.borderColor = '#2c2c2c');
        row.addEventListener('click', () => {
          overlay.remove();
          resolve(r.registration_id);
        });
        list.appendChild(row);
      }
      card.appendChild(list);
    }

    const actions = document.createElement('div');
    actions.style.cssText = 'display: flex; justify-content: flex-end; gap: 8px;';
    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.cssText = `
      background: transparent; border: 1px solid #333; color: #ccc;
      padding: 6px 14px; border-radius: 4px; cursor: pointer;
    `;
    cancelBtn.addEventListener('click', () => {
      overlay.remove();
      reject(new Error('cancelled'));
    });
    actions.appendChild(cancelBtn);
    card.appendChild(actions);

    overlay.appendChild(card);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        overlay.remove();
        reject(new Error('cancelled'));
      }
    });
    // ESC also cancels.
    const esc = (e) => {
      if (e.key === 'Escape') {
        document.removeEventListener('keydown', esc);
        overlay.remove();
        reject(new Error('cancelled'));
      }
    };
    document.addEventListener('keydown', esc);
    document.body.appendChild(overlay);
  });
}

async function _dispatch(receiverId, surfaceUrl, state, surfaceKind) {
  // ``state`` is forwarded by the receiver shell to the play iframe via
  // surface_init — its shape is the contract surfaces depend on (title,
  // cast_source, artifact_id, cast_input_config, cast_strategy). Surfaces
  // that don't need an input chain (Studio docs, slides) pass null.
  const r = await fetch('/api/cast/send', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({
      receiver_id: receiverId,
      surface_kind: surfaceKind || 'html.generic',
      surface_url: surfaceUrl,
      slot: 'main',
      state,
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || err.error || `HTTP ${r.status}`);
  }
  return r.json();
}

/**
 * Generic "send this URL to a TV" primitive. Lists receivers, shows the
 * picker modal, dispatches, returns ``{...resp, receiver_id,
 * receiver_label}``. Any surface that has a URL the receiver can load
 * (Studio artifact previews, library items, files-mode previews) calls
 * this directly instead of reimplementing the picker.
 *
 * @param {object} item - {id, display_name?, title?, name?}
 * @param {string} surfaceUrl - URL the receiver should load
 * @param {object} [opts]
 * @param {string}      [opts.castSource='library']      - analytics tag
 * @param {string}      [opts.surfaceKind='html.generic']
 * @param {object|null} [opts.castInputConfig=null]      - adapter chain + keymap
 * @param {string}      [opts.castStrategy='shim']
 * @param {string}      [opts.fallbackTitle='Surface']   - used when item has no name
 */
export async function castSurface(item, surfaceUrl, opts = {}) {
  if (!item || !surfaceUrl) {
    throw new Error('castSurface requires item + surfaceUrl');
  }
  const {
    castSource = 'library',
    surfaceKind = 'html.generic',
    castInputConfig = null,
    castStrategy = 'shim',
    fallbackTitle = 'Surface',
  } = opts;

  let receivers;
  try {
    receivers = await _listReceivers();
  } catch (err) {
    showToast(`Couldn’t list TVs: ${err.message || err}`, 'error');
    throw err;
  }
  const receiverId = await _pickReceiver(receivers);

  const state = {
    title: item.display_name || item.title || item.name || fallbackTitle,
    cast_source: castSource,
    artifact_id: item.id || '',
    cast_input_config: castInputConfig,
    cast_strategy: castStrategy,
  };

  let resp;
  try {
    resp = await _dispatch(receiverId, surfaceUrl, state, surfaceKind);
  } catch (err) {
    showToast(`Cast failed: ${err.message || err}`, 'error');
    throw err;
  }
  const receiverLabel = receivers.find(
    r => r.registration_id === receiverId,
  )?.label || 'your TV';
  showToast(`Now playing on ${receiverLabel}.`, 'info');
  return { ...resp, receiver_id: receiverId, receiver_label: receiverLabel };
}

// Send a library item to a TV. Dispatches by item nature:
//   game kinds   → classifier + game play URL (with input chain)
//   publications → the sandboxed /play launcher
//   everything else → the server-rendered artifact preview
// Named castGame for back-compat; it casts any castable item now.
export async function castGame(item) {
  if (!item || !item.id) {
    showToast('Nothing to cast for this item.', 'warn');
    throw new Error('not_castable');
  }

  // 1. Games: classify first so the receiver gets the right strategy +
  //    adapter chain. Fall back to the local URL resolver when the
  //    server-side classifier is unavailable (older server, registry init).
  if (_isGameKind(item)) {
    const prepared = await _classify(item);
    const surfaceUrl = prepared?.surface_url || _resolvePlayUrl(item);
    if (surfaceUrl) {
      return castSurface(item, surfaceUrl, {
        castSource: 'library',
        castInputConfig: prepared
          ? { adapters: prepared.input_chain, keymap: prepared.keymap || null }
          : null,
        castStrategy: prepared?.strategy || 'shim',
        fallbackTitle: 'Game',
      });
    }
    // No game URL resolved (e.g. missing embed) — fall through to a generic
    // surface cast rather than dead-ending with an error.
  }

  // 2. Publications: the sandboxed launcher renders the saved app on the TV.
  if (_isPublication(item)) {
    return castSurface(
      item, `/api/library/play/${encodeURIComponent(item.id)}`,
      { castSource: 'library', surfaceKind: 'html.generic', fallbackTitle: 'App' },
    );
  }

  // 3. Any other artifact (pdf/pptx/xlsx/docs/images/single-file apps) —
  //    the server preview renders it and the receiver just loads that URL.
  return castSurface(
    item, `/api/artifacts/${encodeURIComponent(item.id)}/preview`,
    { castSource: 'library', surfaceKind: 'html.generic', fallbackTitle: 'Item' },
  );
}
