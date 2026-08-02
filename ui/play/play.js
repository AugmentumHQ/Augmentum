/**
 * /ui/play/ — standalone URL-addressable play stage for emulator ROMs
 * and AGSP-streamed games.
 *
 * Query params:
 *   title_id  (required)  the artifact id (titles row id)
 *   kiosk     (optional)  '1' to suppress player UI chrome — used by
 *                          cast-receiver embeds where the receiver
 *                          shell owns close affordances
 *
 * Lifecycle:
 *   1. Fetch /api/titles/{title_id} to get display name + metadata so
 *      the existing stage functions get a real artifact object.
 *   2. Delegate to openEmulatorStage() — it handles the
 *      browser-runtime vs WebRTC-streamed branch internally.
 *   3. On stage close, postMessage 'augmentum.surface_closed' to the
 *      parent (cast-receiver shell) so it can release its slot.
 *
 * NOT a kiosk-only surface — the same URL works for in-tab usage so a
 * power-user can bookmark a specific game. Kiosk just toggles chrome.
 */

import { openEmulatorStage } from '../scripts/emulator-stage.js';

const _fallback = document.getElementById('play-fallback');
const _fallbackDetail = document.getElementById('play-fallback-detail');

function _setFallback(title, detail) {
  if (!_fallback) return;
  const t = _fallback.querySelector('.play-fallback-title');
  if (t) t.textContent = title;
  if (_fallbackDetail) _fallbackDetail.textContent = detail || '';
}

function _clearFallback() {
  if (_fallback && _fallback.parentNode) _fallback.parentNode.removeChild(_fallback);
}

async function _bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const titleId = params.get('title_id');
  const kiosk = params.get('kiosk') === '1';
  if (!titleId) {
    _setFallback('Missing title_id', 'Add ?title_id=<id> to the URL.');
    return;
  }

  let artifact;
  try {
    const r = await fetch(`/api/titles/${encodeURIComponent(titleId)}`, {
      credentials: 'same-origin',
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    artifact = await r.json();
  } catch (err) {
    _setFallback('Title not found', String(err.message || err));
    return;
  }

  _clearFallback();

  try {
    await openEmulatorStage(artifact, { kiosk });
  } catch (err) {
    _setFallback('Couldn’t launch', String(err.message || err));
    return;
  }

  // Surface-ready handshake — receiver shell forwards anything that
  // isn't a surface_state echo as a generic surface_event over its WS,
  // which lands in cast_event_store. The controller phone can poll the
  // event store (or, later, subscribe via SSE) to confirm the iframe
  // mounted before flipping its "now playing" UI from optimistic to
  // confirmed.
  //
  // Uses `type:` (not `kind:`) so it matches the receiver's iframe-
  // message forwarder convention. The downstream cast_input frame
  // we receive from the receiver still uses `kind:` because that's
  // a different protocol direction (server-routed cmd, not iframe-
  // emitted event).
  try {
    window.parent?.postMessage({
      type: 'augmentum.surface_ready',
      surface: 'play.emulator',
      title_id: titleId,
    }, '*');
  } catch (_) { /* not embedded — ignore */ }
}

_bootstrap();
