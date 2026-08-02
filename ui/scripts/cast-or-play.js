/**
 * cast-or-play.js — Central interception point for media playback.
 *
 * Every play surface (audio player, video player, image viewer, etc.)
 * calls castOrPlay(...) instead of playing locally. If a device is
 * armed, the play call is routed to that device via the device
 * substrate API. Otherwise the caller's local fallback runs.
 *
 * This keeps the per-surface code free of "is something armed?"
 * checks — each surface just declares its play intent and the helper
 * decides where it lands.
 */

import { getArmed, isArmed, disarm } from './armed-device.js';
import { showToast } from './app.js';


/**
 * Route a play action through the armed device, or fall back to local
 * playback.
 *
 * Arguments
 * ---------
 *   capability      — capability id (e.g. 'media.video_play@1', 'media.audio_play@1')
 *   args            — capability args (content_url, title, file_id, requires_auth, ...)
 *   fallback        — async () => any  — invoked when no device is armed
 *   surface         — short label for diagnostic logs (e.g. 'librivox', 'video-pill')
 *
 * Returns
 * -------
 *   { cast: true, deviceId, result } if the call routed to a device.
 *   { cast: false, fallback: <whatever fallback returned> } otherwise.
 *
 *   On cast failure, the helper disarms and falls back to local play so
 *   the user isn't stuck staring at a dead pill.
 */
export async function castOrPlay({ capability, args = {}, fallback, surface = 'unknown' } = {}) {
  if (!isArmed()) {
    const fallbackResult = fallback ? await fallback() : undefined;
    return { cast: false, fallback: fallbackResult };
  }

  const armed = getArmed();
  const deviceId = armed.deviceId;
  const url = `/api/devices/${encodeURIComponent(deviceId)}/${encodeURIComponent(capability)}/play`;

  let resp;
  try {
    resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ args }),
    });
  } catch (err) {
    console.warn(`[cast-or-play:${surface}] network error casting to ${armed.label}:`, err);
    showToast?.(`Couldn't reach ${armed.label} — playing here instead`, 'error', 3500);
    disarm();
    const fallbackResult = fallback ? await fallback() : undefined;
    return { cast: false, fallback: fallbackResult, error: String(err) };
  }

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    const detail = body.message || body.detail || `HTTP ${resp.status}`;
    console.warn(`[cast-or-play:${surface}] cast failed:`, detail, body);
    showToast?.(`${armed.label} couldn't play this — falling back to here`, 'error', 4000);
    // Don't auto-disarm on per-content failures (e.g. unsupported codec).
    // The user may still want subsequent items to go to the same device.
    const fallbackResult = fallback ? await fallback() : undefined;
    return { cast: false, fallback: fallbackResult, error: detail };
  }

  // Always toast — the previous "once per session" gate created a
  // confusing UX where the first cast confirmed visually but every
  // subsequent one routed silently, leaving users wondering why
  // playback "disappeared" to a different room. The cast-shelf
  // trigger picks up the persistent "has-active" indicator either way.
  const title = (args && (args.title || args.contentKey)) || '';
  const msg = title
    ? `Casting "${String(title).slice(0, 40)}" to ${armed.label}`
    : `Casting to ${armed.label}`;
  showToast?.(msg, 'success', 3000);

  // Kick the cast-shelf into refreshing immediately so the user sees
  // the new session's transport controls land without the 3s poll
  // wait. (Previously called into a separate cast-remote module —
  // folded into cast-shelf as part of the shelf consolidation.)
  import('./cast-shelf.js').then(m => m.notifyCastStarted?.()).catch(() => {});

  const result = await resp.json().catch(() => ({}));
  return { cast: true, deviceId, result };
}


/**
 * Convenience for surfaces that don't have a local fallback (rare —
 * mostly for actions that are cast-only, like "show this image on the TV").
 */
export async function castOnly({ capability, args, surface }) {
  return castOrPlay({ capability, args, surface, fallback: null });
}
