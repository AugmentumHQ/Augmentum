/**
 * avatar-thumbnail.js — ensure each avatar has a usable head-framed
 * thumbnail on the server, so ambient affordances (summon pip, future
 * launcher tiles) can show the user's actual companion face instead
 * of a generic silhouette.
 *
 * Flow:
 *   1. ``ensureAvatarThumbnail(id, vrmUrl, opts)`` checks /thumbnail.
 *   2. If the server returned the placeholder
 *      (``X-Avatar-Thumbnail-Placeholder: 1``), the VRM is loaded into
 *      a dedicated offscreen renderer (preserveDrawingBuffer:true) via
 *      ``renderVRMThumbnail``, the PNG is PUT back, and the avatar
 *      record's ``thumbnail_path`` is updated server-side.
 *   3. Subsequent loads of the same avatar id on this page skip the
 *      probe entirely — see ``_have`` cache.
 *
 * The earlier "sample the live companion canvas" approach was
 * unreliable: the shared avatar.js renderer runs preserveDrawingBuffer:
 * false, so reading its canvas from outside the render loop yielded a
 * transparent buffer most of the time and uploaded an empty PNG that
 * looked broken in the pip. The offscreen-render path here is the
 * same one settings.js's avatar grid uses, so both surfaces converge
 * on a single canonical thumbnail per avatar.
 */

import { renderVRMThumbnail } from './avatar-thumbnail-render.js';

// Per-avatar promise so concurrent callers share one request cycle.
const _inflight = new Map();
// Per-avatar boolean — true once we've confirmed the server has a
// real thumb (either pre-existing or freshly uploaded). Lets later
// calls skip the HEAD-equivalent GET entirely.
const _have = new Set();

function _thumbUrl(avatarId) {
  return `/api/avatar/${encodeURIComponent(avatarId)}/thumbnail`;
}

/**
 * Run the check-and-upload flow for one avatar.
 *
 * @param {string} avatarId         Avatar record id.
 * @param {string} vrmUrl           URL of the .vrm file to render if
 *                                  the server doesn't already have a
 *                                  real thumbnail.
 * @param {object} [opts]           Forwarded to renderVRMThumbnail
 *                                  (e.g. {faceRotationY}).
 * @returns {Promise<string>}       The thumbnail URL the caller can
 *                                  use as a background-image source.
 */
export async function ensureAvatarThumbnail(avatarId, vrmUrl, opts = {}) {
  if (!avatarId) return '';
  const url = _thumbUrl(avatarId);
  if (_have.has(avatarId)) return url;
  if (_inflight.has(avatarId)) return _inflight.get(avatarId);

  const p = (async () => {
    try {
      const r = await fetch(url, { credentials: 'same-origin', cache: 'no-store' });
      const isPlaceholder = r.headers.get('X-Avatar-Thumbnail-Placeholder') === '1';
      if (!isPlaceholder) {
        _have.add(avatarId);
        return url;
      }
      if (!vrmUrl) return url;  // can't render without a source
      const blob = await renderVRMThumbnail(vrmUrl, opts);
      if (!blob) return url;
      const upload = await fetch(url, {
        method: 'PUT',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'image/png' },
        body: blob,
      });
      if (upload.ok) _have.add(avatarId);
    } catch (err) {
      console.warn('[avatar-thumb] ensure failed', err);
    }
    return url;
  })();
  _inflight.set(avatarId, p);
  try { return await p; } finally { _inflight.delete(avatarId); }
}

/**
 * Force a fresh render and upload, ignoring any cached state. Use
 * after the user changes a major visual property (skin tone, outfit,
 * mannerisms.face_rotation_y, etc.) so the pip catches up without a
 * hard reload.
 */
export async function recaptureAvatarThumbnail(avatarId, vrmUrl, opts = {}) {
  if (!avatarId || !vrmUrl) return false;
  const url = _thumbUrl(avatarId);
  try {
    const blob = await renderVRMThumbnail(vrmUrl, opts);
    if (!blob) return false;
    const upload = await fetch(url, {
      method: 'PUT',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'image/png' },
      body: blob,
    });
    if (upload.ok) {
      _have.add(avatarId);
      return true;
    }
  } catch (err) {
    console.warn('[avatar-thumb] recapture failed', err);
  }
  return false;
}
