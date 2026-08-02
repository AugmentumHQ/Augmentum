/**
 * media-resume.js — Remember the last audio/video the user was playing
 * across page refreshes, and offer a one-tap "continue?" prompt on the
 * next app load.
 *
 * Scope: anything driven by ``media-player.js`` (audiobookshelf,
 * LibriVox, podcasts) and the inline video player in ``files/preview.js``
 * (Emby/Jellyfin/Plex/etc.). Grove radio + ambient YT have their own
 * parallel module (``grove-resume.js``) because their prompt fires
 * when the Grove panel opens, not at app init.
 *
 * Why split: Grove's "ambient companion" tone justifies a prompt the
 * MOMENT the user is in that surface. Media-player playback is
 * intentional; the prompt belongs at the app's front door so the user
 * isn't yanked back into a four-hour audiobook they were done with.
 *
 * UX contract (mirrors grove-resume.js):
 *   - Prompt appears at most once per page load.
 *   - Stale entries (>7 days) are silently dropped — don't surface
 *     yesterday's last book if it's been a fortnight.
 *   - "Play" action calls back into the source so it resumes from
 *     wherever the server last knows.
 *   - Dismiss = permanent (user said "stop offering this"); the X is
 *     authoritative until the user clears site data.
 */

import { showToast } from './app.js';
import { getCurrentUser } from './auth.js';
import { getLastPlayed as _groveGetLastPlayed } from './grove-resume.js';

// Per-user storage keys — every read/write must include the active
// user's id, otherwise Profile A's "currently playing" surfaces under
// Profile B on the next reload. `_key()` returns null when no user is
// known yet; callers treat null as "skip" rather than falling back to
// a global key (the old bug we're fixing).
const KEY_LAST_BASE = 'augmentum-media-last-played';
const KEY_DISMISSED_BASE = 'augmentum-media-resume-dismissed';
const FRESHNESS_MS = 7 * 24 * 60 * 60 * 1000;   // 7 days

function _key(base) {
  const u = getCurrentUser();
  return u && u.id ? `${base}::u:${u.id}` : null;
}

/**
 * Record that this media item was just played. Last writer wins —
 * whichever item the user started most recently is what we offer on
 * the next page load.
 *
 * @param {object} entry
 * @param {'audio' | 'video'} entry.kind        Which player surface
 * @param {string} entry.fileId                 File index id (passed back to the player on resume)
 * @param {string} entry.title                  User-facing title
 * @param {string} [entry.subtitle]             Author / show / channel — shown as the toast description
 * @param {string} [entry.coverUrl]             Optional thumbnail (currently unused; for future visual toast)
 */
export function recordLastPlayed(entry) {
  if (!entry || !entry.kind || !entry.fileId || !entry.title) return;
  const key = _key(KEY_LAST_BASE);
  if (!key) return;  // no user yet — don't write to a global key
  try {
    localStorage.setItem(key, JSON.stringify({
      kind: entry.kind,
      fileId: entry.fileId,
      title: entry.title,
      subtitle: entry.subtitle || '',
      coverUrl: entry.coverUrl || '',
      t: Date.now(),
    }));
  } catch { /* private mode */ }
}

export function getLastPlayed() {
  const key = _key(KEY_LAST_BASE);
  if (!key) return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || !parsed.kind || !parsed.fileId || !parsed.title) return null;
    if (Date.now() - (parsed.t || 0) > FRESHNESS_MS) return null;
    return parsed;
  } catch { return null; }
}

export function clearLastPlayed() {
  const key = _key(KEY_LAST_BASE);
  if (!key) return;
  try { localStorage.removeItem(key); } catch { /* */ }
}

export function isPromptDismissed() {
  const key = _key(KEY_DISMISSED_BASE);
  if (!key) return false;
  try { return localStorage.getItem(key) === '1'; }
  catch { return false; }
}

export function markPromptDismissed() {
  const key = _key(KEY_DISMISSED_BASE);
  if (!key) return;
  try { localStorage.setItem(key, '1'); } catch { /* private mode */ }
}

/**
 * Show the resume toast — at most once per page load, only if there's
 * a fresh last-played entry, only if the user hasn't permanently
 * dismissed it. Idempotent: subsequent calls in the same page load
 * after the toast was shown are no-ops.
 */
let _offered = false;
export async function offerMediaResume() {
  if (_offered) return;
  _offered = true;
  if (isPromptDismissed()) return;
  const last = getLastPlayed();
  if (!last) return;
  // Dedupe with grove-resume's parallel toast. Both run from
  // app boot and both call showToast('Resume X?', …); when two
  // are visible at once the user can't tell which Play button
  // matches which title. Whichever record is newer wins this
  // page-load; the other module's _maybeOfferResume will see
  // our record is newer and bow out.
  const groveLast = _groveGetLastPlayed();
  if (groveLast && (groveLast.t || 0) > (last.t || 0)) return;

  // Lazy-load the resume action so this module stays tiny and the
  // expensive imports (media-player.js, files panel events) only
  // happen if the user actually clicks Play.
  const resume = async () => {
    if (last.kind === 'audio') {
      const mp = await import('./media-player.js');
      await mp.play(last.fileId);
    } else if (last.kind === 'video') {
      // Files panel listens for this event and opens the video.
      window.dispatchEvent(new CustomEvent('files:open-and-play', {
        detail: { fileId: last.fileId },
      }));
    }
  };

  const description = last.subtitle
    ? `${last.kind === 'audio' ? 'Audio' : 'Video'} · ${last.subtitle}`
    : (last.kind === 'audio' ? 'Audio' : 'Video');

  const id = showToast(`Resume ${last.title}?`, 'info', 0, {
    description,
    action: { label: 'Play', onClick: resume },
    dismissible: true,
  });
  // Attach permanent-dismiss to the toast X — same pattern as
  // grove-resume.js. requestAnimationFrame so the toast is in the DOM
  // by the time we query for it.
  requestAnimationFrame(() => {
    const el = document.querySelector(`.toast[data-id="${id}"]`);
    el?.querySelector('.toast-dismiss')?.addEventListener('click', markPromptDismissed);
  });
}
