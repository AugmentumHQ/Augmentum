/**
 * Grove resume — remember the last thing the user was listening to across
 * refreshes, and offer a one-tap "continue?" prompt on next load.
 *
 * Scope: grove radio stations + grove ambient YouTube orb. Audiobook resume
 * is handled separately by the media-player via server-side progress.
 *
 * UX contract:
 *   - Prompt appears at most once per page load.
 *   - "Play" action starts playback for whichever source was last active;
 *     it does NOT mark the prompt as dismissed (user wants music — next
 *     refresh should offer the same resume).
 *   - Dismissing via the toast X is permanent: we never prompt again until
 *     the user clears site data. This matches the user's explicit ask.
 */

import { getCurrentUser } from './auth.js';

// Per-user storage keys — every read/write must include the active
// user's id, otherwise Profile A's "currently playing" radio station
// surfaces under Profile B on the next reload. `_key()` returns null
// when no user is known yet; callers treat null as "skip" rather than
// falling back to a global key (the old bug).
const KEY_LAST_BASE = 'augmentum-grove-last-played';
const KEY_DISMISSED_BASE = 'augmentum-grove-resume-dismissed';

function _key(base) {
  const u = getCurrentUser();
  return u && u.id ? `${base}::u:${u.id}` : null;
}

/**
 * Record that this source was just played. Last writer wins — whichever
 * source was started most recently is what we offer on next refresh.
 *
 * @param {object} entry
 * @param {'radio' | 'ambient'} entry.type
 * @param {string} entry.name   user-facing label for the toast ("Chillhop Radio")
 */
export function recordLastPlayed(entry) {
  if (!entry || !entry.type || !entry.name) return;
  const key = _key(KEY_LAST_BASE);
  if (!key) return;
  try {
    localStorage.setItem(key, JSON.stringify({
      type: entry.type,
      name: entry.name,
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
    if (!parsed || !parsed.type || !parsed.name) return null;
    return parsed;
  } catch { return null; }
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
