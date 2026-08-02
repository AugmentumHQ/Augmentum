/**
 * cast-button.js — shared factory for "Cast to TV" buttons.
 *
 * Every content surface that can send something to a paired TV (browse
 * video, comic reader, image viewer, chat attachments, library rows,
 * etc.) mounts one of these. Folding the boilerplate into one place
 * means:
 *
 *   - One visual identity (icon, hover behaviour, size variants)
 *   - One change site if the cast handshake ever evolves
 *   - One consistent a11y story
 *
 * The button calls `getContent()` AT CLICK TIME — not at construction
 * — so callers can return live state (current playback position,
 * current page, currently-rendered image id) without re-mounting.
 */

import { openCastPicker } from './cast-picker.js';


const CAST_ICON_SVG = `
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M2 16.1A5 5 0 0 1 5.9 20"/>
    <path d="M2 12.05A9 9 0 0 1 9.95 20"/>
    <path d="M2 8V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6"/>
    <line x1="2" y1="20" x2="2.01" y2="20"/>
  </svg>
`;


/**
 * Mount a Cast-to-TV button.
 *
 * @param {Object} opts
 * @param {string} opts.capability — capability id (e.g. 'media.video_play@1',
 *   'media.audio_play@1', 'display.image_show@1'). Filters the picker
 *   to receivers that advertise this capability.
 * @param {Function} opts.getContent — () => content | null. Called at
 *   click time. Return the shape cast-picker expects:
 *     { contentUrl, title, posterUrl?, author?, artist?, startTimeS?,
 *       fileId?, contentKey?, contentType?, metadata? }
 *   Return null to no-op the click (e.g. content not ready yet).
 * @param {string} [opts.className] — extra class names to merge in.
 *   The base is always `cast-btn cast-btn-{size}`.
 * @param {string} [opts.label] — optional text shown after the icon.
 * @param {string} [opts.title] — tooltip + a11y label. Default 'Cast to TV'.
 * @param {string} [opts.ariaLabel] — explicit a11y override; defaults
 *   to `title`.
 * @param {'sm'|'md'|'lg'} [opts.size] — padding + icon-size preset.
 *   Default 'md'.
 * @param {HTMLElement|null} [opts.anchor] — picker anchor override.
 *   Default: the button itself (so the popover positions against it).
 * @param {Function} [opts.onCast] — fired after successful cast dispatch.
 * @param {Function} [opts.onError] — fired on cast failure.
 *
 * @returns {HTMLButtonElement} ready to insert into a toolbar.
 */
export function mountCastButton({
  capability,
  getContent,
  className = '',
  label = '',
  title = 'Cast to TV',
  ariaLabel = '',
  size = 'md',
  anchor = null,
  onCast = null,
  onError = null,
} = {}) {
  if (!capability || typeof getContent !== 'function') {
    throw new Error('[cast-button] capability + getContent are required');
  }
  const btn = document.createElement('button');
  btn.type = 'button';
  const classes = ['cast-btn', `cast-btn-${size}`];
  if (className) classes.push(className);
  btn.className = classes.join(' ');
  btn.title = title;
  btn.setAttribute('aria-label', ariaLabel || title);
  btn.innerHTML = `
    <span class="cast-btn-icon" aria-hidden="true">${CAST_ICON_SVG}</span>
    ${label ? `<span class="cast-btn-label">${label}</span>` : ''}
  `;
  btn.addEventListener('click', (e) => {
    // stopPropagation: many call sites place this button INSIDE a
    // larger clickable region (a tile, an attachment card). The tap
    // is a cast intent, not a navigate-to-content intent.
    e.stopPropagation();
    e.preventDefault();
    let content;
    try {
      content = getContent();
    } catch (err) {
      console.warn('[cast-button] getContent threw', err);
      return;
    }
    if (!content) return;  // caller opted out (content not ready)
    openCastPicker({
      anchor: anchor || btn,
      capability,
      content,
      onCast,
      onError,
    });
  });
  return btn;
}
