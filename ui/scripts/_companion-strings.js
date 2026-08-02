/**
 * _companion-strings.js — persona-agnostic UI chrome strings for the
 * companion surfaces.
 *
 * Augmentum is open-source: every deployment configures its own companion
 * (name, voice, pronouns) through the identity / genesis layer. SHIPPED UI
 * labels must therefore never bake in a specific name or pronoun
 * ("she/her/Becca") — they say "Companion" / "your companion" / "it". The
 * persona lives only in prompts and spoken/generated lines, which stay
 * per-instance. (See the persona-agnostic-UI rule.)
 *
 * This module centralizes the chrome strings reused across more than one
 * companion surface so the wording is defined once. One-off labels are
 * written persona-agnostic inline in their own module.
 */

/** How shipped chrome refers to the companion. */
export const COMPANION_NOUN = 'your companion';

export const COMPANION_STRINGS = {
  // The "watch-list" call-to-action shows up on the notes-drawer gear, the
  // first-open coachmark, and the empty-state button.
  watchListCta: 'Tell your companion what to watch',
  // Notes drawer subtitle under the "Notes" title.
  notesSubtitle: 'what your companion has been noticing for you',
};
