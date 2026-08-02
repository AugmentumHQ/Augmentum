/**
 * Shared partner-launch path.
 *
 * The language-learning partner is materialised by GET /api/learning/
 * partner?lang=X and opened in narrative chat. Two surfaces drive this
 * today — the games hub chip and the homepage re-entry card — so the
 * launch logic lives here and is imported from both. Keeps the partner
 * id resolution + card shape + narrative-mode hand-off in one place.
 *
 * Lifted from hub.js (Phase 3, May 2026); see [[project-language-partner]].
 */

import { setMode } from '../app.js';

// English labels for the lang codes the partner system covers. Used
// in chip/card microcopy ("a persistent Japanese conversation tutor…");
// the partner's own greeting + system prompt are in the target
// language. Lang codes not in the seed table fall back to the raw
// code so surfaces still read sensibly. Keep in sync with
// `_SEEDS` in augmentum/learning/partners.py.
export const LANG_LABELS = {
  en: 'English', es: 'Spanish',     ja: 'Japanese',
  fr: 'French',  de: 'German',       it: 'Italian',
  pt: 'Portuguese', ko: 'Korean',    zh: 'Chinese',
};

export function langLabel(lang) {
  if (!lang) return 'target-language';
  return LANG_LABELS[lang] || lang.toUpperCase();
}

// Cheap "is there a bundled partner for this lang?" probe. Returns
// false for unsupported lang codes so callers can hide the surface
// rather than letting a 404 leak through on click. Doesn't materialise
// the card — that happens on actual click via openLanguagePartner.
export function isPartnerSupported(lang) {
  if (!lang) return false;
  return Boolean(LANG_LABELS[lang]);
}

export async function openLanguagePartner(lang) {
  // 1. Materialise / fetch the partner card on the server. INSERT OR
  //    IGNORE on the deterministic id makes this idempotent and
  //    race-safe (see migration 171 partial UNIQUE index).
  const r = await fetch(`/api/learning/partner?lang=${encodeURIComponent(lang)}`);
  if (!r.ok) throw new Error(`partner endpoint ${r.status}`);
  const partner = await r.json();
  if (!partner || !partner.id) throw new Error('partner response missing id');

  // 2. Switch to narrative mode. Lazy-import the narrative module so
  //    callers that don't otherwise need narrative (the homepage card
  //    on a passthrough session) don't pull in the whole bundle on
  //    first paint.
  try { setMode('narrative'); }
  catch (err) { console.warn('[partner] setMode failed', err); }

  const nav = await import('../narrative/index.js');

  // 3. Open chat with the server-returned card directly. Avoids a
  //    race between fetch-and-find: even if narrative's in-memory
  //    character list is stale (we just triggered the first POST),
  //    startChatWithCharacter only reads from the card we pass.
  const card = {
    id: partner.id,
    name: partner.name,
    ...(partner.data || {}),
    avatar: partner.avatar || '',
  };

  if (!nav.narrative?.startChatWithCharacter) {
    throw new Error('narrative module missing startChatWithCharacter');
  }
  nav.narrative.startChatWithCharacter(card);
}
