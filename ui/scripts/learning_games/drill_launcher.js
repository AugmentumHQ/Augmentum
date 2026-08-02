/**
 * Drill launcher — bridges the chat-side `learning:launch_drill` event
 * to the game launchers in this folder.
 *
 * The language-learning partner emits a `suggest_drill` tool call
 * whose card carries `{game_id, lang, focus_words, reason}`. The chat
 * tool-card renderer dispatches `learning:launch_drill` with that
 * payload; this module resolves the game by id and opens it with the
 * focus-words bias forwarded.
 *
 * Game launcher signature evolution: launchers in this folder used to
 * be `launchX({lang, voice})`. We extend them with an optional
 * `focusWords` array. Each game decides what to do with it — drill-
 * focused games (Bubble Pop, Whisper Race, Echo Chamber) bias their
 * round toward those words via `fetchGamePool(..., focus_words: [...])`.
 * Games that don't care silently ignore the extra prop.
 *
 * Kept separate from hub.js so the chat surface can lazy-load only
 * this module — not the whole games hub — when the first drill chip
 * is tapped in a chat session.
 */

import { resolveVoiceForLang } from './_common.js';

// game_id → lazy-import of its launcher. Synced with the SuggestDrill
// enum in augmentum/tools/language_partner.py and the GAMES table in
// hub.js. Adding a new drillable game requires three updates: this
// dispatcher, the tool enum, and the hub.
const _DISPATCH = {
  bubble_pop:    () => import('./bubble_pop.js').then(m => m.launchBubblePop),
  whisper_race:  () => import('./whisper_race.js').then(m => m.launchWhisperRace),
  echo_chamber:  () => import('./echo_chamber.js').then(m => m.launchEchoChamber),
  mirror:        () => import('./mirror.js').then(m => m.launchMirror),
  constellation: () => import('./constellation.js').then(m => m.launchConstellation),
  word_forge:    () => import('./word_forge.js').then(m => m.launchWordForge),
  vocab_quest:   () => import('./vocab_quest.js').then(m => m.launchVocabQuest),
  story_weaver:  () => import('./story_weaver.js').then(m => m.launchStoryWeaver),
  word_garden:   () => import('./word_garden.js').then(m => m.launchWordGarden),
};

export async function launchDrill({ game_id, lang, focus_words } = {}) {
  if (!game_id || !lang) return;
  const loader = _DISPATCH[game_id];
  if (!loader) {
    console.warn('[drill] unknown game_id', game_id);
    return;
  }
  // Resolve a language-matched voice in parallel with the launcher
  // import — same path the hub uses, so audio in the drill round
  // matches the partner conversation it was prescribed from.
  const [launcher, voice] = await Promise.all([
    loader(),
    resolveVoiceForLang(lang, null),
  ]);
  if (typeof launcher !== 'function') {
    console.warn('[drill] launcher import returned no function for', game_id);
    return;
  }
  await launcher({
    lang,
    voice,
    focusWords: Array.isArray(focus_words) ? focus_words.slice(0, 20) : [],
  });
}
