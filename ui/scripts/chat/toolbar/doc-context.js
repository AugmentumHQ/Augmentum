/* ==========================================================================
   Toolbar control — Knowledge context bar (per-surface clone path)

   The bindings themselves were ALWAYS per-session server-side
   (/api/documents/session/{id} + session_knowledge_packs), and every tab's
   stream sends its own X-Augmentum-Session — so retrieval already worked on
   secondary tabs. What was missing was the UI: the pills bar was a
   singleton in the primary's toolbar, so a secondary tab could neither see
   nor edit its own session's packs/docs.

   This wires a clone's bar against the OWNING SURFACE's sessionId via
   app.js::renderDocContextBarInto (the parameterized extraction of the
   singleton renderer). The floating picker is body-mounted and anchored by
   getBoundingClientRect, so it works from any tab's bar; attaches re-render
   through the onChange closure instead of the singleton refresh.
   ========================================================================== */

import { renderDocContextBarInto } from '../../app.js';
import { tbFind } from './util.js';

/**
 * Wire the knowledge context bar inside a per-surface toolbar clone.
 * Returns a cleanup fn (removes the session-changed listener).
 *
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root (clone).
 * @param {object|null}      surface    Owning surface — its _sessionId and
 *                                      mode drive what the bar shows.
 */
export function wireDocContext(toolbarEl, surface) {
  const bar = tbFind(toolbarEl, 'doc-context-bar');
  if (!bar || !surface) return undefined;

  const refresh = () => {
    const mode = surface._mode || surface.getContext?.().mode || 'passthrough';
    const sessionId = surface._sessionId || null;
    if (!sessionId) {
      // Surface hasn't created its session yet (mount ordering) — stay
      // hidden; the session-changed listener below catches up.
      bar.classList.add('hidden');
      return;
    }
    renderDocContextBarInto(bar, sessionId, mode, refresh);
  };
  refresh();

  // The tab's session can flip in place (left-panel session click while
  // this tab is focused) — re-render for the new session. Gated on
  // isActive so background tabs don't re-render for other tabs' flips;
  // they repaint on their own activate() → session promotion anyway.
  const onSession = () => { if (surface.isActive) refresh(); };
  document.addEventListener('augmentum:session-changed', onSession);
  return () => document.removeEventListener('augmentum:session-changed', onSession);
}
