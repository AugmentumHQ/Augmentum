/**
 * companion-voice-surface.js
 * Sprint 5 — minimal voice-summoned surface dispatch table.
 *
 * Matches a transcribed utterance to a surface request and routes
 * the request through the existing surface system (XR + flat). The
 * table is intentionally small; Sprint 6+ may grow it into a proper
 * intent classifier.
 *
 * Returns true if a match was found and dispatched; false otherwise
 * so the caller can fall back to normal chat handling.
 */

/**
 * Voice → surface dispatch is host-routed: the caller passes a
 * `dispatch(kind|action)` function so we don't hard-bind to any one
 * surface system. The XR side wires this into xr-surface-bridge,
 * the flat-page side may wire into the main app router. Keeping the
 * coupling here would force one module's API choice on the other.
 */

const RULES = [
  { rx: /\bshow (me )?(my )?(the )?journal\b/i,         kind: 'journal' },
  { rx: /\bshow (me )?(my )?(the )?files?\b/i,           kind: 'files' },
  { rx: /\bopen (a |the )?browser\b/i,                   kind: 'browser' },
  { rx: /\bdismiss\b|\bclose (this|that)\b/i,            kind: '_dismiss' },
  { rx: /\b(show|play) (me )?(a )?video\b/i,             kind: 'media:video' },
  { rx: /\bplay (some )?music\b/i,                       kind: 'media:audio' },
  { rx: /\bopen (the )?settings\b/i,                     kind: 'settings' },
  { rx: /\bopen (the )?dream(s)?\b/i,                    kind: 'dreams' },
];

/**
 * Match `utterance` against the surface rules. On match, call
 * `opts.dispatch({ action, kind })` with the chosen routing payload
 * and return `true`. On no match, return `false`.
 *
 * @param {string} utterance
 * @param {{
 *   dispatch?: (req: {action: 'open'|'dismiss', kind?: string}) => void,
 *   onDispatch?: (kind: string) => void,
 * }} [opts]
 * @returns {boolean}
 */
export function tryVoiceSummonedSurface(utterance, opts = {}) {
  if (!utterance || typeof utterance !== 'string') return false;
  const trimmed = utterance.trim();
  if (!trimmed) return false;
  const dispatch = typeof opts.dispatch === 'function' ? opts.dispatch : null;

  for (const rule of RULES) {
    if (rule.rx.test(trimmed)) {
      const kind = rule.kind;
      const req = (kind === '_dismiss')
        ? { action: 'dismiss' }
        : { action: 'open', kind };
      try {
        if (dispatch) dispatch(req);
        else {
          // Fall back to a CustomEvent so unbound callers still have
          // a path to wire in. Listener is responsible for routing.
          if (typeof window !== 'undefined' && window.dispatchEvent) {
            window.dispatchEvent(new CustomEvent('companion:voice-surface', { detail: req }));
          }
        }
        if (typeof opts.onDispatch === 'function') {
          try { opts.onDispatch(kind); } catch (_) { /* ignore */ }
        }
        return true;
      } catch (e) {
        console.warn('companion-voice-surface dispatch failed', e);
        return false;
      }
    }
  }
  return false;
}
