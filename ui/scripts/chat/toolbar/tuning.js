/* ==========================================================================
   Toolbar control — Per-chat sampling ("Tuning for this chat")

   Opens the shared sampling editor scoped to THIS conversation. The override
   is stored on the session as `session.chatSampling` and merged into the
   outgoing request `options` by chat/stream.js — where the backend apply-point
   (augmentum/models/sampling_profiles.py) treats it as the highest-precedence
   per-call layer, above the per-model profile and the family default.

   Blank field = inherit the next layer. The editor's placeholders show the
   active model's effective values so you can see what you're overriding.
   ========================================================================== */

import { sessionStore } from '../sessions.js';
import { openSamplingEditor } from '../../sampling-editor.js';
import { flashToolbarBtn, tbFind } from './util.js';

function hasOverride(session) {
  const o = session?.chatSampling;
  return !!(o && typeof o === 'object' && Object.keys(o).length > 0);
}

async function fetchModelSampling(model) {
  if (!model) return { effective: {}, recommended: {}, supported: null };
  try {
    const resp = await fetch(`/api/models/${encodeURIComponent(model)}/sampling`);
    if (!resp.ok) return { effective: {}, recommended: {}, supported: null };
    const d = await resp.json();
    // supported = which knobs this model's backend honors (null = show all).
    return {
      effective: d.effective || {},
      recommended: d.recommended || {},
      supported: Array.isArray(d.supported) ? d.supported : null,
    };
  } catch {
    return { effective: {}, recommended: {}, supported: null };  // editor still works; just no hints
  }
}

async function openChatTuningSheet(btn) {
  const id = sessionStore.getActiveId();
  const session = id ? sessionStore.get(id) : null;
  if (!session) return;
  const model = session.model || window.app?.state?.currentModel || '';
  const { effective, recommended, supported } = await fetchModelSampling(model);

  openSamplingEditor({
    scopeLabel: 'This chat',
    helpText: 'Overrides the per-model profile for this conversation only. '
      + 'Blank = inherit (per-model → family default).',
    values: (session.chatSampling && typeof session.chatSampling === 'object')
      ? session.chatSampling : {},
    effective,
    recommended,
    supported,
    onSave: (override) => {
      if (override && Object.keys(override).length > 0) session.chatSampling = override;
      else delete session.chatSampling;
      sessionStore.markDirty(id);
      sessionStore.save(id);
      if (btn) btn.dataset.active = hasOverride(session) ? 'true' : 'false';
    },
  });
}

/**
 * Wire the per-chat tuning button inside the given toolbar root.
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root.
 */
export function wireChatTuning(toolbarEl) {
  const btn = tbFind(toolbarEl, 'chat-tuning-btn');
  if (!btn) return undefined;
  const refresh = () => {
    const id = sessionStore.getActiveId();
    const session = id ? sessionStore.get(id) : null;
    btn.dataset.active = hasOverride(session) ? 'true' : 'false';
  };
  refresh();
  window.addEventListener('augmentum:session-changed', refresh);
  btn.addEventListener('click', () => { flashToolbarBtn(btn); openChatTuningSheet(btn); });
  // Cleanup for per-surface toolbar clones — without it the window listener
  // outlives a closed tab and keeps its detached button alive.
  return () => window.removeEventListener('augmentum:session-changed', refresh);
}
