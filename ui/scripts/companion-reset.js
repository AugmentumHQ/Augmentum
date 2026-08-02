/**
 * companion-reset.js — Reset gestures UI (Sprint 8 surface).
 *
 * Three reset cards (soft / hard / delete_all) with explicit
 * descriptions of what each wipes vs keeps. Each requires
 * confirmation. Hard + delete require typed confirmation.
 *
 * Wraps the existing recovery.py endpoints exposed via:
 *   POST /api/companion/rebuild      (soft + hard)
 *   POST /api/companion/delete_all
 *
 * Audit log surface shows past resets from companion_rebuild_log
 * (no endpoint yet — placeholder section pending Sprint 7+ wiring).
 */

import { installDialog } from './_focus-trap.js';

let _mounted = false;
let _dialog = null;

async function _currentUserId() {
  try {
    const r = await fetch('/api/auth/me', { credentials: 'same-origin' });
    if (r.ok) {
      const j = await r.json();
      return j.id || j.user_id || j.username || '';
    }
  } catch (_) { /* unauthenticated or network failure — caller treats empty as anon */ }
  return '';
}

async function _postJson(url, body) {
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body || {}),
    });
    const ok = r.ok;
    let data = null;
    try { data = await r.json(); } catch (_) {}
    return { ok, data };
  } catch (e) {
    return { ok: false, data: { error: String(e) } };
  }
}

async function _runSoft() {
  if (!window.confirm(
    "Soft reset\n\n" +
    "Wipes affect baselines, recent noticings, and 'about you' state.\n" +
    "KEEPS your factual conversation history and the things it's made.\n\n" +
    "Proceed?",
  )) return;
  const userId = await _currentUserId();
  const result = await _postJson('/api/companion/rebuild', {
    user_id: userId, kind: 'soft', reason: 'settings_panel',
  });
  alert(result.ok ? 'Soft reset complete.' : `Reset failed: ${JSON.stringify(result.data)}`);
}

async function _runHard() {
  const phrase = window.prompt(
    "Hard reset\n\n" +
    "Wipes everything it's learned about you: affect, observations,\n" +
    "noticings, factual memory, relationship state.\n" +
    "Identity reverts to genesis (the canonical personality doc).\n\n" +
    "Type RESET to confirm:",
  );
  if (phrase !== 'RESET') return;
  const userId = await _currentUserId();
  const result = await _postJson('/api/companion/rebuild', {
    user_id: userId, kind: 'hard_reset', reason: 'settings_panel',
  });
  alert(result.ok ? 'Hard reset complete.' : `Reset failed: ${JSON.stringify(result.data)}`);
}

async function _runDeleteAll() {
  const phrase = window.prompt(
    "Delete all companion data\n\n" +
    "Permanently removes ALL companion data for your account: identity,\n" +
    "memories, observations, creations, mutes. The runtime will\n" +
    "re-provision from the canonical genesis next time you interact.\n\n" +
    "This is the closest thing to 'start over from scratch'.\n\n" +
    "Type DELETE to confirm:",
  );
  if (phrase !== 'DELETE') return;
  const userId = await _currentUserId();
  const result = await _postJson('/api/companion/delete_all', {
    user_id: userId, reason: 'settings_panel',
  });
  alert(result.ok ? 'Companion data deleted.' : `Delete failed: ${JSON.stringify(result.data)}`);
}

function open() {
  if (_mounted) return;
  const overlay = document.createElement('div');
  overlay.id = 'companion-reset-overlay';
  overlay.className = 'companion-reset-overlay';
  overlay.innerHTML = `
    <div class="companion-reset-panel" role="dialog" aria-label="Reset gestures">
      <div class="companion-reset-header">
        <h3>Reset</h3>
        <button type="button" class="companion-reset-close" aria-label="Close">×</button>
      </div>
      <div class="companion-reset-body">
        <article class="companion-reset-card" data-kind="soft">
          <h4>Soft reset</h4>
          <p>
            Wipes affect baselines, recent noticings, and "about you" state.
            <strong>Keeps</strong> factual memory + things it's made.
            Useful after a hard week — clears the mood snapshot without
            losing the relationship.
          </p>
          <button type="button" class="companion-reset-button" data-action="soft">
            Run soft reset
          </button>
        </article>

        <article class="companion-reset-card" data-kind="hard">
          <h4>Hard reset</h4>
          <p>
            Wipes everything it's learned about you. Identity reverts to
            genesis. Your companion will meet you fresh — same essence, no
            memory of prior conversations.
          </p>
          <button type="button" class="companion-reset-button danger" data-action="hard">
            Run hard reset
          </button>
        </article>

        <article class="companion-reset-card danger" data-kind="delete">
          <h4>Delete all companion data</h4>
          <p>
            Permanent removal of all companion data for your account.
            The runtime will re-provision on next interaction. Closest
            thing to starting fresh.
          </p>
          <button type="button" class="companion-reset-button danger" data-action="delete">
            Delete everything
          </button>
        </article>

        <p class="companion-reset-footnote">
          All resets are local to your account. Other accounts on this
          installation are not affected.
        </p>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.querySelector('.companion-reset-close').addEventListener('click', close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });
  overlay.querySelector('[data-action="soft"]').addEventListener('click', _runSoft);
  overlay.querySelector('[data-action="hard"]').addEventListener('click', _runHard);
  overlay.querySelector('[data-action="delete"]').addEventListener('click', _runDeleteAll);
  // Keyboard a11y: Escape closes, Tab wraps within the panel, focus moves
  // in on open and restores on close. setAria:false — the panel already
  // declares role="dialog"/aria-label in its markup.
  const panel = overlay.querySelector('.companion-reset-panel');
  _dialog = installDialog(panel, {
    onClose: close,
    initialFocus: '.companion-reset-close',
    setAria: false,
  });
  _mounted = true;
}

function close() {
  if (!_mounted) return;
  _mounted = false;
  if (_dialog) { try { _dialog.release(); } catch (_) {} _dialog = null; }
  document.querySelector('#companion-reset-overlay')?.remove();
}

export const CompanionReset = { open, close };

if (typeof window !== 'undefined') {
  window.CompanionReset = CompanionReset;
}
