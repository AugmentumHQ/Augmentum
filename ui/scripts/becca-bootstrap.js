/**
 * becca-bootstrap.js
 *
 * Reads /api/config/tools on page load to discover whether
 * companion_persona_mode is on. When it is, mounts the
 * becca-presence widget. When it isn't, this module is silent and
 * has no DOM impact.
 *
 * Becca is observer-only: she watches system state via the presence
 * bus and acts via her own direct-address surface, never by
 * intercepting the user's chat turn. Persona mode controls only
 * whether her widget is mounted.
 *
 * Also exposes:
 *   window.__beccaPersonaMode      bool — widget mount gate
 *   window.__beccaSettings         dict — full setting block for the UI
 *   window.__beccaRefreshFromBackend() — re-fetch settings (called after
 *                                        the settings panel saves)
 */

const SETTING_KEYS = [
  'companion_persona_mode',
  'companion_runtime_enabled',
  'companion_auto_summon',
  'companion_presence_mode',
  'companion_care_cadence',
  'companion_locale',
  'companion_audio_cues',
  'companion_keyboard_shortcuts',
  'companion_notify_eod',
  'companion_notify_drift_audit_push',
  'companion_cooldown_minutes',
  'companion_quiet_hours_start',
  'companion_quiet_hours_end',
  'companion_discreet_auto_exit_minutes',
  'companion_discreet_location_aware',
  // Architect / always-listening activation mode. Drives whether the
  // widget mounts the wake-word session or holds the PTT session
  // continuously open with server-side address classification.
  'companion_activation_mode',
  // Live-camera ("eye") capability gate — drives whether the presence
  // widget shows the eye and the call surface shows the camera button.
  'companion_live_vision_enabled',
  // Voice decision HUD — opt-in overlay showing the per-turn routing
  // verdict (act/converse/idle/drop). The widget reads this to decide
  // whether to render the HUD; the subtle status-row tell is always on.
  'companion_voice_decision_hud',
];

let _mounted = false;
// Tracks whether the last _refreshFromBackend pass got real data. When
// the page loads against an unreachable server (cold boot, restart
// mid-reload) the initial fetch returns null and the widget never
// mounts AND the pip never appears — the page sits in a stranded
// state until the next manual reload. Set on success, cleared on
// failure; consulted by retry + reconnect hooks below.
let _lastFetchOk = false;

async function _fetchSettings() {
  try {
    const resp = await fetch('/api/config/tools', { credentials: 'same-origin' });
    if (!resp.ok) return null;
    return await resp.json();
  } catch (_) {
    return null;
  }
}

async function _refreshFromBackend() {
  const cfg = await _fetchSettings();
  if (!cfg) { _lastFetchOk = false; return; }
  _lastFetchOk = true;
  const settings = {};
  for (const k of SETTING_KEYS) {
    if (cfg[k] !== undefined) settings[k] = cfg[k];
  }
  window.__beccaSettings = settings;
  window.__beccaPersonaMode = !!settings.companion_persona_mode
    && !!settings.companion_runtime_enabled;
  // Default true so an upgrade preserves the original always-on
  // behavior; explicit false opts into manual summon via the
  // header-logo affordance.
  window.__beccaAutoSummon = settings.companion_auto_summon !== false;
  // Aletheia × Augmentum — sync presence_mode to localStorage so the
  // companion-notes module reads the live value. Backend is source of
  // truth; localStorage is just the fast-path cache the JS module
  // checks on every render. Dispatching ``companion:presence-mode-changed``
  // lets the drawer update its dot visibility immediately instead of
  // waiting up to 60s for its next poll tick.
  let presenceMode = 'silent';
  let presenceModeChanged = false;
  try {
    const mode = (settings.companion_presence_mode || 'silent').toLowerCase();
    const valid = (mode === 'silent' || mode === 'gentle' || mode === 'engaged') ? mode : 'silent';
    const prev = window.__companionPresenceMode;
    presenceMode = valid;
    presenceModeChanged = prev !== valid;
    localStorage.setItem('companion.presence.mode', valid);
    window.__companionPresenceMode = valid;
  } catch (_) { /* private browsing / quota — global flag still set, mount proceeds */ }
  if (presenceModeChanged) {
    try {
      window.dispatchEvent(new CustomEvent('companion:presence-mode-changed', {
        detail: { mode: presenceMode },
      }));
    } catch (_) { /* listener throws are non-fatal; mount still proceeds below */ }
  }
  // Generic "settings re-fetched" signal — lets a mounted widget re-read
  // capability gates (e.g. the live-camera eye) the moment the settings
  // panel saves, instead of waiting for the next VRM activation / reload.
  try {
    window.dispatchEvent(new CustomEvent('becca:settings-refreshed', {
      detail: { settings: window.__beccaSettings },
    }));
  } catch (_) { /* non-fatal */ }
  await _applyMountState();
}

async function _applyMountState() {
  const personaOn = window.__beccaPersonaMode === true;
  const autoSummon = window.__beccaAutoSummon !== false;
  const shouldMount = personaOn && autoSummon;
  // Sync local _mounted flag from actual DOM state. The user can
  // dismiss the widget directly via becca-presence.js's dismiss
  // button — that unmounts the DOM but doesn't notify us. Without
  // this re-read, subsequent _refreshFromBackend passes see a stale
  // _mounted=true and skip the remount branch, then fall through and
  // remove the pip — stranding the page with neither widget nor pip.
  _mounted = !!document.querySelector('.becca-presence');
  // Likewise: if the user dismissed the widget for this tab, we
  // shouldn't auto-resummon on a settings refresh / reconnect — the
  // dismiss is "for this tab only" by design (aria-label says so).
  // Without this gate the body.becca-dismissed flag gets clobbered
  // by the mount path and the user's local intent evaporates.
  const dismissedLocally = document.body.classList.contains('becca-dismissed');

  // Aletheia × Augmentum arc — note pip mounts alongside the avatar widget.
  // Visibility is gated by per-user presence_mode (silent suppresses
  // internally). Always mount when the runtime is enabled so the dot is
  // ready to appear the moment a note exists; the JS module reads
  // presence_mode at render time.
  const runtimeOn = (window.__beccaSettings || {}).companion_runtime_enabled !== false
    && personaOn;
  if (runtimeOn) {
    try {
      const notesMod = await import('./companion-notes.js');
      if (notesMod && notesMod.CompanionNotes) {
        notesMod.CompanionNotes.mount();
      }
    } catch (e) {
      console.warn('[companion-notes] mount failed', e);
    }
  }

  if (shouldMount && !_mounted && !dismissedLocally) {
    try {
      const mod = await import('./becca-presence.js');
      mod.mountBeccaPresence();
      _mounted = true;
      console.info('[becca] persona mode active — widget mounted');
    } catch (e) {
      console.warn('[becca] mount failed', e);
    }
  } else if (!personaOn && _mounted) {
    // Scoped to the master persona toggle, NOT `!shouldMount`. The
    // previous `!shouldMount` test conflated "persona off" with "auto-
    // summon off" — so when auto_summon was false and the user had
    // manually summoned via the pip / header-logo, every subsequent
    // visibilitychange / window.focus / window.online reconcile would
    // see `shouldMount = personaOn && autoSummon = false`, hit this
    // branch, and unmount her mid-session. Symptom: widget disappears
    // on every page state change, summon pip reappears. (Diagnosed
    // 2026-05-25.) Auto-summon governs *initial* load behavior only;
    // a manually-mounted widget should persist until persona itself
    // is toggled off.
    try {
      const mod = await import('./becca-presence.js');
      mod.unmountBeccaPresence();
      _mounted = false;
    } catch (_) { /* dynamic import or unmount failed — mounted-state flag stays true */ }
  }

  // Persona off: hide the summon button regardless of mount state —
  // there's nothing to summon back to.
  if (!personaOn) {
    document.getElementById('becca-summon-btn')?.classList.add('hidden');
  }

  // Persona on + auto-summon off: don't mount, but mark the body as
  // dismissed AND drop the summon pip so the user has a visible
  // affordance (the header-logo summon is also live for power users —
  // see _attachHeaderLogoSummon — but the pip is the discoverable one).
  if (personaOn && !autoSummon && !_mounted) {
    document.body.classList.add('becca-dismissed');
    try {
      const mod = await import('./becca-presence.js');
      mod.ensureSummonPip();
    } catch (e) {
      console.warn('[becca] summon pip mount failed', e);
    }
  }

  // Persona on + auto-summon flipped back ON after being off: hide the
  // summon button if it's still showing. The mount itself happens above
  // when shouldMount becomes true. EXCEPTION: if the user dismissed
  // locally, leave the button — they want it.
  if (personaOn && autoSummon && !dismissedLocally) {
    document.getElementById('becca-summon-btn')?.classList.add('hidden');
  }
}

// Expose refresh hook so settings.js can call it after save.
window.__beccaRefreshFromBackend = _refreshFromBackend;

// ── Rebuild / delete handlers (settings tab) ─────────────────────

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

async function _confirmAndPost(url, body, confirmTitle, confirmMsg) {
  if (!window.confirm(`${confirmTitle}\n\n${confirmMsg}`)) return null;
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body),
    });
    return await r.json().catch(() => ({ ok: false, reason: 'parse_error' }));
  } catch (e) {
    return { ok: false, reason: String(e) };
  }
}

function _showResult(label, result) {
  if (!result) return;
  const ok = !!result.ok;
  const msg = ok
    ? `${label}: done.`
    : `${label}: ${result.reason || 'failed'}.`;
  try {
    const evt = new CustomEvent('augmentum:toast',
                                { detail: { msg, kind: ok ? 'success' : 'error' } });
    window.dispatchEvent(evt);
  } catch (_) { /* no toast handler — alert() below ensures the user sees the result */ }
  alert(msg);
}

function _attachRebuildHandlers() {
  const soft = document.getElementById('becca-rebuild-soft-btn');
  const hard = document.getElementById('becca-rebuild-hard-btn');
  const del = document.getElementById('becca-delete-all-btn');

  if (soft && !soft._beccaBound) {
    soft._beccaBound = true;
    soft.addEventListener('click', async () => {
      const user_id = await _currentUserId();
      if (!user_id) { alert('No active user — can\'t identify the relationship to rebuild.'); return; }
      const res = await _confirmAndPost(
        '/api/companion/rebuild',
        { user_id, kind: 'soft', user_signal: 'settings_panel' },
        'Something changed?',
        'This wipes Becca\'s affect baselines and her graduated noticings about you. ' +
        'She keeps factual memories and her side of the relationship. She\'ll re-tune ' +
        'how she reads you over the next several conversations. Continue?',
      );
      _showResult('Soft reset', res);
    });
  }

  if (hard && !hard._beccaBound) {
    hard._beccaBound = true;
    hard.addEventListener('click', async () => {
      const user_id = await _currentUserId();
      if (!user_id) { alert('No active user.'); return; }
      const res = await _confirmAndPost(
        '/api/companion/rebuild',
        { user_id, kind: 'hard_reset', user_signal: 'settings_panel' },
        'Start over with her?',
        'This wipes baselines, noticings, AND factual memories about you. The ' +
        'relationship continues but from a clean slate. Continue?',
      );
      _showResult('Hard reset', res);
    });
  }

  if (del && !del._beccaBound) {
    del._beccaBound = true;
    del.addEventListener('click', async () => {
      const user_id = await _currentUserId();
      if (!user_id) { alert('No active user.'); return; }
      const res = await _confirmAndPost(
        '/api/companion/delete_all',
        { user_id, confirm: true },
        'Delete everything Becca knows about you?',
        'This is a hard delete cascade. The next time you talk to her, she will ' +
        'not know you. Nothing will be retained. This cannot be undone.',
      );
      _showResult('Delete', res);
    });
  }
}

// Re-attach handlers when the settings panel opens (DOM elements are
// created lazily by settings.js).
const _observer = new MutationObserver(() => _attachRebuildHandlers());

// Header-logo summon. Clicking the orbiting logo brings Becca back
// if she's been dismissed (body.becca-dismissed is set by the widget
// at dismiss time). Gated by the body class so the logo doesn't feel
// clickable while she's already mounted — and unconditionally
// reflects persona-mode state: if the user has the setting OFF we
// silently no-op rather than fighting their settings.
function _attachHeaderLogoSummon() {
  const logo = document.querySelector('.header-logo');
  if (!logo || logo._beccaSummonBound) return;
  logo._beccaSummonBound = true;
  logo.addEventListener('click', async () => {
    if (!document.body.classList.contains('becca-dismissed')) return;
    if (window.__beccaPersonaMode !== true) return;
    try {
      document.getElementById('becca-summon-btn')?.classList.add('hidden');
      const mod = await import('./becca-presence.js');
      mod.mountBeccaPresence();
    } catch (e) {
      console.warn('[becca] header-logo summon failed', e);
    }
  });
}

// Reconciliation hook. Re-fetches settings and re-applies mount state
// when the page transitions from offline → online (server restart,
// disconnected wifi, sleep wake) so a stranded summon-pip from before
// the disconnect gets reconciled with whatever the server says now.
// Cheap — one /api/config/tools call. Throttled to once per 4s so a
// burst of visibility/online events from the OS doesn't spam fetches.
let _reconcileLast = 0;
function _scheduleReconcile() {
  const now = Date.now();
  if (now - _reconcileLast < 4000) return;
  _reconcileLast = now;
  _refreshFromBackend().catch(() => {});
}

window.addEventListener('DOMContentLoaded', () => {
  _refreshFromBackend()
    .then(() => {
      // If the very first attempt landed against a 502 (server still
      // booting), the page would otherwise sit forever with no widget
      // and no pip. Retry a few times on a short ramp so a cold boot
      // recovers without the user reloading.
      if (_lastFetchOk) return;
      const delays = [800, 1600, 3200, 6400];
      let i = 0;
      const tick = () => {
        if (_lastFetchOk || i >= delays.length) return;
        setTimeout(() => {
          _refreshFromBackend().finally(tick);
          i += 1;
        }, delays[i]);
      };
      tick();
    })
    .catch(() => {});
  _attachHeaderLogoSummon();
  _observer.observe(document.body, { childList: true, subtree: true });

  // Tab-focus + online-event hooks so a server restart in the
  // background reconciles the moment the user comes back.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) _scheduleReconcile();
  });
  window.addEventListener('online', _scheduleReconcile);
  window.addEventListener('focus', _scheduleReconcile);
});
