/* ==========================================================================
   Toolbar control — Auto-read (TTS) button

   Toggles `settings.voiceAutoRead`. Persists to server via /api/config/ui so
   the preference survives across browsers/devices. Special behavior: while a
   TTS message is actively streaming (button has `.tts-streaming`), a click
   cancels playback without changing the preference — matches voice-chat
   stop-button intuition (pulsing icon = pressable to silence).

   Step 2 of the surface-owned composer migration. `surface` accepted but
   unused; Step 3 will route the setting read through it.
   ========================================================================== */

import { getSettings, save as saveSettings } from '../../settings.js';
import { showToast } from '../../app.js';
import { ttsStopCurrent, ttsChatWarmup } from '../tts.js';
import { flashToolbarBtn, tbFind } from './util.js';

/**
 * Wire the auto-read button inside the given toolbar root.
 *
 * @param {HTMLElement|null} toolbarEl  Composer toolbar root.
 * @param {object|null}      surface    Owning surface (unused in Step 2).
 */
export function wireAutoRead(toolbarEl, surface) {
  const btn = tbFind(toolbarEl, 'auto-read-btn');
  if (!btn) return;

  btn.addEventListener('click', () => {
    if (btn.classList.contains('tts-streaming')) {
      ttsStopCurrent();
      showToast('Auto-read paused for this message', 'info');
      return;
    }
    const s = getSettings();
    s.voiceAutoRead = !s.voiceAutoRead;
    btn.dataset.active = s.voiceAutoRead ? 'true' : 'false';
    flashToolbarBtn(btn);
    saveSettings();
    fetch('/api/config/ui', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ voiceAutoRead: s.voiceAutoRead }),
    }).catch(() => { /* best effort — local state already updated */ });
    showToast(s.voiceAutoRead ? 'Auto-read enabled' : 'Auto-read disabled', 'info');
    if (s.voiceAutoRead) ttsChatWarmup();
  });
}
