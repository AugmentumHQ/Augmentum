/* notifications/push-prompt.js — just-in-time push subscription prompt.
 *
 * When a user takes an action that benefits from out-of-tab
 * notifications (scheduling a briefing, setting a reminder, etc.),
 * we offer to turn on Web Push right there in the conversation. The
 * prompt:
 *
 *   * Only shows when supported + not already subscribed + not resolved.
 *   * Has three actions: Enable / Not now / Don't ask again.
 *   * Resolution persists via localStorage so we don't pester — set on
 *     "Don't ask again" AND on successful Enable. Once the user has
 *     answered "yes" on this client, we trust that answer even if the
 *     browser later loses its push subscription (cleared site data,
 *     SW evicted, private window, push-service hiccup). Settings →
 *     Notifications is the explicit re-engagement path.
 *   * "Not now" is per-call — same browsing session can still trigger
 *     a new prompt for a different action.
 *
 * Persistence is intentionally per-client (localStorage, not server
 * settings) because the push subscription itself is per-browser and
 * different browsers have different push restrictions. Treating the
 * answer as device-local matches what the substrate can actually act
 * on.
 *
 * The caller mounts the prompt into a host element; the module owns
 * its DOM lifecycle. Returns true if a prompt was actually mounted,
 * false otherwise (so callers can avoid layout churn when there's
 * nothing to show).
 */

import {
  detectOS,
  osLabel,
  renderPrimaryAction,
  renderFallbackInstructions,
} from './cert-trust.js';

// Legacy key from when this flag only tracked "Don't ask again".
// Kept for back-compat — users who dismissed pre-fix stay dismissed.
const DISMISS_KEY = 'augmentum.pushPromptDismissedAt';
// New key written on any resolution (dismissed OR successfully enabled).
const RESOLVED_KEY = 'augmentum.pushPromptResolvedAt';

export async function mountPushPromptIfNeeded(rootEl, opts = {}) {
  if (!rootEl || rootEl.querySelector(':scope > .push-prompt-card')) return false;
  if (_isResolved()) return false;

  let state;
  try {
    const { getPushState } = await import('./push-subscribe.js');
    state = await getPushState();
  } catch (_) {
    return false;
  }
  if (!state || !state.supported) return false;
  if (state.subscribed) return false;
  if (state.permission === 'denied') return false;  // can't unblock from JS

  const card = _buildCard(opts);
  rootEl.appendChild(card);
  return true;
}

export function isPushPromptDismissed() {
  return _isResolved();
}

export function clearPushPromptDismissal() {
  try {
    localStorage.removeItem(DISMISS_KEY);
    localStorage.removeItem(RESOLVED_KEY);
  } catch (_) { /* best-effort: localStorage may be unavailable/full */ }
}

// ── Internals ───────────────────────────────────────────────────

function _isResolved() {
  try {
    // Either key suppresses the prompt — RESOLVED is the canonical home
    // for new writes; DISMISS_KEY is honored for back-compat with users
    // who dismissed before this fix landed.
    return !!(localStorage.getItem(RESOLVED_KEY)
      || localStorage.getItem(DISMISS_KEY));
  } catch (_) {
    return false;
  }
}

function _markResolved() {
  try {
    localStorage.setItem(RESOLVED_KEY, String(Date.now()));
  } catch (_) { /* best-effort: localStorage may be unavailable/full */ }
}

function _buildCard(opts) {
  const card = document.createElement('div');
  card.className = 'push-prompt-card';
  // Inline styles so this works without depending on a new CSS file
  // landing in the same change. Subtle, not screaming.
  Object.assign(card.style, {
    marginTop: '8px',
    padding: '10px 12px',
    borderRadius: '8px',
    background: 'var(--surface-2, rgba(127, 127, 127, 0.08))',
    border: '1px solid var(--border-subtle, rgba(127, 127, 127, 0.2))',
    fontSize: 'var(--text-sm, 0.875rem)',
    lineHeight: '1.45',
  });

  const headline = opts.headline
    || 'Want this to reach you when the tab is closed?';
  const body = opts.body
    || 'Enabling browser notifications lets briefings and reminders '
       + 'buzz your device even when Augmentum isn’t open.';

  const titleEl = document.createElement('div');
  titleEl.textContent = headline;
  titleEl.style.fontWeight = '500';
  titleEl.style.marginBottom = '4px';
  card.appendChild(titleEl);

  const bodyEl = document.createElement('div');
  bodyEl.textContent = body;
  bodyEl.style.opacity = '0.85';
  bodyEl.style.marginBottom = '8px';
  card.appendChild(bodyEl);

  const row = document.createElement('div');
  row.style.display = 'flex';
  row.style.gap = '8px';
  row.style.flexWrap = 'wrap';

  const enableBtn = _btn('Enable', 'primary');
  const notNowBtn = _btn('Not now', 'ghost');
  const neverBtn = _btn("Don’t ask again", 'ghost');
  row.appendChild(enableBtn);
  row.appendChild(notNowBtn);
  row.appendChild(neverBtn);
  card.appendChild(row);

  const statusEl = document.createElement('div');
  statusEl.style.marginTop = '6px';
  statusEl.style.minHeight = '0';
  statusEl.style.fontSize = 'var(--text-xs, 0.75rem)';
  statusEl.style.opacity = '0.75';
  card.appendChild(statusEl);

  enableBtn.addEventListener('click', async () => {
    enableBtn.disabled = true;
    notNowBtn.disabled = true;
    neverBtn.disabled = true;
    enableBtn.textContent = 'Enabling…';
    statusEl.textContent = '';
    try {
      const mod = await import('./push-subscribe.js');
      await mod.enablePush({ channelPattern: '*', importanceFloor: 0 });
      _markResolved();
      statusEl.textContent = 'Subscribed. You’ll get notifications even when this tab is closed.';
      enableBtn.textContent = 'Enabled';
      setTimeout(() => { _fadeOut(card); }, 2500);
    } catch (err) {
      const msg = String(err?.message || err || '');
      if (_looksLikeCertError(msg)) {
        // SW registration failed because the cert isn't trusted —
        // route to the trust-flow panel which downloads the root CA
        // and gives per-OS install instructions.
        _mountTrustFlow(card);
        return;
      }
      if (_looksLikePushServiceError(msg) && await _isBrave()) {
        // SW registered (cert is fine) but pushManager.subscribe()
        // failed because Brave disables Google push services by
        // default for privacy. Surface the exact toggle to flip
        // rather than dropping the user into a generic error state.
        _mountBraveFix(card);
        return;
      }
      statusEl.textContent = _friendlyError(msg);
      enableBtn.disabled = false;
      notNowBtn.disabled = false;
      neverBtn.disabled = false;
      enableBtn.textContent = 'Try again';
    }
  });

  notNowBtn.addEventListener('click', () => {
    _fadeOut(card);
  });

  neverBtn.addEventListener('click', () => {
    _markResolved();
    _fadeOut(card);
  });

  return card;
}

function _btn(label, variant) {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = label;
  b.className = variant === 'primary'
    ? 'btn btn-primary btn-sm'
    : 'btn btn-secondary btn-sm';
  Object.assign(b.style, {
    fontSize: 'var(--text-xs, 0.75rem)',
    padding: '4px 10px',
  });
  if (variant === 'ghost') {
    b.style.background = 'transparent';
    b.style.opacity = '0.8';
  }
  return b;
}

function _looksLikePushServiceError(msg) {
  const m = msg.toLowerCase();
  return (
    m.includes('push service')
    || m.includes('registration failed')
    || m.includes('abort')  // some Chromium variants surface this
  );
}

async function _isBrave() {
  // navigator.brave.isBrave() is the canonical Brave detection —
  // returns a Promise<true> on real Brave, throws/missing elsewhere.
  if (navigator.brave && typeof navigator.brave.isBrave === 'function') {
    try {
      return !!(await navigator.brave.isBrave());
    } catch (_) {
      return false;
    }
  }
  return false;
}

function _mountBraveFix(rootEl) {
  while (rootEl.firstChild) rootEl.removeChild(rootEl.firstChild);

  const title = document.createElement('div');
  title.textContent = 'Brave blocks Google push services by default';
  title.style.fontWeight = '500';
  title.style.marginBottom = '6px';
  rootEl.appendChild(title);

  const body = document.createElement('div');
  body.style.opacity = '0.9';
  body.style.marginBottom = '10px';
  body.style.lineHeight = '1.45';
  body.textContent = (
    'Brave disables Web Push by default because it requires routing '
    + 'through Google’s FCM service. Flip one setting and restart '
    + 'Brave — this only needs to happen once per profile.'
  );
  rootEl.appendChild(body);

  // The brave:// URL can't be auto-opened by a regular page; copy the
  // URL to clipboard so the user can paste it into the address bar.
  const stepsWrap = document.createElement('div');
  stepsWrap.style.marginBottom = '10px';
  stepsWrap.innerHTML = `
    <div style="margin-bottom:8px">
      <strong>1.</strong> Paste this into Brave’s address bar:
    </div>
    <div id="brave-url-block"
         style="background:var(--surface-3, rgba(0,0,0,0.08));
                padding:8px 10px;border-radius:6px;
                font-family:monospace;
                font-size:var(--text-xs, 0.75rem);
                margin-bottom:8px;user-select:all">
      brave://settings/privacy
    </div>
    <div style="margin-bottom:4px">
      <strong>2.</strong> Find the toggle
      <em>“Use Google services for push messaging”</em> and turn it on.
    </div>
    <div style="margin-bottom:4px">
      <strong>3.</strong> Quit Brave completely (close all windows
      <em>and</em> the tray icon — Brave caches the setting per
      process).
    </div>
    <div>
      <strong>4.</strong> Reopen Brave, come back here, hit
      <em>“try again”</em>.
    </div>
  `;
  rootEl.appendChild(stepsWrap);

  const actionRow = document.createElement('div');
  actionRow.style.display = 'flex';
  actionRow.style.gap = '8px';
  actionRow.style.flexWrap = 'wrap';
  actionRow.style.marginBottom = '8px';

  const copyBtn = _btn('Copy URL', 'primary');
  copyBtn.addEventListener('click', async () => {
    const orig = copyBtn.textContent;
    try {
      await _copyToClipboard('brave://settings/privacy');
      copyBtn.textContent = 'Copied';
      setTimeout(() => { copyBtn.textContent = orig; }, 1500);
    } catch (_) {
      copyBtn.textContent = 'Copy failed';
      setTimeout(() => { copyBtn.textContent = orig; }, 2200);
    }
  });
  actionRow.appendChild(copyBtn);

  const retryBtn = _btn('I enabled it — try again', 'primary');
  const cancelBtn = _btn('Not now', 'ghost');
  actionRow.appendChild(retryBtn);
  actionRow.appendChild(cancelBtn);
  rootEl.appendChild(actionRow);

  const note = document.createElement('details');
  note.style.fontSize = 'var(--text-xs, 0.75rem)';
  note.style.opacity = '0.85';
  note.style.lineHeight = '1.55';
  note.innerHTML = `
    <summary style="cursor:pointer">Why does Brave do this?</summary>
    <div style="margin-top:6px;padding-left:8px">
      Brave routes telemetry away from Google by default and Web Push
      requires going through Google’s Firebase Cloud Messaging
      service. Enabling this toggle only allows pushes routed
      through FCM to your browser — it does NOT change Brave’s
      tracking-blocker, fingerprint defense, or any other privacy
      defaults. Augmentum never sends data to Google directly; the
      push payload is end-to-end encrypted using VAPID keys that
      Augmentum generated locally.
    </div>
  `;
  rootEl.appendChild(note);

  const status = document.createElement('div');
  status.style.marginTop = '6px';
  status.style.fontSize = 'var(--text-xs, 0.75rem)';
  status.style.opacity = '0.75';
  rootEl.appendChild(status);

  retryBtn.addEventListener('click', async () => {
    retryBtn.disabled = true;
    copyBtn.disabled = true;
    cancelBtn.disabled = true;
    retryBtn.textContent = 'Retrying…';
    status.textContent = '';
    try {
      const mod = await import('./push-subscribe.js');
      await mod.enablePush({ channelPattern: '*', importanceFloor: 0 });
      _markResolved();
      status.textContent = 'Subscribed. You’ll get notifications even when this tab is closed.';
      retryBtn.textContent = 'Enabled';
      setTimeout(() => { _fadeOut(rootEl); }, 2500);
    } catch (err) {
      const msg = String(err?.message || err || '');
      if (_looksLikePushServiceError(msg)) {
        status.textContent = (
          'Still blocked. Did you fully quit Brave (including '
          + 'background processes / tray icon) before reopening? '
          + 'The setting won’t take effect until Brave restarts.'
        );
      } else {
        status.textContent = _friendlyError(msg);
      }
      retryBtn.disabled = false;
      copyBtn.disabled = false;
      cancelBtn.disabled = false;
      retryBtn.textContent = 'Try again';
    }
  });

  cancelBtn.addEventListener('click', () => { _fadeOut(rootEl); });
}

function _looksLikeCertError(msg) {
  const m = msg.toLowerCase();
  return (
    m.includes('ssl certificate')
    || m.includes('ssl error')
    || m.includes('certificate error')
    || m.includes('net::err_cert')
    || m.includes('secure connection failed')
    || m.includes('not trusted')
    || m.includes('untrusted')
    || m.includes('self signed')
    || m.includes('self-signed')
  );
}

function _mountTrustFlow(rootEl) {
  // Drop the existing prompt content, render the trust-flow panel.
  while (rootEl.firstChild) rootEl.removeChild(rootEl.firstChild);

  const os = detectOS();
  const origin = window.location.origin;

  const title = document.createElement('div');
  title.textContent = 'Trust the server cert first';
  title.style.fontWeight = '500';
  title.style.marginBottom = '6px';
  rootEl.appendChild(title);

  const body = document.createElement('div');
  body.style.opacity = '0.9';
  body.style.marginBottom = '10px';
  body.style.lineHeight = '1.45';
  body.textContent = (
    'Your browser refuses to install the push handler because '
    + 'Augmentum’s HTTPS certificate isn’t trusted on this device. '
    + 'One-time setup below — every future cert this server issues '
    + 'will then be auto-trusted.'
  );
  rootEl.appendChild(body);

  // Detected platform label.
  const detected = document.createElement('div');
  detected.style.fontSize = 'var(--text-xs, 0.75rem)';
  detected.style.opacity = '0.7';
  detected.style.marginBottom = '6px';
  detected.textContent = `Detected: ${osLabel(os)}.`;
  rootEl.appendChild(detected);

  // Primary per-OS action.
  const primary = document.createElement('div');
  primary.style.marginBottom = '10px';
  rootEl.appendChild(primary);
  renderPrimaryAction(primary, os, origin);

  // Retry + cancel row.
  const actionRow = document.createElement('div');
  actionRow.style.display = 'flex';
  actionRow.style.gap = '8px';
  actionRow.style.flexWrap = 'wrap';
  actionRow.style.marginBottom = '8px';
  const retryBtn = _btn('I installed it — try again', 'primary');
  const cancelBtn = _btn('Not now', 'ghost');
  actionRow.appendChild(retryBtn);
  actionRow.appendChild(cancelBtn);
  rootEl.appendChild(actionRow);

  // Fallbacks: other-platform instructions + raw .crt download.
  renderFallbackInstructions(rootEl);

  const status = document.createElement('div');
  status.style.marginTop = '6px';
  status.style.fontSize = 'var(--text-xs, 0.75rem)';
  status.style.opacity = '0.75';
  rootEl.appendChild(status);

  retryBtn.addEventListener('click', async () => {
    retryBtn.disabled = true;
    cancelBtn.disabled = true;
    retryBtn.textContent = 'Retrying…';
    status.textContent = '';
    try {
      const mod = await import('./push-subscribe.js');
      await mod.enablePush({ channelPattern: '*', importanceFloor: 0 });
      _markResolved();
      status.textContent = 'Subscribed. You’ll get notifications even when this tab is closed.';
      retryBtn.textContent = 'Enabled';
      setTimeout(() => { _fadeOut(rootEl); }, 2500);
    } catch (err) {
      const msg = String(err?.message || err || '');
      if (_looksLikeCertError(msg)) {
        status.textContent = (
          'Still untrusted. Browsers usually need a full restart '
          + '(quit + reopen) before trust changes take effect — '
          + 'and Firefox uses its own trust store separate from '
          + 'the OS one. Try again after a restart.'
        );
      } else if (_looksLikePushServiceError(msg) && await _isBrave()) {
        // Cert is now trusted but Brave still blocks subscribe.
        // Swap the trust panel out for the Brave fix panel — same
        // host element, different content.
        _mountBraveFix(rootEl);
        return;
      } else {
        status.textContent = _friendlyError(msg);
      }
      retryBtn.disabled = false;
      cancelBtn.disabled = false;
      retryBtn.textContent = 'Try again';
    }
  });

  cancelBtn.addEventListener('click', () => { _fadeOut(rootEl); });
}

async function _copyToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try {
    if (!document.execCommand('copy')) throw new Error('execCommand-copy-failed');
  } finally {
    ta.remove();
  }
}

function _friendlyError(code) {
  if (code.startsWith('permission_')) {
    return 'You declined the browser permission. You can change it in your browser’s site settings, then try again from Settings → Notifications.';
  }
  if (code.startsWith('vapid_')) {
    return 'Server isn’t set up for Web Push yet (missing VAPID keys).';
  }
  if (code === 'push_unsupported') {
    return 'This browser doesn’t support Web Push.';
  }
  return `Couldn’t enable: ${code}`;
}

function _fadeOut(el) {
  el.style.transition = 'opacity 250ms ease-out, max-height 250ms ease-out';
  el.style.opacity = '0';
  el.style.maxHeight = '0';
  el.style.overflow = 'hidden';
  setTimeout(() => { try { el.remove(); } catch (_) {} }, 300);
}
