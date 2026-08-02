/* connect/invite-mint.js — the ONE place invites are minted + shown.
 *
 * Class fix (2026-07-16 guest-gateway spec): three separate mint sites
 * (connect home dialog, guests panel, settings account modal) each rolled
 * their own POST /api/auth/invites + link display. Two of them never sent
 * `recipient_scope` (so the server defaulted to LAN) and ignored the `reach`
 * metadata — handing out unreachable 192.168/100.x links for external guests
 * with no warning. This module centralises:
 *   - ALWAYS asking the scope (never auto-pick — product rule),
 *   - honest reach display,
 *   - a BLOCKED state (no copyable dead link) when an external scope has no
 *     public door provisioned,
 *   - link + QR side by side, the copy string being the full bundle URL
 *     (with the #k= identity pin fragment the server attached).
 */
import { escapeHtml, showToast } from '../app.js';
import { icon } from './icons.js';

export const SCOPE_OPTIONS = [
  { v: 'lan', label: 'Same network', hint: "They're on your Wi-Fi / LAN. No exposure beyond your network." },
  { v: 'tailnet', label: 'My tailnet', hint: "They're on (or you'll add them to) your Tailscale tailnet. Private, no public exposure." },
  { v: 'public', label: 'Anywhere', hint: "They're on the open internet. Opens a temporary, anonymous door just for the handshake — torn down once they're in." },
];

/** Mint an external_guest invite for a scope. Returns the invite dict or
 *  throws with a friendly message. */
export async function mintInvite({ scope = 'lan', role = 'guest' } = {}) {
  const resp = await fetch('/api/auth/invites', {
    method: 'POST', credentials: 'same-origin',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      kind: 'external_guest', role,
      expires_in_hours: 168, max_uses: 1, recipient_scope: scope,
    }),
  });
  if (resp.status === 401 || resp.status === 403) {
    throw new Error('Only an admin can create invites. Ask your Augmentum admin to send you one.');
  }
  if (!resp.ok) throw new Error('Could not create an invite. Try again.');
  const { invite } = await resp.json();
  return invite;
}

/** True when the chosen scope couldn't actually be satisfied, so the only link
 *  we could build is a dead LAN URL. The blocked state — never hand that out.
 *   - public : blocked unless a public door opened (reach.public).
 *   - tailnet: blocked unless a tailnet-or-more-exposed tier was reachable; a
 *     null/LAN fallback means no tailnet host was found (Tailscale not running
 *     or its IP not in the TLS SANs), so "My tailnet" would otherwise leak a
 *     192.168 link.
 *   - lan    : never blocked (a LAN link is exactly what was asked for). */
export function isBlocked(invite, scope) {
  const reach = invite.reach || {};
  if (scope === 'public') return !reach.public;
  if (scope === 'tailnet') {
    return !['tailnet', 'ts_funnel', 'cloudflared'].includes(reach.tier);
  }
  return false;
}

/** Render the mint RESULT (link + QR, or the blocked remediation) into
 *  `host`. Pure DOM; caller owns layout around it. */
export function renderResult(host, invite, scope) {
  const reach = invite.reach || {};
  const url = invite.join_url || (location.origin + (invite.join_path || ''));
  const qrPath = invite.qr_path || (invite.token ? `/api/invite/${invite.token}/qr.png` : '');

  if (isBlocked(invite, scope)) {
    const tailnetBlocked = scope === 'tailnet';
    const title = tailnetBlocked ? 'No tailnet address found' : 'No public door yet';
    const defaultNote = tailnetBlocked
      ? "This machine doesn't have a Tailscale address, so a “My tailnet” link would only work on your own network."
      : "This instance can't reach the open internet yet, so an “Anywhere” link would only work on your own network.";
    const remedy = tailnetBlocked
      ? `<ul class="connect-invite-remedy">
          <li>Make sure <strong>Tailscale is running</strong> on this machine and it has a <code>100.x</code> address.</li>
          <li>Restart Augmentum after joining the tailnet so its address is picked up (it's auto-detected into the TLS SANs).</li>
        </ul>
        <p class="connect-invite-sub">Or pick <strong>Same network</strong> if they're on your Wi-Fi, or <strong>Anywhere</strong> for someone off-network.</p>`
      : `<ul class="connect-invite-remedy">
          <li><strong>Tailscale Funnel</strong> — most private; set <code>AUGMENTUM_CONNECT_FUNNEL=1</code> with a tailnet.</li>
          <li><strong>cloudflared</strong> — zero-config; already bundled in the Docker image, so just pick “Anywhere”. (Bare-metal installs: put the <code>cloudflared</code> binary on <code>PATH</code>.)</li>
          <li><strong>Public host</strong> — set <code>AUGMENTUM_PUBLIC_HOST</code> to your domain.</li>
        </ul>
        <p class="connect-invite-sub">Or pick <strong>Same network</strong> / <strong>My tailnet</strong> if they can reach you that way.</p>`;
    host.innerHTML = `
      <div class="connect-invite-blocked">
        <div class="connect-invite-blocked-title">${icon('alert-triangle', { size: 16 })}<span>${escapeHtml(title)}</span></div>
        <p class="connect-invite-note is-warn">${escapeHtml(reach.note || defaultNote)}</p>
        <p class="connect-invite-sub">To fix it:</p>
        ${remedy}
      </div>`;
    return;
  }

  const tierLabel = reach.tier ? String(reach.tier).replace(/_/g, ' ') : 'local';
  const warn = reach.privacy_downgrade || (reach.note && !reach.public && scope === 'public');
  // Worst-case (anonymous cloudflared) but a ts.net node exists → nudge the
  // operator toward the private, durable Funnel address.
  const upgrade = reach.upgrade_available
    ? `<p class="connect-invite-note is-warn">${escapeHtml(reach.upgrade_hint
        || 'Temporary anonymous address — enable Tailscale Funnel for a private, durable link.')}
        ${reach.tailnet_hostname ? `<br><span class="connect-invite-sub">Your Tailscale address: <code>${escapeHtml(reach.tailnet_hostname)}</code></span>` : ''}</p>`
    : '';
  const durability = reach.public && reach.durable === false
    ? '<span class="connect-invite-sub"> · this address changes if the host restarts</span>'
    : '';
  host.innerHTML = `
    <div class="connect-invite-share">
      <div class="connect-invite-linkrow">
        <input class="connect-invite-link" type="text" readonly value="${escapeHtml(url)}">
        <button class="connect-invite-copy" type="button">Copy</button>
      </div>
      ${qrPath ? `<div class="connect-invite-qr"><img alt="Invite QR code" src="${escapeHtml(qrPath)}"></div>` : ''}
      <p class="connect-invite-meta">One-time link · expires in 7 days · reaches via <strong>${escapeHtml(tierLabel)}</strong>${durability}.</p>
      ${reach.note ? `<p class="connect-invite-note ${warn ? 'is-warn' : ''}">${escapeHtml(reach.note)}</p>` : ''}
      ${upgrade}
    </div>`;
  const input = host.querySelector('.connect-invite-link');
  const copyBtn = host.querySelector('.connect-invite-copy');
  copyBtn.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(url); }
    catch { input.select(); document.execCommand('copy'); }
    copyBtn.textContent = 'Copied';
    showToast('Copied', 'info');
    setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
  });
  input.focus(); input.select();
}

/** A full scope-picker + mint form mounted into `host`. `onDone` fires with
 *  the minted invite after a successful mint (e.g. to refresh a list). */
export function mountMintForm(host, { role = 'guest', onDone } = {}) {
  host.innerHTML = `
    <label class="connect-invite-scopelabel">Where are they?
      <select class="connect-invite-scope">
        ${SCOPE_OPTIONS.map((o) => `<option value="${o.v}">${escapeHtml(o.label)}</option>`).join('')}
      </select>
    </label>
    <p class="connect-invite-scopehint" data-scopehint>${escapeHtml(SCOPE_OPTIONS[0].hint)}</p>
    <div class="connect-invite-result" data-result hidden></div>
    <div class="connect-invite-actions">
      <button class="connect-invite-create" type="button">Create link</button>
    </div>`;
  const scopeSel = host.querySelector('.connect-invite-scope');
  const scopeHint = host.querySelector('[data-scopehint]');
  const result = host.querySelector('[data-result]');
  const createBtn = host.querySelector('.connect-invite-create');

  scopeSel.addEventListener('change', () => {
    const opt = SCOPE_OPTIONS.find((o) => o.v === scopeSel.value);
    scopeHint.textContent = opt ? opt.hint : '';
  });
  createBtn.addEventListener('click', async () => {
    createBtn.disabled = true; createBtn.textContent = 'Creating…';
    try {
      const invite = await mintInvite({ scope: scopeSel.value, role });
      result.hidden = false;
      renderResult(result, invite, scopeSel.value);
      createBtn.textContent = 'Create another';
      if (onDone) onDone(invite);
    } catch (ex) {
      showToast(ex.message, ex.message.includes('admin') ? 'info' : 'error');
      createBtn.textContent = 'Create link';
    } finally {
      createBtn.disabled = false;
    }
  });
}
