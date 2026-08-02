/* notifications/cert-trust.js — shared root-CA trust flow.
 *
 * Augmentum serves its own HTTPS via a self-signed Caddy root CA. Every
 * device has to trust that root once before Service Workers, Web Push,
 * WebSocket voice/live-updates, and Cast will work over the LAN origin.
 *
 * The install path differs per OS (iOS profile, Android download, macOS
 * Keychain, Windows CurrentUser\Root, Linux ca-certificates). This module
 * owns the OS detection and per-OS rendering so EVERY surface that needs a
 * "trust this cert" control renders the SAME correct flow — the push-prompt
 * trust panel and Settings → General both consume it. Do NOT re-implement
 * OS detection or per-OS install steps elsewhere; call into here.
 *
 * The direct download endpoint is `/caddy-root-ca` (the .crt) and
 * `/caddy-root-ca.mobileconfig` (Apple one-tap profile) — served
 * unauthenticated by Caddy (public-key material, no secrets).
 */

// ── OS detection ──────────────────────────────────────────────────────

export function detectOS() {
  const ua = (navigator.userAgent || '').toLowerCase();
  const platform = (navigator.platform || '').toLowerCase();
  // iPadOS 13+ reports as MacIntel with touch — sniff that.
  if (/iphone|ipad|ipod/.test(ua)
      || (platform === 'macintel' && navigator.maxTouchPoints > 1)) {
    return 'ios';
  }
  if (/android/.test(ua)) return 'android';
  if (/mac/.test(platform)) return 'macos';
  if (/win/.test(platform)) return 'windows';
  if (/linux|x11/.test(platform)) return 'linux';
  return 'unknown';
}

export function osLabel(os) {
  return ({
    ios: 'iOS / iPadOS',
    android: 'Android',
    macos: 'macOS',
    windows: 'Windows',
    linux: 'Linux',
    unknown: 'this device',
  })[os] || os;
}

// ── Native bridge (Android app) ───────────────────────────────────────
// Inside the Android WebView, window.AugmentumAndroid.installCert() opens
// the OS secure KeyChain dialog directly — strictly better than a browser
// download. Surfaces prefer this when present.

export function hasNativeCertBridge() {
  const b = window.AugmentumAndroid;
  return !!(b && typeof b.installCert === 'function');
}

export function installViaNativeBridge() {
  window.AugmentumAndroid.installCert();
}

// ── Per-OS primary action ─────────────────────────────────────────────

export function renderPrimaryAction(host, os, origin) {
  if (os === 'ios') {
    _renderIOSPrimary(host);
    return;
  }
  if (os === 'android') {
    _renderAndroidPrimary(host);
    return;
  }
  const cmd = shellCommandFor(os, origin);
  if (cmd) {
    _renderCommandPrimary(host, cmd, os);
    return;
  }
  // Fallback for unknown OS — just the raw download.
  _renderAndroidPrimary(host);
}

function _renderIOSPrimary(host) {
  const wrap = document.createElement('div');
  const label = document.createElement('div');
  label.textContent = 'Tap below to install the Augmentum trust profile. Safari will hand off to Settings.';
  label.style.marginBottom = '6px';
  wrap.appendChild(label);
  const a = document.createElement('a');
  a.href = '/caddy-root-ca.mobileconfig';
  a.textContent = 'Install profile';
  a.className = 'btn btn-primary btn-sm';
  Object.assign(a.style, {
    fontSize: 'var(--text-xs, 0.75rem)',
    padding: '4px 10px',
    textDecoration: 'none',
    display: 'inline-block',
  });
  wrap.appendChild(a);
  const note = document.createElement('div');
  note.style.fontSize = 'var(--text-xs, 0.7rem)';
  note.style.opacity = '0.75';
  note.style.marginTop = '6px';
  note.innerHTML = (
    'After installing, you must enable full trust under '
    + '<em>Settings → General → About → Certificate Trust Settings</em>. '
    + 'Then reload Augmentum.'
  );
  wrap.appendChild(note);
  host.appendChild(wrap);
}

function _renderAndroidPrimary(host) {
  const wrap = document.createElement('div');
  const label = document.createElement('div');
  label.textContent = 'Download the certificate — your device will route you to the cert install dialog.';
  label.style.marginBottom = '6px';
  wrap.appendChild(label);
  const a = document.createElement('a');
  a.href = '/caddy-root-ca';
  a.download = 'augmentum-root-ca.crt';
  a.textContent = 'Download certificate';
  a.className = 'btn btn-primary btn-sm';
  Object.assign(a.style, {
    fontSize: 'var(--text-xs, 0.75rem)',
    padding: '4px 10px',
    textDecoration: 'none',
    display: 'inline-block',
  });
  wrap.appendChild(a);
  const note = document.createElement('div');
  note.style.fontSize = 'var(--text-xs, 0.7rem)';
  note.style.opacity = '0.75';
  note.style.marginTop = '6px';
  note.textContent = 'Install as “CA certificate” when prompted. Chrome trusts user-installed roots; Firefox needs a separate import.';
  wrap.appendChild(note);
  host.appendChild(wrap);
}

function _renderCommandPrimary(host, cmd, os) {
  const wrap = document.createElement('div');

  // Windows path: download the cert via the browser (which already
  // negotiated TLS 1.2+ for the page load), then a tiny PowerShell
  // one-liner imports the local file. Avoids the "PS 5.1 defaults to
  // TLS 1.0/1.1 and can't reach the server" failure mode.
  if (os === 'windows') {
    const label = document.createElement('div');
    label.innerHTML = (
      '<strong>Step 1:</strong> download the certificate '
      + '(browser handles the TLS connection cleanly):'
    );
    label.style.marginBottom = '6px';
    wrap.appendChild(label);

    const dlLink = document.createElement('a');
    dlLink.href = '/caddy-root-ca';
    dlLink.download = 'augmentum-root-ca.crt';
    dlLink.textContent = 'Download certificate';
    dlLink.className = 'btn btn-primary btn-sm';
    Object.assign(dlLink.style, {
      fontSize: 'var(--text-xs, 0.75rem)',
      padding: '4px 10px',
      textDecoration: 'none',
      display: 'inline-block',
      marginBottom: '10px',
    });
    wrap.appendChild(dlLink);

    const step2 = document.createElement('div');
    step2.innerHTML = (
      '<strong>Step 2:</strong> open PowerShell and run '
      + '(no admin needed — uses CurrentUser store):'
    );
    step2.style.marginBottom = '6px';
    wrap.appendChild(step2);
  } else {
    const label = document.createElement('div');
    label.textContent = 'Paste this into your terminal:';
    label.style.marginBottom = '6px';
    wrap.appendChild(label);
  }

  const block = document.createElement('div');
  Object.assign(block.style, {
    background: 'var(--surface-3, rgba(0,0,0,0.08))',
    padding: '8px 10px',
    borderRadius: '6px',
    fontFamily: 'monospace',
    fontSize: 'var(--text-xs, 0.75rem)',
    lineHeight: '1.5',
    wordBreak: 'break-all',
    whiteSpace: 'pre-wrap',
    marginBottom: '6px',
    userSelect: 'all',
  });
  block.textContent = cmd;
  wrap.appendChild(block);

  const copyBtn = _btn('Copy command', 'primary');
  copyBtn.addEventListener('click', async () => {
    const orig = copyBtn.textContent;
    try {
      await _copyToClipboard(cmd);
      copyBtn.textContent = 'Copied';
      setTimeout(() => { copyBtn.textContent = orig; }, 1500);
    } catch (_) {
      copyBtn.textContent = 'Copy failed — select manually';
      setTimeout(() => { copyBtn.textContent = orig; }, 2200);
    }
  });
  wrap.appendChild(copyBtn);

  // The obvious move — double-clicking the .crt — silently fails: the
  // Certificate Import Wizard's default "automatically select the store"
  // files a self-signed root under Intermediate Certification Authorities,
  // where it's inert. It still shows up in certmgr and in the browser's
  // cert viewer, so it LOOKS installed. Call this out explicitly.
  if (os === 'windows') {
    const gotcha = document.createElement('div');
    gotcha.style.fontSize = 'var(--text-xs, 0.7rem)';
    gotcha.style.opacity = '0.75';
    gotcha.style.marginTop = '8px';
    gotcha.style.lineHeight = '1.55';
    gotcha.innerHTML = (
      'Prefer to double-click the <code>.crt</code> instead? Choose '
      + '<em>Install Certificate</em> → <em>Place all certificates in the '
      + 'following store</em> → Browse → <strong>Trusted Root Certification '
      + 'Authorities</strong>. Do <strong>not</strong> leave it on '
      + '“Automatically select” — that files it under Intermediate, where '
      + 'it looks installed but does nothing.'
    );
    wrap.appendChild(gotcha);
  }

  host.appendChild(wrap);
}

export function shellCommandFor(os, origin) {
  const url = `${origin}/caddy-root-ca`;
  if (os === 'macos') {
    // -k to bypass the not-yet-trusted cert. add-trusted-cert from
    // stdin via /dev/stdin. login keychain so no admin is needed.
    return `curl -k '${url}' | sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain /dev/stdin`;
  }
  if (os === 'linux') {
    return `curl -k '${url}' | sudo tee /usr/local/share/ca-certificates/augmentum-root.crt >/dev/null && sudo update-ca-certificates`;
  }
  if (os === 'windows') {
    // Two-step: browser downloads cert (already negotiated TLS for the
    // page load), PowerShell only does the import. Avoids the
    // "PS 5.1 defaults to TLS 1.0/1.1, can't reach Caddy" failure where
    // ServerCertificateValidationCallback never gets a chance to run
    // because the SSL handshake fails before validation.
    // CurrentUser\\Root needs no admin; Chrome/Edge/IE trust it.
    // Firefox uses its own store — see fallback instructions.
    return `Import-Certificate -FilePath "$HOME\\Downloads\\augmentum-root-ca.crt" -CertStoreLocation Cert:\\CurrentUser\\Root`;
  }
  return '';
}

// ── Cross-platform fallback instructions ──────────────────────────────

export function renderFallbackInstructions(host, summary = 'Different device, or trouble with the above?') {
  const fallback = document.createElement('details');
  fallback.style.fontSize = 'var(--text-xs, 0.75rem)';
  fallback.style.opacity = '0.85';
  fallback.style.lineHeight = '1.55';
  fallback.innerHTML = `
    <summary style="cursor:pointer">${summary}</summary>
    <div style="margin-top:6px;padding-left:8px">
      <div style="margin-bottom:6px">
        Direct download: <a href="/caddy-root-ca"
          download="augmentum-root-ca.crt"
          style="color:inherit">augmentum-root-ca.crt</a>
        — install per your OS docs.
      </div>
      <div style="margin-bottom:4px"><strong>macOS</strong>: open
        the <code>.crt</code> → Keychain → double-click the
        imported root → set <em>Trust</em> to <em>Always Trust</em>.
        Restart the browser.</div>
      <div style="margin-bottom:4px"><strong>Windows</strong>: open
        the <code>.crt</code> → <em>Install Certificate</em> →
        <em>Place all certificates in the following store</em> → Browse →
        <em>Trusted Root Certification Authorities</em>. Leaving it on
        “Automatically select” files it under Intermediate, where it
        looks installed but is never trusted. Restart the browser.</div>
      <div style="margin-bottom:4px"><strong>Linux</strong>: copy
        to <code>/usr/local/share/ca-certificates/</code> →
        <code>sudo update-ca-certificates</code>. For Firefox also
        import via <code>about:preferences#privacy</code> → View
        Certificates → Authorities.</div>
      <div style="margin-bottom:4px"><strong>iOS / iPadOS</strong>:
        tap the “Install profile” link from Safari → Settings →
        Profile Downloaded → Install. Then enable full trust under
        <em>Settings → General → About → Certificate Trust
        Settings</em>.</div>
      <div style="margin-bottom:4px"><strong>Android</strong>: open the
        <code>.crt</code> → install as <em>CA certificate</em>. Some
        Android versions only trust user roots for Chrome
        navigations.</div>
      <div><strong>Still untrusted after installing?</strong> Certificate
        stores are not shared. Firefox never uses the OS store
        (<code>about:preferences#privacy</code> → View Certificates →
        Authorities → Import). Chrome and Brave 133+ each carry their own
        at <code>chrome://certificate-manager</code> → Custom → Trusted
        Certificates — importing there covers that browser only, so
        installing to the OS store is the one that carries everywhere.
        Fully quit the browser afterward; trust is cached per-process.</div>
    </div>
  `;
  host.appendChild(fallback);
  return fallback;
}

// ── High-level panel ──────────────────────────────────────────────────
// One call renders the full self-serve trust flow into `host`: the detected
// platform, the correct primary action for it, and the cross-platform
// fallback. Used by Settings → General. The push-prompt trust panel builds
// its own variant because it adds a subscribe-retry step on top.

export function mountCertTrustPanel(host) {
  while (host.firstChild) host.removeChild(host.firstChild);

  // Inside the Android app, the native secure dialog is the best path.
  if (hasNativeCertBridge()) {
    const label = document.createElement('div');
    label.style.marginBottom = '8px';
    label.textContent = 'Install your Augmentum server’s certificate so voice and live updates work over your local network.';
    host.appendChild(label);
    const btn = _btn('Install certificate', 'primary');
    const status = document.createElement('div');
    status.style.marginTop = '6px';
    status.style.fontSize = 'var(--text-xs, 0.75rem)';
    status.style.opacity = '0.8';
    btn.addEventListener('click', () => {
      try {
        installViaNativeBridge();
        status.textContent = 'Opening the certificate dialog… confirm to trust your server.';
      } catch (_) {
        status.textContent = 'Couldn’t start the certificate install.';
      }
    });
    host.appendChild(btn);
    host.appendChild(status);
    // Still offer the manual fallback in case the bridge misbehaves.
    renderFallbackInstructions(host);
    return;
  }

  const os = detectOS();
  const origin = window.location.origin;

  const detected = document.createElement('div');
  detected.style.fontSize = 'var(--text-xs, 0.75rem)';
  detected.style.opacity = '0.7';
  detected.style.marginBottom = '6px';
  detected.textContent = `Detected: ${osLabel(os)}.`;
  host.appendChild(detected);

  const primary = document.createElement('div');
  primary.style.marginBottom = '10px';
  host.appendChild(primary);
  renderPrimaryAction(primary, os, origin);

  renderFallbackInstructions(host);
}

// ── Local helpers (kept self-contained so this module has no deps) ─────

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
