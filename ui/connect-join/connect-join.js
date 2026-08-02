// Connect invite onboarding — claim an account from an invite link.
//
// Flow: read ?token= → GET /api/auth/invite/{token} (public preview) → render
// a create-account form → POST /api/auth/invite/{token}/claim (self-set
// password, auto-login) → redirect into the app. See auth_routes.py invite
// endpoints and the design doc 2026-06-20-connect-comms-platform-design.md.

const $ = (sel, root = document) => root.querySelector(sel);

function show(state) {
  for (const card of document.querySelectorAll('.cj-card')) {
    card.hidden = card.getAttribute('data-state') !== state;
  }
}

function terminal(title, sub) {
  $('[data-terminal-title]').textContent = title;
  $('[data-terminal-sub]').textContent = sub || '';
  show('terminal');
}

function getToken() {
  const p = new URLSearchParams(location.search);
  return (p.get('token') || '').trim();
}

const STATUS_MESSAGES = {
  expired: ['Invite expired', 'This invite link has passed its expiry. Ask whoever invited you for a fresh one.'],
  used: ['Invite already used', 'This invite has been claimed the maximum number of times.'],
  revoked: ['Invite revoked', 'This invite was turned off by the person who created it.'],
};

async function loadPreview(token) {
  let res;
  try {
    res = await fetch(`/api/auth/invite/${encodeURIComponent(token)}`, {
      headers: { 'Accept': 'application/json' },
    });
  } catch {
    return terminal('Connection problem', 'Could not reach the server. Check your connection and try again.');
  }
  if (res.status === 404) {
    return terminal('Invite not found', "This invite link isn't valid. Double-check the link you were sent.");
  }
  if (!res.ok) {
    return terminal('Invite unavailable', 'Something went wrong checking this invite. Try again shortly.');
  }
  const { invite } = await res.json();
  if (invite.status && invite.status !== 'active') {
    const [t, s] = STATUS_MESSAGES[invite.status] || ['Invite unavailable', ''];
    return terminal(t, s);
  }

  const inviter = (invite.inviter_display_name || '').trim();
  const instance = (invite.instance_handle || 'this Augmentum').trim();
  $('[data-form-title]').textContent = inviter ? `${inviter} invited you` : 'You’re invited';
  $('[data-form-sub]').innerHTML = `Create your account to join <strong>${escapeHtml(instance)}</strong> on Augmentum Connect.`;

  // Personalise the visibility choice with the real inviter / instance names so
  // the privacy decision is concrete ("Only Sara" vs "Anyone on home.example").
  const privTitle = $('[data-vis-private-title]');
  const pubTitle = $('[data-vis-public-title]');
  if (privTitle && inviter) privTitle.textContent = `Only ${inviter} (who invited you)`;
  if (pubTitle) pubTitle.textContent = `Anyone on ${instance}`;

  // Pre-fill display name hint if the inviter suggested a handle.
  if (invite.handle_hint) $('[data-display]').value = invite.handle_hint;

  show('form');
  $('[data-username]').focus();
  wireForm(token);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

const USERNAME_STATUS = {
  ok: ['available', 'cj-ok'],
  taken: ['That username is taken — pick another.', 'cj-err'],
  reserved: ['That username is reserved — pick another.', 'cj-err'],
  invalid: ['3–32 characters: letters, numbers, or underscore.', 'cj-err'],
};

// Live username availability check, gated behind the live invite token and
// debounced so we don't probe on every keystroke. Best-effort: a network blip
// just leaves the hint blank — the claim handler re-validates authoritatively.
function wireUsernameCheck(token) {
  const input = $('[data-username]');
  const statusEl = $('[data-username-status]');
  if (!input || !statusEl) return;

  let timer = null;
  let seq = 0;

  const render = (reason) => {
    const [msg, cls] = USERNAME_STATUS[reason] || ['', ''];
    statusEl.textContent = msg;
    statusEl.className = `cj-username-status ${cls}`;
    statusEl.hidden = !msg;
  };

  const check = async () => {
    const u = input.value.trim();
    if (!/^[a-zA-Z0-9_]{3,32}$/.test(u)) {
      render(u ? 'invalid' : null);
      return;
    }
    render(null);
    const mine = ++seq;
    let res;
    try {
      res = await fetch(
        `/api/auth/invite/${encodeURIComponent(token)}/check-username?u=${encodeURIComponent(u)}`,
        { headers: { 'Accept': 'application/json' } },
      );
    } catch { return; }            // offline — stay silent, claim re-validates
    if (mine !== seq || !res.ok) return;   // a newer keystroke superseded this
    let data;
    try { data = await res.json(); } catch { return; }
    render(data.available ? 'ok' : (data.reason || null));
  };

  input.addEventListener('input', () => {
    clearTimeout(timer);
    statusEl.hidden = true;
    timer = setTimeout(check, 350);
  });
}

function wireForm(token) {
  const form = $('[data-claim-form]');
  const errEl = $('[data-form-error]');
  const submit = $('[data-submit]');

  wireUsernameCheck(token);

  const setError = (msg) => {
    errEl.textContent = msg;
    errEl.hidden = !msg;
  };

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setError('');

    const username = $('[data-username]').value.trim();
    const password = $('[data-password]').value;
    const confirm = $('[data-confirm]').value;
    const displayName = $('[data-display]').value.trim();
    // Default to private (only the inviter) unless the user opted to be listed.
    const discoverable = $('[data-vis-public]')?.checked === true;

    if (!/^[a-zA-Z0-9_]{3,32}$/.test(username)) {
      return setError('Username must be 3–32 characters: letters, numbers, or underscore.');
    }
    if (password.length < 8) {
      return setError('Password must be at least 8 characters.');
    }
    if (password !== confirm) {
      return setError('Passwords don’t match.');
    }

    submit.disabled = true;
    submit.textContent = 'Creating account…';
    let res;
    try {
      res = await fetch(`/api/auth/invite/${encodeURIComponent(token)}/claim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, display_name: displayName, discoverable }),
      });
    } catch {
      submit.disabled = false;
      submit.textContent = 'Create account & join';
      return setError('Could not reach the server. Try again.');
    }

    if (res.status === 201) {
      show('done');
      // External-guest claims return a durable grant token — stash it and hand
      // off to the installable guest surface instead of the full app. Full
      // accounts land in the SPA, which is mounted at /ui/ (NOT bare "/", which
      // is a plain-text stub / can resolve to the wrong upstream behind a proxy).
      let dest = '/ui/';
      try {
        const data = await res.json();
        if (data && data.guest_grant_token) {
          localStorage.setItem('augmentum_guest_grant_token', data.guest_grant_token);
          dest = '/ui/connect-guest/';
        }
      } catch { /* ignore — fall through to the app */ }
      setTimeout(() => { location.href = dest; }, 900);
      return;
    }

    submit.disabled = false;
    submit.textContent = 'Create account & join';
    let detail = '';
    try { detail = (await res.json()).error || ''; } catch { /* ignore */ }

    if (res.status === 410) {
      // Invite was consumed/expired/revoked between preview and claim.
      return terminal('Invite no longer valid', detail || 'This invite was just used up or turned off.');
    }
    setError(detail || 'Could not create your account. Try a different username.');
  });
}

const token = getToken();
if (!token) {
  terminal('Missing invite', 'This page needs an invite link. Ask whoever invited you to resend it.');
} else {
  loadPreview(token);
}
