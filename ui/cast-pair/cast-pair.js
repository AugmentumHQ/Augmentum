/**
 * cast-pair.js — phone-side approval handler.
 *
 * Flow: page loads with ?code=XXXXX in the URL. We POST to
 * /api/cast/pair/approve/{code} via the session cookie.
 *
 * If the user isn't logged in (401), we show an inline login form
 * RIGHT HERE — the canonical QR-scan flow opens a fresh browser
 * tab on the phone with no session, and bouncing to /ui/ for login
 * loses the pair code in the URL. Keeping the form inline preserves
 * the code through the login.
 *
 * On approve success the receiver poll picks up the ws_token within
 * its next tick and the receiver page goes live.
 */

const body = document.querySelector('[data-cp-body]');

const params = new URLSearchParams(location.search);
const PAIR_CODE = (params.get('code') || '').trim().toUpperCase();


function render(html) {
  body.innerHTML = html;
}


function showError(message, opts = {}) {
  let footer = '';
  if (opts.retry) {
    footer = `<button class="cp-btn" data-cp-retry style="margin-top:12px">Try again</button>`;
  }
  render(`
    <p class="cp-status error">${message}</p>
    ${footer}
  `);
  if (opts.retry) {
    body.querySelector('[data-cp-retry]')?.addEventListener('click', () => approve());
  }
}


function showSuccess() {
  // Auto-jump straight into the cast-control surface so the user
  // lands in "browse for TV" mode after pairing — no detour through
  // the main UI. Small grace window so they see the "paired" beat
  // (visual confirmation that the QR scan worked) before the swap.
  render(`
    <p class="cp-status success">Receiver paired</p>
    <div class="cp-code">${PAIR_CODE}</div>
    <p class="cp-status" style="font-size:13px;color:#888;margin-top:6px">
      Opening TV controls…
    </p>
    <p class="cp-status" style="font-size:11px;color:#666;margin-top:4px">
      <a href="/ui/cast-control/" style="color:inherit;text-decoration:underline">Tap here</a>
      if it doesn't redirect.
    </p>
  `);
  // The 700ms grace window lets the user register the "Receiver
  // paired" line so the redirect doesn't feel like the page glitched.
  setTimeout(() => {
    // ``replace`` so the back button skips the pair page (no point
    // returning to a one-shot QR scan; back should go to whatever
    // the user was doing before scanning).
    window.location.replace('/ui/cast-control/');
  }, 700);
}


function showLoginForm(initialError = '') {
  render(`
    <p class="cp-status">Log in to pair this receiver</p>
    <form data-cp-login class="cp-login" autocomplete="on" style="display:flex;flex-direction:column;gap:10px;width:100%;margin-top:8px">
      <input class="cp-input" type="text" name="username" placeholder="Username" autocomplete="username" autofocus required>
      <input class="cp-input" type="password" name="password" placeholder="Password" autocomplete="current-password" required>
      <p class="cp-status error" data-cp-login-error style="min-height:18px">${initialError || ''}</p>
      <button type="submit" class="cp-btn">Sign in &amp; pair</button>
    </form>
  `);

  const form = body.querySelector('[data-cp-login]');
  const errEl = body.querySelector('[data-cp-login-error]');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errEl.textContent = '';
    const username = form.username.value.trim();
    const password = form.password.value;
    if (!username || !password) {
      errEl.textContent = 'Both fields are required.';
      return;
    }
    const btn = form.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.textContent = 'Signing in…';

    let loginResp;
    try {
      loginResp = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ username, password }),
      });
    } catch (err) {
      errEl.textContent = `Connection error: ${err.message || err}`;
      btn.disabled = false;
      btn.textContent = 'Sign in & pair';
      return;
    }

    if (!loginResp.ok) {
      const data = await loginResp.json().catch(() => ({}));
      if (data.retry_after) {
        const mins = Math.ceil(data.retry_after / 60);
        errEl.textContent = `Too many attempts. Try again in ${mins} minute${mins > 1 ? 's' : ''}.`;
      } else {
        errEl.textContent = data.error || `Login failed (${loginResp.status}).`;
      }
      btn.disabled = false;
      btn.textContent = 'Sign in & pair';
      return;
    }

    // Logged in — session cookie is set on this origin. Retry approve.
    await approve();
  });
}


// The TV-location choice, captured before we grant the token and reused
// across the login round-trip (401 → login → approve again).
let _chosenLifetime = '';


function promptLocation() {
  if (!PAIR_CODE) {
    showError('Missing pair code in URL.');
    return;
  }
  // Ask EVERY pairing, before any token is issued. Home = a trusted,
  // always-on cast target that stays signed in (silent reconnect across
  // restarts); Away = a borrowed/public screen that re-pairs the same day.
  render(`
    <p class="cp-status">Where is this TV?</p>
    <p class="cp-status" style="font-size:13px;color:#888;margin:6px 0 16px">
      This decides how long it stays signed in.
    </p>
    <div style="display:flex;flex-direction:column;gap:10px">
      <button class="cp-btn" data-cp-home
        style="padding:14px;font-size:15px">
        🏠 At home — stay connected
        <div style="font-size:12px;color:#bbb;font-weight:400;margin-top:3px">
          Reconnects on its own after restarts. Play to it anytime.
        </div>
      </button>
      <button class="cp-btn" data-cp-away
        style="padding:14px;font-size:15px;background:#2b2b30">
        🧳 Away / shared screen
        <div style="font-size:12px;color:#bbb;font-weight:400;margin-top:3px">
          Signs out within the day — re-pair to use again.
        </div>
      </button>
    </div>
  `);
  body.querySelector('[data-cp-home]')?.addEventListener('click', () => {
    _chosenLifetime = 'home';
    approve();
  });
  body.querySelector('[data-cp-away]')?.addEventListener('click', () => {
    _chosenLifetime = 'away';
    approve();
  });
}


async function approve() {
  if (!PAIR_CODE) {
    showError('Missing pair code in URL.');
    return;
  }
  // If we somehow reached approve without a location choice (e.g. a
  // retry path), ask first rather than silently defaulting.
  if (!_chosenLifetime) {
    promptLocation();
    return;
  }

  // Spinner while we wait.
  render(`<div class="cp-spinner"></div><p class="cp-status">Pairing…</p>`);

  let resp;
  try {
    resp = await fetch(`/api/cast/pair/approve/${encodeURIComponent(PAIR_CODE)}`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lifetime: _chosenLifetime }),
    });
  } catch (err) {
    showError(`Network error: ${err.message || err}`, { retry: true });
    return;
  }

  if (resp.status === 401) {
    // Not logged in — show inline login. After login, approve()
    // gets called again from the login submit handler.
    showLoginForm();
    return;
  }
  if (resp.status === 409) {
    showError('That pair code is expired or already claimed. Restart pairing on the receiver.');
    return;
  }
  if (!resp.ok) {
    showError(`Pairing failed (status ${resp.status}).`, { retry: true });
    return;
  }

  showSuccess();
}


// Entry: ask where the TV is BEFORE granting a token (every pairing).
promptLocation();
