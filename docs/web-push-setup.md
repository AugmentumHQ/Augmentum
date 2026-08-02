# Web Push Setup

Augmentum's standing tasks (recurring briefings, reminders) and Connect
(incoming calls, messages) can buzz your device with OS-level
notifications when the Augmentum tab isn't open — but only if the
browser has been allowed to subscribe to Web Push. This guide covers
the one-time setup per device and the common failure modes.

When everything's wired correctly, the experience is:

* Tab open → in-app banner via WebSocket dispatch (works immediately,
  no setup needed)
* Tab closed → OS-level notification via Web Push (requires the setup
  described below)
* Either way the event also lands in the companion notes drawer and
  the notifications feed, so missing a push is never silent data loss.

---

## What needs to be true

Web Push requires **all** of:

1. The Augmentum server is reachable over HTTPS with a certificate
   the browser trusts.
2. The Service Worker at `/notification-sw.js` registers successfully
   (depends on #1).
3. `Notification.requestPermission()` resolved to `granted`.
4. `pushManager.subscribe()` succeeded, which itself depends on the
   browser being able to reach **its** push provider (FCM for
   Chrome/Edge/Brave-with-Google-services-on, Mozilla autopush for
   Firefox, etc.).
5. The subscription endpoint + p256dh + auth keys were POSTed to
   `/api/notify/subscriptions` and persisted.

Augmentum's UI walks you through (1)-(5) when you click "Enable
browser notifications" from Settings → Notifications or from the
just-in-time prompt that appears after scheduling a briefing.
Failures at each step surface a specific recovery panel — none of
them drop you into a generic "Couldn't enable" dead end.

---

## Step 1 — HTTPS trust

The Augmentum server runs behind Caddy with a self-signed root CA
that's generated on first start (persisted in the `caddy_data`
volume so the fingerprint stays stable forever). You import the root
CA once per device and every cert this server ever issues — now and
after future regenerations (IP changes, new Tailscale hostname,
etc.) — is auto-trusted.

### Skip this if you're using localhost

If you access Augmentum via `http://localhost:6100` or
`http://127.0.0.1:6100` (i.e. you're on the same machine as the
Docker host), Service Workers treat localhost as a secure context
even over plain HTTP. Skip trust setup entirely.

### Skip this if you have a publicly-trusted cert

If you put a domain in front of Augmentum and have Let's Encrypt
issue real certs (or you use Tailscale's `*.ts.net` hostnames with
their built-in cert), browsers trust those automatically. The trust
setup is only needed for direct LAN/IP access with the bundled
self-signed cert.

### One-time install per device

The in-app trust panel detects your OS and gives you the right path:

#### macOS

```bash
curl -k 'https://<your-server>:6443/caddy-root-ca' | \
  sudo security add-trusted-cert -d -r trustRoot \
    -k /Library/Keychains/System.keychain /dev/stdin
```

#### Linux

```bash
curl -k 'https://<your-server>:6443/caddy-root-ca' | \
  sudo tee /usr/local/share/ca-certificates/augmentum-root.crt >/dev/null && \
  sudo update-ca-certificates
```

#### Windows (PowerShell, no admin needed)

1. Click "Download certificate" in the in-app trust panel (your
   browser handles TLS for the page already, so the download
   bypasses PowerShell 5.1's TLS 1.0/1.1 default).
2. Run in PowerShell:

   ```powershell
   Import-Certificate -FilePath "$HOME\Downloads\augmentum-root-ca.crt" `
     -CertStoreLocation Cert:\CurrentUser\Root
   ```

#### iOS / iPadOS

Tap "Install profile" in the in-app trust panel. Safari hands the
`.mobileconfig` to Settings → Profile Downloaded → Install.
**Important:** after installing the profile, you must also enable
full trust under
*Settings → General → About → Certificate Trust Settings* — iOS
disables newly-installed roots by default.

#### Android

Tap "Download certificate" in the in-app trust panel. Android's OS
dialog opens automatically and prompts you to install as
*CA certificate*. Chrome trusts user-installed roots; Firefox uses
its own store and needs a separate import.

### After install: restart the browser

Cert trust is cached per browser process. Quit fully (close all
windows + tray icon) and reopen before testing.

---

## Step 2 — Browser-specific quirks

### Brave: enable Google services for push

Brave **disables Web Push by default** because the spec routes
through Google's Firebase Cloud Messaging. The error surfaces as
*"Registration failed - push service error"* during
`pushManager.subscribe()`.

Fix:

1. Open `brave://settings/privacy` in Brave
2. Toggle **"Use Google services for push messaging"** on
3. **Fully quit Brave** (close all windows + tray icon — the setting
   is cached per process)
4. Reopen Brave, return to Augmentum, retry

This enables FCM routing for push only. It does **not** change
Brave's tracker blocker, fingerprint defenses, or any other privacy
defaults. The push payload itself is end-to-end encrypted via the
VAPID keys Augmentum generated locally — Google's FCM just brokers
delivery, never sees the contents.

The in-app prompt detects Brave via `navigator.brave.isBrave()` and
swaps to a Brave-specific panel automatically when this failure
mode hits.

### Firefox: uses its own cert + push stores

Firefox doesn't share the OS cert trust store on most platforms.
Even after you install the Augmentum root via OS-level commands
above, Firefox will still flag the cert as untrusted. Import via
`about:preferences#privacy` → View Certificates → Authorities →
Import.

Firefox also uses Mozilla autopush instead of FCM, so the Brave-
style "Google services" toggle is irrelevant; Firefox push works
out of the box once cert trust is sorted.

### Un-Googled Chromium variants

Chromium builds without Google Play Services (Ungoogled Chromium,
some de-Googled forks) can't reach FCM and Web Push won't work at
all. Use a different browser for Augmentum, or rely on in-app
notifications when the tab is open + the notes drawer when it
isn't.

### Safari (macOS / iOS)

Safari supports Web Push only for installed PWAs (since macOS 13 /
iOS 16.4). To get push on Safari, you'd need to install Augmentum
as a PWA via *File → Add to Dock* (macOS) or *Share → Add to Home
Screen* (iOS). Push from a regular Safari tab is not supported.

---

## Step 3 — Confirm subscription

After "Enable browser notifications" reports
*"Subscribed — you'll get notifications even when this tab is
closed,"* verify the row landed server-side:

```bash
docker exec augmentum-augmentum-1 sh -c \
  'python3 -c "
import sqlite3
c = sqlite3.connect(\"/data/augmentum.db\")
for r in c.execute(
    \"SELECT user_id, target_kind, substr(target_address, 1, 60), \"
    \"channel_pattern, created_at FROM notification_subscriptions\"
):
    print(r)
"'
```

You should see a row with `target_kind='webpush'` and the FCM /
autopush endpoint URL.

To send a synthetic test push and confirm end-to-end:

```bash
docker exec augmentum-augmentum-1 sh -c \
  'python3 -c "
import asyncio, aiosqlite
from augmentum.notifications.webpush import send_webpush_to_user

async def main():
    conn = await aiosqlite.connect(\"/data/augmentum.db\")
    await send_webpush_to_user(
        conn, user_id=\"<your-user-id>\", title=\"Test\",
        body=\"Hello from Augmentum\",
    )
    await conn.close()
asyncio.run(main())
"'
```

(Replace `<your-user-id>` with your actual user id from the `users`
table.) The notification should appear on your device within a few
seconds. If it doesn't, check the augmentum container logs for
`webpush_send_failed` and the HTTP status code returned by the push
service.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Couldn't enable: Failed to register a ServiceWorker... SSL certificate error" | Root CA not installed / not trusted by browser | Run trust install for your OS, **restart browser fully** |
| "Couldn't enable: Failed to register... 404" | `/notification-sw.js` route missing | Upgrade Augmentum — root-level SW route was added 2026-06-04 |
| "Couldn't enable: Registration failed - push service error" (Brave) | Brave disables FCM by default | Toggle on `brave://settings/privacy` → "Use Google services for push messaging" |
| "Couldn't enable: Registration failed - push service error" (other browsers) | Network can't reach `fcm.googleapis.com` or `autopush.services.mozilla.com` | Check firewall, VPN, corporate / school network rules, privacy extensions, DNS filters (Pi-hole, NextDNS) |
| "Couldn't enable: Server isn't set up for Web Push yet (missing VAPID keys)" | `notifications_enabled` setting is False, so `/api/notify/vapid-public-key` returns 503 before VAPID can auto-generate | Set `notifications_enabled` to True in Settings (or directly in `app_settings` table); restart augmentum container |
| Subscription succeeds but no push arrives | `notification_subscriptions` row stale (browser was uninstalled / push token expired); or push service returned 410 Gone | The dispatcher prunes 410s automatically on next send — re-enable from Settings to create a fresh subscription |
| Push works in Chrome but not Brave / Firefox | Browser-specific quirk — see [Browser-specific quirks](#step-2--browser-specific-quirks) above | |

---

## Related files

* `augmentum/notifications/webpush.py` — VAPID lifecycle, send path
* `augmentum/notifications/hub.py` — `publish_and_dispatch`,
  offline-only fan-out
* `augmentum/proxy/notifications_routes.py` — VAPID + subscription
  HTTP routes
* `ui/scripts/notifications/push-subscribe.js` — client subscribe
  flow
* `ui/scripts/notifications/push-prompt.js` — in-app trust + Brave
  recovery UI
* `ui/notification-sw.js` — push event handler
* `compose.yaml` (caddy service) — root CA generation, leaf
  signing, `/caddy-root-ca` + `/caddy-root-ca.mobileconfig` routes
