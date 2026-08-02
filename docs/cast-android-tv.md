# Android TV Receiver — Build & Sideload Guide

The `augmentum/cast/android-tv-receiver/` project is a thin Android TV
APK that boots into the Augmentum cast-receiver web app fullscreen.
Once installed it survives reboots, auto-launches, and behaves like a
real TV app — no browser tab to manage.

This guide covers building the APK and sideloading it onto an Android
TV / Google TV device (Onn 4K box, NVIDIA Shield, Chromecast w/ Google
TV, Fire TV with Android TV launcher, etc.).

## Prerequisites

| Need | Where to get it |
|---|---|
| Java 17 JDK | `apt install openjdk-17-jdk` (Linux) / [adoptium.net](https://adoptium.net/) (Win/Mac) |
| Android Studio **or** standalone Android SDK + command-line tools | [developer.android.com/studio](https://developer.android.com/studio) |
| `adb` (Android Debug Bridge) | Bundled with Android SDK platform-tools |

Verify:

```bash
java -version        # 17+
adb version          # Anything 1.0.41+ is fine
```

## 1 — Build the APK

```bash
cd augmentum/cast/android-tv-receiver
./gradlew assembleDebug
```

(On Windows: `.\gradlew.bat assembleDebug`.)

First build pulls Gradle wrappers + AGP — takes ~3-5 min. Subsequent
builds are seconds.

The signed debug APK lands at:

```
augmentum/cast/android-tv-receiver/app/build/outputs/apk/debug/app-debug.apk
```

Debug-signed is fine for sideload — Android TV accepts it via ADB
without registering for Google Play signing.

## 2 — Enable Developer Mode on the TV

This is one-time per device. Steps differ slightly per launcher but
follow the same pattern:

**Onn box / Google TV / Chromecast w/ Google TV:**
1. Settings → System → About
2. Scroll to "Android TV OS build" (or "Build")
3. Click it 7 times — message says "You are now a developer"
4. Back out, Settings → System → Developer Options
5. Enable **USB debugging**
6. Enable **Network debugging** (also called "Wireless debugging" on
   some builds)

**NVIDIA Shield / Sony Bravia / similar:**
Same pattern, sometimes Settings → Device Preferences → About → Build.

After enabling, note the TV's LAN IP address: Settings → Network →
look for "IP address" (e.g. `192.168.1.42`).

## 3 — Connect via ADB

From the PC where you built the APK:

```bash
adb connect 192.168.1.42:5555
```

**On the TV**, you'll see a one-time prompt: "Allow USB debugging from
this computer?" → check "Always allow" → OK.

Verify:

```bash
adb devices
# Should list:  192.168.1.42:5555    device
```

If you see `offline`, retry the connect. If you see `unauthorized`,
the TV prompt was missed — issue a fresh `adb connect` and accept on
TV.

## 4 — Install the APK

```bash
adb -s 192.168.1.42:5555 install -r \
  app/build/outputs/apk/debug/app-debug.apk
```

`-r` reinstalls when upgrading; safe on first install too.

Success looks like:

```
Performing Streamed Install
Success
```

## 5 — Configure on the TV

Open the Augmentum Receiver app from the Android TV launcher (it's
listed in the "Apps" row with a small purple **A** banner).

On first launch you'll see the config screen:

- Enter your Augmentum URL — e.g. `https://192.168.1.10:6443`
  (the host where Augmentum is running, not the TV's IP). Including
  the port is important; `https://...:6443`.
- Press OK on the remote → "Save and connect"

The app reloads into the cast-receiver page. From there pair via the
QR (scan with your phone while logged into Augmentum) and you're done.

## 6 — Verification

- **On the TV**: the receiver shell should show its idle "Ready"
  state (or the QR pair panel if not yet paired).
- **In Augmentum**: the cast shelf (bottom-right pill in the main UI)
  should list the TV as a connected receiver within ~8 seconds.

## After install

- **Reboot**: the receiver auto-launches on TV boot (BootReceiver in
  the manifest). No manual relaunch needed.
- **Re-pair**: if the cast-receiver page expires its WS auth, it'll
  show a fresh QR. Scan from the phone again.
- **Re-configure URL**: if Augmentum moves to a new host, force-close
  the receiver app (Settings → Apps → Augmentum Receiver → Force
  Stop), then relaunch — the config screen reappears.

## Troubleshooting

**`adb: device offline`**
- Reboot the TV: Settings → System → Restart
- Re-issue `adb connect <ip>:5555` after boot

**`adb: unauthorized`**
- The TV's "Allow debugging" prompt was dismissed. From the TV:
  Settings → System → Developer Options → "Revoke USB debugging
  authorizations" → reissue `adb connect` → accept the prompt this
  time.

**Receiver app shows "Loading…" forever**
- WebView can't reach the URL. Verify from a PC browser:
  `curl -k https://<augmentum-host>:6443/ui/cast-receiver/` should
  return 200. If not, Augmentum isn't reachable on that URL from
  the TV's network position.
- Self-signed certificate? Some TVs strict-reject self-signed certs
  even with `usesCleartextTraffic=true`. If you're on Caddy + self-
  signed for LAN, either accept the cert on first launch via the
  TV browser, or wire Caddy to use a LAN-trusted CA cert.

**No banner on the launcher**
- Some TV launchers cache app banners. Restart the TV (full restart,
  not standby).

## Updating the APK

After code changes:

```bash
./gradlew assembleDebug
adb -s 192.168.1.42:5555 install -r \
  app/build/outputs/apk/debug/app-debug.apk
```

`-r` keeps the user's configured URL in SharedPreferences across
upgrades. No reconfigure needed.

## What this APK does NOT do (yet)

- **No auto-update from Augmentum.** Future work: an in-app "Check
  for updates" that pulls the latest APK from Augmentum and
  installs it. Today, manual rebuild + reinstall.
- **No discovery / mDNS advertising.** The TV doesn't tell Augmentum
  "here I am" until the receiver page connects. A future revision
  will advertise via NSD on the LAN so Augmentum can auto-detect
  installed TVs.
- **No Cast Built-in launcher integration.** This APK is the
  alternative to Cast Built-in — for households where Cast App ID
  registration isn't worth it. Both paths coexist.

## File layout reference

```
augmentum/cast/android-tv-receiver/
├── app/
│   ├── build.gradle.kts                       # App module config
│   ├── proguard-rules.pro
│   └── src/main/
│       ├── AndroidManifest.xml                # Permissions, activities, banner
│       ├── java/com/augmentum/castreceiver/
│       │   ├── MainActivity.kt                # Fullscreen WebView host
│       │   ├── ConfigActivity.kt              # First-launch URL config
│       │   └── BootReceiver.kt                # Auto-launch on TV boot
│       └── res/
│           ├── layout/activity_config.xml     # D-pad-friendly form
│           ├── values/strings.xml
│           ├── values/colors.xml
│           ├── drawable/ic_launcher_foreground.xml
│           ├── drawable/tv_banner.xml         # 320x180 Leanback banner
│           └── mipmap-anydpi-v26/ic_launcher.xml
├── build.gradle.kts                           # Top-level
├── settings.gradle.kts
├── gradle.properties
└── .gitignore
```

Total: ~200 lines of code + ~150 lines of config / resources.
