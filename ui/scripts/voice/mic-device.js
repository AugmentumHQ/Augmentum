/* voice/mic-device.js — shared microphone primitive.
 *
 * The mic counterpart to camera.js. Built because Augmentum's STT was
 * noticeably worse on BT headsets / USB gaming mics than on phones, and
 * the root cause was twofold:
 *   1. No device picker. Every getUserMedia({audio: …}) site (voice
 *      call, wake word, PTT, enrollment, chat mic button) used the OS
 *      default. If Windows auto-switched the default to a BT headset
 *      mid-session, the user had no way to override.
 *   2. Always-on echoCancellation/noiseSuppression/autoGainControl.
 *      These browser flags target laptop built-in mics in VoIP calls.
 *      On BT headsets the codec already AGCs; on Razer/HyperX/SteelSeries
 *      gaming mics the driver already NS's. Browser DSP on top produces
 *      compression pumping + gating + dropped consonants — exactly what
 *      a STT model is most sensitive to.
 *
 * What this module owns:
 *   - Enumerate available audio inputs (labels-only-after-permission
 *     handled via a one-shot probe, same shape as camera.js).
 *   - Persist the user's preferred mic deviceId in localStorage.
 *   - Build a getUserMedia constraint object whose AGC/NS/AEC defaults
 *     are tuned per detected device family (heuristic from track.label).
 *   - Provide one acquireMic({usage}) entrypoint that all 5 callsites
 *     (call / enrollment / wake / PTT / chat-mic-button) flow through,
 *     so we keep one source of truth for constraints.
 *   - Multiplex 'devicechange' for live-updating settings pickers.
 *
 * What this module deliberately doesn't own:
 *   - The AudioWorklet PCM resampler (still in voice.js, separate fix).
 *   - Server-side STT / VAD / DTLN pipeline.
 *   - Output device routing (lives in connect/ui.js via setSinkId).
 */

const PREF_KEY = 'augmentum.mic.preferredDeviceId';

let _labelsUnlocked = false;

// ── Device enumeration ───────────────────────────────────────────

/**
 * Enumerate available audio input devices.
 *
 * Returns `[{ deviceId, label, kind: 'audioinput' }]`. Browsers withhold
 * the `label` field until the user has granted mic permission at least
 * once on this origin — the `probeForLabels` option does a one-shot
 * getUserMedia({audio:true}) to unlock them before listing.
 */
export async function listAudioInputDevices({ probeForLabels = true } = {}) {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  if (probeForLabels && !_labelsUnlocked) {
    await _unlockLabels();
  }
  let devices;
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (_) {
    return [];
  }
  return devices
    .filter((d) => d.kind === 'audioinput' && d.deviceId)
    .map((d) => ({
      deviceId: d.deviceId,
      label: d.label || '',
      kind: 'audioinput',
    }));
}

async function _unlockLabels() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    for (const t of stream.getTracks()) {
      try { t.stop(); } catch (_) { /* defensive */ }
    }
    _labelsUnlocked = true;
  } catch (_) {
    // Permission denied or no mic — labels stay empty but the list
    // still returns. Settings UI falls back to "Microphone 1 / 2".
  }
}

// ── Preferred device persistence ─────────────────────────────────

/** Preferred deviceId from localStorage. Empty string if unset. */
export function getPreferredAudioDeviceId() {
  try {
    return String(localStorage.getItem(PREF_KEY) || '');
  } catch (_) {
    return '';
  }
}

/** Persist the user's choice. Pass '' to clear (revert to system default). */
export function setPreferredAudioDeviceId(deviceId) {
  try {
    if (deviceId) localStorage.setItem(PREF_KEY, String(deviceId));
    else localStorage.removeItem(PREF_KEY);
  } catch (_) { /* localStorage disabled — caller still has choice in-memory */ }
}

/**
 * Resolve the deviceId to actually request. Picks the explicit arg
 * first, then the persisted preference, then '' (system default).
 */
export function resolveAudioDeviceId(explicit) {
  if (explicit) return String(explicit);
  return getPreferredAudioDeviceId();
}

// ── Per-device constraint heuristic ──────────────────────────────
//
// Maps a device label substring → which browser DSP flags to suppress.
// Heuristic only (Chrome doesn't expose device class via API), so the
// matching is intentionally generous. False positives skew toward
// "raw signal" which is what STT wants anyway.

const _DEVICE_PROFILES = [
  {
    family: 'bluetooth',
    // BT codecs (HFP/HSP) already AGC at the radio layer; double-AGC pumps.
    // Some BT mics also do NS, so we suppress that too.
    match: /bluetooth|bt[\s-]|airpods|wf-|wh-|jabra|plantronics|jaybird|powerbeats|liberty/i,
    constraints: { autoGainControl: false, noiseSuppression: false, echoCancellation: true },
  },
  {
    family: 'gaming-headset',
    // Gaming headsets apply hardware NS + presence boost via their driver.
    // Browser NS strips the harmonic structure the driver enhanced.
    match: /razer|hyperx|steelseries|astro|logitech\s+g\b|corsair\s+(void|virtuoso)|nari|seiren|cloud\s+ii|arctis|game\s*one/i,
    constraints: { autoGainControl: false, noiseSuppression: false, echoCancellation: true },
  },
  {
    family: 'usb-studio-mic',
    // Studio condensers / USB mics (Blue Yeti, Shure, etc) — let the
    // signal come through clean; the user picked them for fidelity.
    match: /yeti|blue\s+snowball|shure|rode|elgato\s+wave|focusrite|presonus|samson|mxl/i,
    constraints: { autoGainControl: false, noiseSuppression: false, echoCancellation: false },
  },
  {
    family: 'webcam',
    // Webcam mics are typically far-field — keep all 3 on, they help.
    match: /(webcam|camera)\b|brio|c920|c930|streamcam|kiyo/i,
    constraints: { autoGainControl: true, noiseSuppression: true, echoCancellation: true },
  },
];

/**
 * Classify a device by its label. Returns one of:
 *   'bluetooth' | 'gaming-headset' | 'usb-studio-mic' | 'webcam' | 'default'
 *
 * Exported so the settings UI can show the family next to the device
 * name ("Razer Seiren X — gaming headset") as a sanity-check that the
 * heuristic matched what the user expects.
 */
export function classifyDeviceLabel(label) {
  const s = String(label || '');
  for (const p of _DEVICE_PROFILES) {
    if (p.match.test(s)) return p.family;
  }
  return 'default';
}

function _constraintsForLabel(label) {
  const s = String(label || '');
  for (const p of _DEVICE_PROFILES) {
    if (p.match.test(s)) return p.constraints;
  }
  // Default profile — laptop built-in / unknown. Keep the browser
  // VoIP defaults; they're tuned for this case.
  return { autoGainControl: true, noiseSuppression: true, echoCancellation: true };
}

/**
 * Build the audio constraint object for a getUserMedia call.
 *
 * @param {object} opts
 * @param {string} [opts.deviceId]    Explicit deviceId; falls back to
 *                                    the persisted preference, then ''.
 * @param {string} [opts.label]       If already known (from prior probe),
 *                                    skip the deviceId→label lookup.
 * @param {string} [opts.usage]       'streaming' | 'enrollment' | 'snapshot'.
 *                                    Influences sample rate / channel hints.
 * @returns {object} ready to use as the `audio:` key in getUserMedia
 */
export function buildAudioConstraints({ deviceId, label = '', usage = 'streaming', echoCancellation } = {}) {
  const profile = _constraintsForLabel(label);
  // echoCancellation is a tri-state override: undefined → use the device
  // profile; true/false → force it. The companion widget flips this off so
  // the browser's AEC stops low-passing the whole device output (which
  // muffles music/media while the mic is open). The profile still owns
  // NS/AGC — this override is scoped to AEC only.
  const aec = echoCancellation === undefined
    ? profile.echoCancellation
    : !!echoCancellation;
  const constraints = {
    echoCancellation: aec,
    noiseSuppression: profile.noiseSuppression,
    autoGainControl: profile.autoGainControl,
    channelCount: 1, // mono — avoids stereo downmix artifacts on every device
  };
  // Streaming + wake paths benefit from a 16 kHz ideal (skips a resample
  // on platforms that honor the hint — most browsers ignore but it's free).
  if (usage === 'streaming' || usage === 'snapshot') {
    constraints.sampleRate = { ideal: 16000 };
  }
  // deviceId as `ideal` so a stale preference (user yanked the USB mic)
  // falls back to the default instead of throwing OverconstrainedError.
  if (deviceId) {
    constraints.deviceId = { ideal: String(deviceId) };
  }
  return constraints;
}

// ── Single mic-acquisition entrypoint ───────────────────────────

/**
 * Acquire a MediaStream for microphone capture.
 *
 * Replaces the 5 prior inline getUserMedia callsites. Handles:
 *   - Resolving the user's preferred deviceId (or system default)
 *   - Applying per-device constraint heuristics from the label
 *   - Falling back to bare `audio: true` if the constraint set is rejected
 *     (mobile browsers occasionally reject the full set; mirrors the old
 *     voice.js fallback behavior)
 *   - Logging the resolved track label + settings to console for diagnostics
 *
 * Throws on permission denial, no-mic, or other terminal failures —
 * caller surfaces a user-facing error.
 *
 * @param {object} opts
 * @param {string} [opts.usage]   'streaming' | 'enrollment' | 'snapshot'
 * @param {string} [opts.deviceId]  Override; otherwise pulls from preference
 * @returns {Promise<MediaStream>}
 */
export async function acquireMic({ usage = 'streaming', deviceId, echoCancellation } = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia not supported');
  }
  const resolvedId = deviceId !== undefined ? deviceId : resolveAudioDeviceId();
  // Probe the label first so the constraint heuristic can fire on the
  // RIGHT device. Cheap — already cached after first call.
  let label = '';
  if (resolvedId) {
    try {
      const devices = await listAudioInputDevices({ probeForLabels: false });
      label = devices.find((d) => d.deviceId === resolvedId)?.label || '';
    } catch (_) { /* fall through */ }
  }

  const audio = buildAudioConstraints({ deviceId: resolvedId, label, usage, echoCancellation });
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio });
  } catch (err) {
    // Two retry paths, in order of specificity:
    //   1. deviceId constraint failed (device yanked) → retry without
    //      pinning a deviceId.
    //   2. Full constraint set rejected (some mobile browsers) → bare
    //      `audio: true`, matching the prior voice.js fallback.
    if (_isDeviceConstraintError(err) && audio.deviceId) {
      const noDevice = { ...audio };
      delete noDevice.deviceId;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: noDevice });
      } catch (_) {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
    } else {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    }
  }
  // Reconcile DSP flags against the device we ACTUALLY got. The label is
  // only knowable after permission, and for the system-default device
  // (the common case — user just connected AirPods, never opened the
  // picker) we requested with an empty label and got the default
  // AGC+NS+AEC profile. Now that the real label is in hand, correct it on
  // the live track so the Bluetooth/gaming/studio profile fires even when
  // the user never picked the device. Without this the heuristic is
  // unreachable for default devices — which is why AirPods got the full
  // browser DSP stack on top of the BT codec's, gating/mangling STT.
  await _reconcileDeviceConstraints(stream, echoCancellation);
  _logAcquired(stream, usage, resolvedId);
  return stream;
}

/**
 * Apply the label-derived DSP profile to an already-open track. Best
 * effort: applyConstraints on a live track is honored unevenly across
 * browsers, and a failure just leaves the requested constraints in place.
 * No-ops when the track already matches the profile (the explicit-pick
 * path) or when the label is still withheld.
 */
async function _reconcileDeviceConstraints(stream, echoCancellation) {
  try {
    const track = stream.getAudioTracks?.()[0];
    if (!track || typeof track.applyConstraints !== 'function') return;
    const label = track.label || '';
    if (!label) return; // label withheld — nothing to classify against
    const profile = _constraintsForLabel(label);
    // The caller's AEC override (if any) wins over the device profile here
    // too — otherwise reconcile would silently flip echoCancellation back on
    // for the default-device path, undoing the companion's "off" toggle.
    const applied = {
      noiseSuppression: profile.noiseSuppression,
      autoGainControl: profile.autoGainControl,
      echoCancellation: echoCancellation === undefined
        ? profile.echoCancellation
        : !!echoCancellation,
    };
    const cur = track.getSettings?.() || {};
    const differs = (key) =>
      cur[key] !== undefined && cur[key] !== applied[key];
    if (
      !differs('noiseSuppression') &&
      !differs('autoGainControl') &&
      !differs('echoCancellation')
    ) {
      return; // already matches (e.g. user picked the device explicitly)
    }
    await track.applyConstraints(applied);
    console.info(
      '[mic-device] reconciled DSP for default device',
      { family: classifyDeviceLabel(label), label, applied },
    );
  } catch (e) {
    console.warn(
      '[mic-device] applyConstraints reconcile failed (best-effort):',
      e?.name || e,
    );
  }
}

function _isDeviceConstraintError(err) {
  const name = String(err?.name || '');
  return name === 'OverconstrainedError' || name === 'NotFoundError';
}

function _logAcquired(stream, usage, requestedDeviceId) {
  try {
    const track = stream.getAudioTracks?.()[0];
    if (!track) return;
    const settings = track.getSettings?.() || {};
    // Single source of truth for "what mic are we actually using" — every
    // mic-acquiring callsite goes through here so devtools always shows
    // the right thing for the active session.
    console.info('[mic-device] acquired', {
      usage,
      label: track.label || '(no label)',
      family: classifyDeviceLabel(track.label),
      requested_device_id: requestedDeviceId
        ? requestedDeviceId.slice(0, 12) + '…' : '(default)',
      actual_device_id: settings.deviceId
        ? String(settings.deviceId).slice(0, 12) + '…' : '(none)',
      sample_rate: settings.sampleRate,
      channels: settings.channelCount,
      echo_cancellation: settings.echoCancellation,
      noise_suppression: settings.noiseSuppression,
      auto_gain: settings.autoGainControl,
    });
  } catch (_) { /* logging is best-effort */ }
}

/**
 * Pull the human-readable label off an already-open stream. Used by
 * the voice overlay to show "Razer Seiren X" so the user sees which
 * mic Chrome actually honored.
 */
export function streamMicLabel(stream) {
  try {
    return stream?.getAudioTracks?.()[0]?.label || '';
  } catch (_) {
    return '';
  }
}

// ── Device-change subscription ──────────────────────────────────

const _deviceChangeListeners = new Set();
let _deviceChangeWired = false;

/**
 * Subscribe to audio-device-change events (BT pair/unpair, USB plug,
 * default switch). Returns an unsubscribe function. Use in settings UI
 * to live-refresh the mic picker without polling.
 */
export function onAudioDeviceChange(fn) {
  if (typeof fn !== 'function') return () => {};
  _deviceChangeListeners.add(fn);
  if (!_deviceChangeWired && navigator.mediaDevices?.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', _fanOutDeviceChange);
    _deviceChangeWired = true;
  }
  return () => _deviceChangeListeners.delete(fn);
}

function _fanOutDeviceChange() {
  for (const fn of _deviceChangeListeners) {
    try { fn(); } catch (_) { /* defensive */ }
  }
}
