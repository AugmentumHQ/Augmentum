/* connect/quality.js — encoder bandwidth adaptation.
 *
 * Both dialer.js and incoming.js run a 2s quality poll over
 * pc.getStats() and bucket samples into measuring / excellent / good /
 * weak / poor. This module turns those buckets into RTCRtpSender
 * parameters: when quality degrades, cap maxBitrate + drop framerate
 * + downscale resolution; when it recovers, lift the caps so video
 * tracks return to full quality without manual intervention.
 *
 * Why this lives here and not inside each session: the policy must
 * stay symmetric (caller + receiver send + receive the same call;
 * if their adaptation curves disagree, one side over-restricts and
 * the other does nothing). One file = one curve.
 *
 * What this module deliberately doesn't do:
 *   - Simulcast / SVC negotiation. Those need SDP munging and a
 *     receiver that actually consumes multiple spatial layers.
 *     v1 just drops the single layer's quality.
 *   - Simulcast for audio. There is one audio layer; the knob is bitrate.
 *   - Per-bucket policy persistence. The user doesn't get to tune
 *     these — the curve is opinionated. Power-user knob is a backlog
 *     item once we have telemetry.
 */

// Bitrate + resolution caps per bucket. maxBitrate is in bits/sec.
// scaleResolutionDownBy >1 tells the encoder to downscale before
// encoding; the receiver's <video> element scales back up. This is
// the cheapest knob — saves both encode + bandwidth — and is what
// Meet / Discord lean on hardest during congestion.
//
// Excellent ceiling sized for ~720p30 with the FaceTime-style
// fullscreen render the in-call overlay does. FaceTime itself targets
// ~5 Mbps on a strong connection for HD; 4 Mbps is the
// "looks-sharp-without-saturating-LAN-uploads" sweet spot for the
// home/LAN dogfood loop. The degradation curve below it is unchanged.
const VIDEO_PROFILES = {
  excellent: { maxBitrate: 4_000_000, scaleResolutionDownBy: 1,   maxFramerate: 30 },
  good:      { maxBitrate: 2_500_000, scaleResolutionDownBy: 1,   maxFramerate: 30 },
  weak:      { maxBitrate:   900_000, scaleResolutionDownBy: 1.5, maxFramerate: 24 },
  poor:      { maxBitrate:   300_000, scaleResolutionDownBy: 2,   maxFramerate: 15 },
};

const DEFAULT_PROFILE = VIDEO_PROFILES.excellent;

// Audio target bitrate per bucket, bits/sec, applied to the single Opus
// encoding.
//
// This exists because the BROWSER DEFAULT is the real ceiling, and it is
// low: Chrome negotiates mono Opus at roughly 32 kbps for a WebRTC audio
// track unless the application asks for more. Nothing in Connect was
// asking, so every call — including a gigabit LAN call between two
// machines in the same house — ran at the congested-cellular default and
// sounded thin. We were not chopping the audio anywhere; we were simply
// accepting a bandwidth-conservative default that the link never needed.
//
// 96k mono Opus is transparent for speech and has headroom for music or a
// shared-screen soundtrack. The floor is 24k: below that Opus starts
// trading intelligibility for bitrate, which is the wrong trade on a call
// where the words are the entire payload. Note the curve is much flatter
// than the video one on purpose — audio is ~2% of a video call's
// bandwidth, so starving it buys almost nothing and costs the thing the
// user actually came for. Video degrades first, and by a lot; audio only
// gives ground once the link is genuinely poor.
const AUDIO_PROFILES = {
  excellent: 96_000,
  good:      64_000,
  weak:      40_000,
  poor:      24_000,
};

const DEFAULT_AUDIO_BITRATE = AUDIO_PROFILES.excellent;

/**
 * Raise the audio sender off the browser default and adapt it to `bucket`.
 *
 * Takes the RTCPeerConnection rather than a sender: unlike video (which
 * is added, removed, and replaced through escalation and screen-share, so
 * the caller already tracks its sender), the audio sender is created once
 * at setup and never swapped. Looking it up here means no new variable to
 * keep in sync across those paths — one fewer thing that can go stale.
 *
 * Returns the bitrate applied, or null if there was no audio sender or
 * setParameters refused.
 */
export async function applyAudioQualityProfile(pc, bucket) {
  if (!pc || typeof pc.getSenders !== 'function') return null;
  let sender = null;
  try {
    sender = pc.getSenders().find((s) => s.track && s.track.kind === 'audio') || null;
  } catch (_) { return null; }
  if (!sender) return null;

  const bitrate = AUDIO_PROFILES[bucket] || DEFAULT_AUDIO_BITRATE;

  let params;
  try { params = sender.getParameters(); }
  catch (_) { return null; }
  if (!params.encodings || params.encodings.length === 0) {
    params.encodings = [{}];
  }
  params.encodings[0].maxBitrate = bitrate;
  // Voice is the payload of a call; if the browser has to choose what to
  // starve under congestion, it should not be this stream.
  params.encodings[0].networkPriority = 'high';
  params.encodings[0].priority = 'high';

  try {
    await sender.setParameters(params);
  } catch (_) {
    // Same recoverable failures as the video path (stale transactionId,
    // not-yet-negotiated sender). The next poll retries.
    return null;
  }
  return bitrate;
}

/** Snapshot of the audio curve, for UI/telemetry symmetry with video. */
export function getAudioBitrate(bucket) {
  return AUDIO_PROFILES[bucket] || DEFAULT_AUDIO_BITRATE;
}

/**
 * Apply the bitrate / resolution / framerate cap for `bucket` to the
 * given RTCRtpSender. Safe to call with bucket='measuring' (treated
 * as excellent — don't pre-emptively penalise a fresh call).
 *
 * Returns the profile that was applied (or null if no sender / not
 * a video sender). Callers can log or emit this for telemetry.
 */
export async function applyVideoQualityProfile(sender, bucket) {
  if (!sender || !sender.track || sender.track.kind !== 'video') return null;
  const profile = VIDEO_PROFILES[bucket] || DEFAULT_PROFILE;

  // RTCRtpSender.getParameters can throw on senders that haven't
  // been negotiated yet — early in the call before setLocalDescription
  // resolves. Defensive: bail rather than fail the quality emit.
  let params;
  try { params = sender.getParameters(); }
  catch (_) { return null; }

  if (!params.encodings || params.encodings.length === 0) {
    // setParameters requires at least one encoding entry. Some
    // browsers omit it for newly-added tracks; fill in a minimal one.
    params.encodings = [{}];
  }
  // Single-layer for v1 — we apply the profile to encoding[0]. When
  // we ship simulcast, this becomes a per-layer policy.
  const enc = params.encodings[0];
  enc.maxBitrate = profile.maxBitrate;
  enc.scaleResolutionDownBy = profile.scaleResolutionDownBy;
  enc.maxFramerate = profile.maxFramerate;

  try {
    await sender.setParameters(params);
  } catch (err) {
    // Firefox historically threw on setParameters with a stale
    // transactionId; Chrome with InvalidStateError on senders
    // that haven't been negotiated. Both are recoverable on the
    // next poll. Don't escalate.
    return null;
  }
  return profile;
}

/** Snapshot of the profile table — useful for the UI to surface the
 *  current cap (e.g. "Video reduced — weak connection"). */
export function getVideoProfile(bucket) {
  return VIDEO_PROFILES[bucket] || DEFAULT_PROFILE;
}
