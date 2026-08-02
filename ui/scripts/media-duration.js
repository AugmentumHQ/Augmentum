/**
 * media-duration.js — single source of truth for "what's the real
 * runtime of this media?" when the upstream provider may be
 * transcoding mid-stream.
 *
 * Problem this solves: Emby/Jellyfin transcode a video on demand when
 * the browser can't direct-play the audio (DTS, TrueHD, ...). During
 * transcode the upstream Content-Length reflects only the bytes
 * generated SO FAR, so the browser's `videoEl.duration` reports the
 * buffered transcode head — not the full film. Every site that reads
 * `videoEl.duration` raw will then:
 *
 *   - show "0:00 / 0:32" instead of full runtime
 *   - fill its seek bar to 100% at the buffered head
 *   - POST duration_s=32 to /api/media/progress, marking the row
 *     "finished" 30s into a 90-minute movie
 *
 * Fix: every duration consumer goes through `effectiveDuration(raw,
 * known)`, where `known` is the upstream-reported runtime saved in
 * `file_index.source_metadata.duration_s` at catalog-sync time. The
 * helper returns the larger of the two — direct-play streams get the
 * raw value (zero `known` falls through), transcode streams get the
 * authoritative upstream value.
 *
 * Used by:
 *   - ui/scripts/files/preview.js   (in-app preview/player)
 *   - ui/cast-video/cast-video.js   (TV video receiver)
 *   - ui/cast-audio/cast-audio.js   (TV audio receiver)
 */


/**
 * Pick the more trustworthy of two duration values, in seconds.
 *
 * Either argument may be falsy / non-finite / negative — that input is
 * ignored. Returns 0 only when both inputs are unusable.
 *
 * @param {number} rawS    seconds reported by the media element
 *                         (videoEl.duration / audioEl.duration)
 * @param {number} knownS  seconds reported by the upstream provider
 *                         (source_metadata.duration_s)
 * @returns {number}       max of the two clean values, or 0
 */
export function effectiveDuration(rawS, knownS) {
  const raw = Number.isFinite(rawS) && rawS > 0 ? Number(rawS) : 0;
  const known = Number.isFinite(knownS) && knownS > 0 ? Number(knownS) : 0;
  return Math.max(raw, known);
}
