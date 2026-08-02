/*
 * AugmentumTvBridge — JS bridge for system-level controls.
 *
 * The cast-receiver WebView is a sandboxed web context and can't
 * reach Android's AudioManager directly. This bridge gives the
 * cast-receiver page a thin, well-typed surface for things the
 * Android shell needs to do on its behalf — today system volume,
 * later screen brightness / power state / wakelock toggles as the
 * cast vocabulary grows.
 *
 * Wire-up in MainActivity:
 *
 *     webView.settings.javaScriptEnabled = true
 *     webView.addJavascriptInterface(
 *         AugmentumTvBridge(this),
 *         AugmentumTvBridge.JS_NAME,
 *     )
 *
 * On the web side, ``window.AugmentumTV`` is the bridge surface.
 * Pages MUST guard with a typeof check — the bridge only exists in
 * the bundled APK; cast-receiver loaded in a desktop browser tab
 * sees ``window.AugmentumTV === undefined`` and falls back to the
 * in-page <video>.volume slider.
 */
package com.augmentum.castreceiver

import android.content.Context
import android.media.AudioManager
import android.os.Handler
import android.os.Looper
import android.support.v4.media.MediaMetadataCompat
import android.support.v4.media.session.PlaybackStateCompat
import android.webkit.JavascriptInterface
import org.json.JSONObject

class AugmentumTvBridge(context: Context) {

    companion object {
        const val JS_NAME = "AugmentumTV"
    }

    // STREAM_MUSIC is the right stream for casted media — same one
    // YouTube / Plex / Disney+ etc. tag their playback against. Using
    // STREAM_SYSTEM (the global ringer-style stream) would clobber
    // notification beeps; STREAM_VOICE_CALL is reserved for telephony.
    private val audio: AudioManager =
        context.applicationContext.getSystemService(Context.AUDIO_SERVICE)
            as AudioManager

    // MediaSession calls must land on the main thread; JS bridge calls
    // arrive on a binder thread. Bounce everything through this.
    private val main = Handler(Looper.getMainLooper())

    /**
     * Set the system music-stream volume as a fraction 0.0..1.0.
     *
     * Returns the actual fraction the system ended up at — Android
     * snaps to integer step counts (typically 15-25 steps on a TV),
     * so a 0.37 request becomes 0.40 / 0.33 / whatever the nearest
     * snap point is. Caller can echo the snapped value back to the
     * controller UI so the slider mirrors reality.
     */
    @JavascriptInterface
    fun setSystemVolume(fraction: Double): Double {
        val safe = fraction.coerceIn(0.0, 1.0)
        val max = audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC).coerceAtLeast(1)
        val target = (safe * max).toInt().coerceIn(0, max)
        audio.setStreamVolume(AudioManager.STREAM_MUSIC, target, 0)
        return target.toDouble() / max.toDouble()
    }

    /**
     * Current system music-stream volume as a fraction 0.0..1.0.
     * Used by the bridge-aware web layer to seed the controller
     * slider on first paint so it doesn't start at a wrong default.
     */
    @JavascriptInterface
    fun getSystemVolume(): Double {
        val max = audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC).coerceAtLeast(1)
        val cur = audio.getStreamVolume(AudioManager.STREAM_MUSIC)
        return cur.toDouble() / max.toDouble()
    }

    /**
     * Single-step adjust — equivalent to a hardware volume key press.
     * ``delta`` is the number of steps to move (positive raises,
     * negative lowers). Returns the new level as a 0.0..1.0 fraction.
     *
     * Useful for future TV-remote-key handlers if/when we wire those:
     * pressing VOL+ on the Onn remote would translate to
     * adjustVolume(+1) without the web layer having to know the step
     * count.
     */
    @JavascriptInterface
    fun adjustVolume(delta: Int): Double {
        val direction = when {
            delta > 0 -> AudioManager.ADJUST_RAISE
            delta < 0 -> AudioManager.ADJUST_LOWER
            else -> AudioManager.ADJUST_SAME
        }
        val steps = kotlin.math.abs(delta).coerceAtMost(20)
        repeat(steps) {
            audio.adjustStreamVolume(AudioManager.STREAM_MUSIC, direction, 0)
        }
        return getSystemVolume()
    }

    /** Mute / unmute the music stream. Returns the new state. */
    @JavascriptInterface
    fun setSystemMuted(muted: Boolean): Boolean {
        audio.adjustStreamVolume(
            AudioManager.STREAM_MUSIC,
            if (muted) AudioManager.ADJUST_MUTE else AudioManager.ADJUST_UNMUTE,
            0,
        )
        return isSystemMuted()
    }

    /** True when the music stream is currently muted. */
    @JavascriptInterface
    fun isSystemMuted(): Boolean = audio.isStreamMute(AudioManager.STREAM_MUSIC)

    /**
     * Cheap capability probe the web layer can use to decide whether
     * to surface the TV-master slider. Always returns true on the
     * Android TV APK; web layers in non-bridged contexts get
     * ``undefined`` because the entire bridge object is absent.
     */
    @JavascriptInterface
    fun hasSystemVolume(): Boolean = true


    /* ── MediaSession metadata + state ─────────────────────────────
     *
     * JS calls these whenever the cast surface_state changes so the
     * TV remote, the system now-playing card, and Google Assistant
     * all see the same "what's playing" picture. The bridge silently
     * no-ops when the PlaybackService isn't attached yet (race during
     * boot — happens once, the next surface_state catches up).
     */

    /**
     * Update the now-playing metadata.
     *
     * ``metaJson`` is a JSON string with optional fields:
     *   - title:       primary label ("Inception", "Episode 3", ...)
     *   - subtitle:    secondary label ("2010 · 2h 28m", "S1 · Ep 3", ...)
     *   - artist:      author / show / channel — displayed beneath title
     *   - duration_ms: track length, used for the seek bar
     *   - cover_url:   poster / thumbnail URL (NOT fetched here — the
     *                  system will request it via the content URI if
     *                  needed; for simplicity we just pass the string
     *                  and rely on the system fetching cached bytes
     *                  from a future content provider iteration)
     *
     * Passing JSON instead of multiple positional args keeps the bridge
     * stable as we add fields — callers omit what they don't have.
     */
    @JavascriptInterface
    fun setNowPlaying(metaJson: String) {
        val obj = try { JSONObject(metaJson) } catch (_: Throwable) { return }
        main.post {
            val session = MediaSessionHolder.current() ?: return@post
            val builder = MediaMetadataCompat.Builder()
            obj.optString("title").takeIf { it.isNotEmpty() }?.let {
                builder.putString(MediaMetadataCompat.METADATA_KEY_TITLE, it)
                builder.putString(MediaMetadataCompat.METADATA_KEY_DISPLAY_TITLE, it)
            }
            obj.optString("subtitle").takeIf { it.isNotEmpty() }?.let {
                builder.putString(MediaMetadataCompat.METADATA_KEY_DISPLAY_SUBTITLE, it)
            }
            obj.optString("artist").takeIf { it.isNotEmpty() }?.let {
                builder.putString(MediaMetadataCompat.METADATA_KEY_ARTIST, it)
            }
            val dur = obj.optLong("duration_ms", -1L)
            if (dur > 0) builder.putLong(MediaMetadataCompat.METADATA_KEY_DURATION, dur)
            obj.optString("cover_url").takeIf { it.isNotEmpty() }?.let {
                builder.putString(MediaMetadataCompat.METADATA_KEY_ART_URI, it)
                builder.putString(MediaMetadataCompat.METADATA_KEY_ALBUM_ART_URI, it)
            }
            session.setMetadata(builder.build())
        }
    }

    /**
     * Update transport state. ``stateJson`` fields:
     *   - state:       "playing" | "paused" | "buffering" | "stopped"
     *   - position_ms: current playback position
     *   - speed:       playback rate (1.0 = normal, 0 when paused/stopped)
     *   - actions:     optional array of allowed transport verbs the
     *                  surface supports — narrows the system's available
     *                  buttons (e.g. comic surfaces drop "seek").
     *                  Recognised values: play, pause, stop, next, prev,
     *                  ffwd, rewind, seek. Omit or pass an empty array
     *                  to keep the broad default set.
     */
    @JavascriptInterface
    fun setPlaybackState(stateJson: String) {
        val obj = try { JSONObject(stateJson) } catch (_: Throwable) { return }
        main.post {
            val session = MediaSessionHolder.current() ?: return@post
            val name = obj.optString("state", "stopped").lowercase()
            val pos = obj.optLong("position_ms", 0L)
            val speedRaw = obj.optDouble("speed", 1.0)
            val systemState = when (name) {
                "playing" -> PlaybackStateCompat.STATE_PLAYING
                "paused" -> PlaybackStateCompat.STATE_PAUSED
                "buffering" -> PlaybackStateCompat.STATE_BUFFERING
                "stopped" -> PlaybackStateCompat.STATE_STOPPED
                else -> PlaybackStateCompat.STATE_NONE
            }
            val speed = if (systemState == PlaybackStateCompat.STATE_PLAYING) {
                speedRaw.toFloat().coerceIn(0.25f, 4f)
            } else {
                0f
            }
            val actions = obj.optJSONArray("actions")
            val actionMask = if (actions == null || actions.length() == 0) {
                BASE_ACTIONS
            } else {
                actionsFromArray(actions)
            }
            session.setPlaybackState(
                PlaybackStateCompat.Builder()
                    .setActions(actionMask)
                    .setState(systemState, pos, speed)
                    .build(),
            )
        }
    }

    /** Clear the session — the cast surface was closed / nothing's playing. */
    @JavascriptInterface
    fun clearNowPlaying() {
        main.post {
            val session = MediaSessionHolder.current() ?: return@post
            session.setMetadata(MediaMetadataCompat.Builder().build())
            session.setPlaybackState(
                PlaybackStateCompat.Builder()
                    .setActions(BASE_ACTIONS)
                    .setState(PlaybackStateCompat.STATE_STOPPED, 0L, 0f)
                    .build(),
            )
        }
    }

    /** Probe so the JS layer can branch on whether MediaSession is wired. */
    @JavascriptInterface
    fun hasMediaSession(): Boolean = MediaSessionHolder.current() != null
}


/** Map a JSON array of action names to the PlaybackStateCompat bitmask. */
private fun actionsFromArray(arr: org.json.JSONArray): Long {
    var mask = 0L
    for (i in 0 until arr.length()) {
        val name = arr.optString(i, "").lowercase()
        mask = mask or when (name) {
            "play" -> PlaybackStateCompat.ACTION_PLAY
            "pause" -> PlaybackStateCompat.ACTION_PAUSE
            "play_pause" -> PlaybackStateCompat.ACTION_PLAY_PAUSE
            "stop" -> PlaybackStateCompat.ACTION_STOP
            "next" -> PlaybackStateCompat.ACTION_SKIP_TO_NEXT
            "prev", "previous" -> PlaybackStateCompat.ACTION_SKIP_TO_PREVIOUS
            "ffwd", "fast_forward" -> PlaybackStateCompat.ACTION_FAST_FORWARD
            "rewind" -> PlaybackStateCompat.ACTION_REWIND
            "seek", "seek_to" -> PlaybackStateCompat.ACTION_SEEK_TO
            else -> 0L
        }
    }
    // Always allow play_pause if either play or pause is allowed so
    // the system can deliver the combined remote key.
    val playOrPause = PlaybackStateCompat.ACTION_PLAY or PlaybackStateCompat.ACTION_PAUSE
    if (mask and playOrPause != 0L) {
        mask = mask or PlaybackStateCompat.ACTION_PLAY_PAUSE
    }
    return mask
}

private const val BASE_ACTIONS =
    PlaybackStateCompat.ACTION_PLAY or
        PlaybackStateCompat.ACTION_PAUSE or
        PlaybackStateCompat.ACTION_PLAY_PAUSE or
        PlaybackStateCompat.ACTION_STOP or
        PlaybackStateCompat.ACTION_SKIP_TO_NEXT or
        PlaybackStateCompat.ACTION_SKIP_TO_PREVIOUS or
        PlaybackStateCompat.ACTION_FAST_FORWARD or
        PlaybackStateCompat.ACTION_REWIND or
        PlaybackStateCompat.ACTION_SEEK_TO
