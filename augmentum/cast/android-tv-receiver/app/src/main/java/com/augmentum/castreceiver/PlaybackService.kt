/*
 * PlaybackService — foreground service hosting the MediaSession.
 *
 * Why a service: a MediaSession needs a stable owner that outlives any
 * one Activity teardown. If we put the session on MainActivity, every
 * configuration change / momentary unfocus would tear it down and the
 * system would lose its "now playing" anchor mid-cast. A foreground
 * service is the canonical place for this on Android.
 *
 * Why foreground (vs. background): Android kills background services
 * aggressively on Android TV. While the user is casting, we need to
 * stay alive to keep the MediaSession registered and to receive media
 * button intents from the remote. The service displays a low-priority
 * notification (required for foreground type) summarising what's
 * casting; on Android TV that notification is invisible to the user
 * but counts toward the OS's "this app is doing something legit" check.
 *
 * The session itself is held in a process-static singleton
 * (``MediaSessionHolder``) so the JS bridge can update metadata /
 * playback state without going through a Service binding round-trip
 * for every JS call. There's only ever one PlaybackService instance
 * in the app, so the singleton is safe.
 */
package com.augmentum.castreceiver

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.os.Build
import android.os.IBinder
import android.support.v4.media.MediaMetadataCompat
import android.support.v4.media.session.MediaSessionCompat
import android.support.v4.media.session.PlaybackStateCompat
import androidx.core.app.NotificationCompat
import androidx.media.session.MediaButtonReceiver

class PlaybackService : Service() {

    companion object {
        const val CHANNEL_ID = "augmentum_cast_playback"
        const val NOTIF_ID = 4271
        const val SESSION_TAG = "AugmentumCastReceiver"

        // Convenience: start the service from anywhere. Idempotent —
        // the system de-dupes startService calls for the same component.
        fun start(context: Context) {
            val intent = Intent(context, PlaybackService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, PlaybackService::class.java))
        }
    }

    private lateinit var session: MediaSessionCompat

    override fun onCreate() {
        super.onCreate()
        ensureChannel()

        // Build the session with PLAY_PAUSE pre-handled so the
        // MediaButtonReceiver registered in the manifest can route
        // remote keys to us even when the activity is not foreground.
        session = MediaSessionCompat(this, SESSION_TAG).apply {
            setFlags(
                MediaSessionCompat.FLAG_HANDLES_MEDIA_BUTTONS or
                MediaSessionCompat.FLAG_HANDLES_TRANSPORT_CONTROLS,
            )
            setCallback(MediaCallback())
            // Start in STOPPED with a baseline action set so the
            // system recognises the session as a real receiver. JS
            // will refine this once a cast actually mounts.
            setPlaybackState(
                PlaybackStateCompat.Builder()
                    .setActions(BASE_ACTIONS)
                    .setState(PlaybackStateCompat.STATE_STOPPED, 0L, 1f)
                    .build(),
            )
            isActive = true
        }
        MediaSessionHolder.attach(session)

        startForeground(NOTIF_ID, buildNotification("Augmentum", "Ready to cast"))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Forward media button intents to the session so the
        // MediaButtonReceiver in the manifest can dispatch transport
        // key events without owning the session itself.
        MediaButtonReceiver.handleIntent(session, intent)
        return START_STICKY
    }

    override fun onDestroy() {
        MediaSessionHolder.detach()
        session.isActive = false
        session.release()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val mgr = getSystemService(NotificationManager::class.java) ?: return
        if (mgr.getNotificationChannel(CHANNEL_ID) != null) return
        val ch = NotificationChannel(
            CHANNEL_ID,
            "Cast playback",
            // LOW so the notification doesn't make sound or vibrate —
            // it exists only to satisfy the foreground-service contract.
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Active when Augmentum is casting media to this TV."
            setShowBadge(false)
            enableLights(false)
            enableVibration(false)
        }
        mgr.createNotificationChannel(ch)
    }

    /**
     * Build (or rebuild) the foreground notification.
     *
     * We use MediaStyle so the system surfaces this on the TV remote /
     * Google Assistant card with proper transport controls instead of
     * a generic "running service" badge. Title + subtitle come from
     * whatever the JS layer last pushed.
     */
    fun buildNotification(title: String, subtitle: String): Notification {
        // Tap the notification → launch MainActivity. On TV the user
        // can't tap, but Google's "Now Playing" surface uses this
        // pending intent to deep-link back to us.
        val contentPi = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
            },
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val style = androidx.media.app.NotificationCompat.MediaStyle()
            .setMediaSession(session.sessionToken)
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(title)
            .setContentText(subtitle)
            .setContentIntent(contentPi)
            .setStyle(style)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setOngoing(true)
            .setColor(Color.parseColor("#d9a14e"))
            .build()
    }

    /**
     * MediaSession callback. Each transport event from the system
     * (remote, Assistant, system tray) lands here; we hand it off to
     * the JS bridge so cast-receiver.js can forward it as a patch to
     * whichever surface is currently mounted.
     *
     * The bridge keeps a JS callback registered via setRemoteHandler;
     * if none is registered yet we drop the event silently — the cast
     * isn't observing remote events anyway.
     */
    private inner class MediaCallback : MediaSessionCompat.Callback() {
        override fun onPlay() { MediaSessionHolder.dispatchRemote("play", 0L) }
        override fun onPause() { MediaSessionHolder.dispatchRemote("pause", 0L) }
        override fun onStop() { MediaSessionHolder.dispatchRemote("stop", 0L) }
        override fun onSkipToNext() { MediaSessionHolder.dispatchRemote("next", 0L) }
        override fun onSkipToPrevious() { MediaSessionHolder.dispatchRemote("previous", 0L) }
        override fun onFastForward() { MediaSessionHolder.dispatchRemote("ffwd", 0L) }
        override fun onRewind() { MediaSessionHolder.dispatchRemote("rewind", 0L) }
        override fun onSeekTo(pos: Long) { MediaSessionHolder.dispatchRemote("seek_ms", pos) }
    }
}

// Default action set the session advertises when nothing's playing.
// The cast surface narrows this down via the bridge once a real item
// mounts (e.g. comic surfaces drop the seek action, image surfaces
// drop everything).
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


/**
 * Process-static handle to the MediaSession.
 *
 * The bridge needs to update metadata + state without owning a Service
 * binding. There's exactly one PlaybackService per process, so a
 * static reference is the simplest stable hand-off.
 *
 * ``dispatchRemote`` is the callback path the service uses to surface
 * transport events back to the JS layer; the bridge registers a JS-
 * callback via setRemoteHandler when cast-receiver.js is ready.
 */
object MediaSessionHolder {

    @Volatile private var session: MediaSessionCompat? = null
    @Volatile private var remoteHandler: ((String, Long) -> Unit)? = null

    fun attach(s: MediaSessionCompat) { session = s }
    fun detach() { session = null }

    fun current(): MediaSessionCompat? = session

    fun setRemoteHandler(cb: ((String, Long) -> Unit)?) { remoteHandler = cb }
    fun dispatchRemote(action: String, value: Long) {
        remoteHandler?.invoke(action, value)
    }
}
