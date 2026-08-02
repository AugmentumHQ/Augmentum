/*
 * BootReceiver — auto-relaunches MainActivity when the TV powers on.
 *
 * Without this, the user has to navigate to the Augmentum icon in the
 * Leanback launcher every time the TV boots. With it, the receiver
 * shows up immediately — closer to "the TV is always part of
 * Augmentum" rather than "the TV runs an app I have to remember to
 * open."
 */
package com.augmentum.castreceiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent?) {
        if (intent?.action == Intent.ACTION_BOOT_COMPLETED) {
            // Only auto-launch if a URL is configured — otherwise we'd
            // bounce the user into ConfigActivity on every boot, which
            // is annoying noise.
            val prefs = context.getSharedPreferences(
                MainActivity.PREFS_NAME,
                Context.MODE_PRIVATE,
            )
            val baseUrl = prefs.getString(MainActivity.KEY_BASE_URL, "") ?: ""
            if (baseUrl.isEmpty()) return

            val launchIntent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            context.startActivity(launchIntent)
        }
    }
}
