/*
 * MainActivity — fullscreen WebView host.
 *
 * Reads the configured Augmentum URL from SharedPreferences. If unset,
 * routes to ConfigActivity so the user can enter it via the on-screen
 * keyboard (we keep config text-only — QR pairing happens INSIDE the
 * loaded receiver page; Augmentum handles auth, not us).
 *
 * Once configured, opens "{base_url}/ui/cast-receiver/" in a WebView
 * that's deliberately permissive: autoplay, media playback, mixed
 * content. The receiver page is same-origin to whatever it loads
 * (cast-vrm, cast-tv-demo, etc.) so cookies + WS auth Just Work.
 */
package com.augmentum.castreceiver

import android.content.Intent
import android.content.SharedPreferences
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.KeyEvent
import android.view.WindowManager
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity
import java.util.UUID

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var prefs: SharedPreferences

    companion object {
        const val PREFS_NAME = "augmentum_receiver"
        const val KEY_BASE_URL = "base_url"
        const val KEY_DEVICE_ID = "device_id"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        val baseUrl = prefs.getString(KEY_BASE_URL, "") ?: ""

        if (baseUrl.isEmpty()) {
            // First launch — bounce to config. ConfigActivity persists
            // the URL then restarts MainActivity.
            startActivity(Intent(this, ConfigActivity::class.java))
            finish()
            return
        }

        UpdateChecker.checkAndInstall(this, baseUrl)

        webView = WebView(this).apply {
            layoutParams = android.view.ViewGroup.LayoutParams(
                android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            )
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                mediaPlaybackRequiresUserGesture = false
                allowFileAccess = false
                allowContentAccess = false
                cacheMode = WebSettings.LOAD_DEFAULT
                mixedContentMode = WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE
                userAgentString = "$userAgentString AugmentumTVReceiver/1.0"
            }
            webViewClient = ReceiverWebViewClient(baseUrl) { reconfigure() }
            webChromeClient = WebChromeClient()
            // System-level controls (volume today, brightness / power
            // later) need to reach Android APIs the WebView itself
            // can't touch. The bridge exposes a narrow
            // ``window.AugmentumTV`` surface that cast-receiver.js
            // feature-detects — browsers without the bridge silently
            // fall back to in-page <video>.volume control.
            addJavascriptInterface(
                AugmentumTvBridge(this@MainActivity),
                AugmentumTvBridge.JS_NAME,
            )
        }
        // Explicitly enable cookie acceptance — first-party (same-origin
        // iframes for surfaces) and third-party (defensive, in case a
        // future surface is served from a sibling host). Default is on
        // but some OEM TV builds ship with it off. Without this the
        // session cookie set by establish-session never persists, so
        // iframe surfaces 401 on every /api call.
        CookieManager.getInstance().setAcceptCookie(true)
        CookieManager.getInstance().setAcceptThirdPartyCookies(webView, true)
        setContentView(webView)
        webView.loadUrl(receiverUrl(baseUrl))

        // Boot the MediaSession-owning foreground service. Doing this
        // here (rather than in App.onCreate) keeps the service tied to
        // the activity's lifecycle — the TV-remote / system-tray now-
        // playing card lives only when the receiver is mounted. When
        // the user backs all the way out of the app we tear it down
        // in onDestroy so we're not advertising a stale session.
        PlaybackService.start(this)

        // Wire remote-key callbacks from the MediaSession back into
        // the WebView. The service's MediaCallback delivers events
        // here on the main thread; we forward them as a synthetic JS
        // event so cast-receiver.js can translate them into patches
        // for whichever surface is currently mounted.
        MediaSessionHolder.setRemoteHandler { action, value ->
            // evaluateJavascript MUST run on the WebView's thread.
            // ``post`` schedules onto the UI thread, which is also
            // the WebView's thread for this single-Activity setup.
            //
            // We invoke a standalone global (``__augmentumOnRemote``)
            // rather than monkey-patching ``window.AugmentumTV._onRemote``
            // — the bridge object is a Java-injected proxy and its
            // property bag isn't a safe place to attach JS-defined
            // callbacks across WebView versions. The standalone global
            // is owned by cast-receiver.js, no ambiguity.
            webView.post {
                val safeAction = action.replace("'", "\\'")
                webView.evaluateJavascript(
                    "typeof window.__augmentumOnRemote === 'function' && " +
                        "window.__augmentumOnRemote('$safeAction', $value)",
                    null,
                )
            }
        }
    }

    /**
     * Build the receiver URL with device identity baked in as query
     * params. The receiver page reads these on connect and forwards
     * them in its ``ready`` event so the server can bind this APK
     * instance to a persistent ``trusted_receivers`` row — meaning the
     * user's chosen name + revocation state survive TV reboots.
     *
     * device_id is generated once on first launch and pinned in
     * SharedPreferences. The user can wipe it via app-data reset
     * (intentional — that's the "factory reset" path for re-pairing).
     */
    private fun receiverUrl(baseUrl: String): String {
        val trimmed = baseUrl.trimEnd('/')
        val deviceId = deviceId()
        val builder = Uri.parse("$trimmed/ui/cast-receiver/").buildUpon()
            .appendQueryParameter("device_id", deviceId)
            .appendQueryParameter("platform", "android-tv")
            .appendQueryParameter("label", defaultLabel())
        return builder.build().toString()
    }

    private fun deviceId(): String {
        val existing = prefs.getString(KEY_DEVICE_ID, "") ?: ""
        if (existing.isNotEmpty()) return existing
        val fresh = UUID.randomUUID().toString()
        prefs.edit().putString(KEY_DEVICE_ID, fresh).apply()
        return fresh
    }

    /**
     * Default label hint the receiver page forwards to the server on
     * first connect. The user can rename later from the Manage TVs UI
     * — once set, the server-side label wins. We just want the
     * initial row to read something better than the platform string.
     */
    private fun defaultLabel(): String {
        val model = Build.MODEL?.takeIf { it.isNotBlank() } ?: "Android TV"
        // Trim manufacturer prefix duplication ("ONN onn 4K Box" → "onn 4K Box").
        val manufacturer = Build.MANUFACTURER ?: ""
        return if (manufacturer.isNotBlank() &&
                   !model.lowercase().startsWith(manufacturer.lowercase())) {
            "$manufacturer $model"
        } else {
            model
        }
    }

    private fun reconfigure() {
        // Called on persistent load failure — bounce back to config so
        // the user can fix the URL without reinstalling.
        prefs.edit().remove(KEY_BASE_URL).apply()
        startActivity(Intent(this, ConfigActivity::class.java))
        finish()
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        // TV remote BACK should not close the app — let WebView absorb
        // it where useful, but otherwise consume so the user can't
        // accidentally drop out of the receiver. Long-press HOME on
        // the remote still works for app switching.
        if (keyCode == KeyEvent.KEYCODE_BACK) {
            if (::webView.isInitialized && webView.canGoBack()) {
                webView.goBack()
                return true
            }
            return true   // swallow rather than exit
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onDestroy() {
        if (::webView.isInitialized) {
            webView.destroy()
        }
        // Drop the remote handler before the service goes away — the
        // closure holds a reference to ``webView`` which is dying.
        MediaSessionHolder.setRemoteHandler(null)
        PlaybackService.stop(this)
        super.onDestroy()
    }
}


/**
 * WebView client that detects persistent load failures and bounces
 * the user back to the config screen. Receiver pages are always
 * online — if we can't reach the base URL after a few retries,
 * the URL is wrong or Augmentum is offline. Better to surface that
 * than spin in a broken WebView forever.
 *
 * Also accepts the self-signed cert that Caddy mints for LAN hosts
 * — but ONLY for the configured Augmentum host. Any other host's
 * SSL error is rejected, so a stray off-host link can't piggyback
 * on our trust. The receiver page is self-contained, so the
 * WebView should never need to leave the configured host anyway.
 */
private class ReceiverWebViewClient(
    private val baseUrl: String,
    private val onPersistentFailure: () -> Unit,
) : WebViewClient() {

    private var consecutiveFailures = 0
    private val failureThreshold = 5
    private val trustedHost: String? = run {
        try {
            val u = android.net.Uri.parse(baseUrl)
            val host = u.host ?: return@run null
            val port = if (u.port > 0) ":${u.port}" else ""
            "$host$port"
        } catch (_: Exception) { null }
    }

    override fun onPageFinished(view: WebView?, url: String?) {
        super.onPageFinished(view, url)
        consecutiveFailures = 0
    }

    override fun onReceivedError(
        view: WebView?,
        request: android.webkit.WebResourceRequest?,
        error: android.webkit.WebResourceError?,
    ) {
        // Sub-resource errors (images, fonts) don't count — only the
        // main frame failing repeatedly triggers the reconfigure path.
        if (request?.isForMainFrame == true) {
            consecutiveFailures += 1
            if (consecutiveFailures >= failureThreshold) {
                onPersistentFailure()
            }
        }
        super.onReceivedError(view, request, error)
    }

    override fun onReceivedSslError(
        view: WebView?,
        handler: android.webkit.SslErrorHandler?,
        error: android.net.http.SslError?,
    ) {
        val errHost = try {
            val u = android.net.Uri.parse(error?.url)
            val host = u.host ?: return super.onReceivedSslError(view, handler, error)
            val port = if (u.port > 0) ":${u.port}" else ""
            "$host$port"
        } catch (_: Exception) {
            return super.onReceivedSslError(view, handler, error)
        }
        if (trustedHost != null && errHost.equals(trustedHost, ignoreCase = true)) {
            // Configured Augmentum host with a self-signed cert —
            // user explicitly trusted this server by entering its URL.
            handler?.proceed()
            return
        }
        // Anything else — reject as normal.
        super.onReceivedSslError(view, handler, error)
    }
}
