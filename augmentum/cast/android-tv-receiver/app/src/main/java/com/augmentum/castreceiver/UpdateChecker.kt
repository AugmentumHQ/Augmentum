package com.augmentum.castreceiver

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.util.Log
import androidx.core.content.FileProvider
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import javax.net.ssl.HttpsURLConnection
import kotlin.concurrent.thread

object UpdateChecker {
    private const val TAG = "UpdateChecker"

    private const val PREFS_NAME = "augmentum_update"
    private const val KEY_LAST_CHECK_MS = "last_check_ms"
    private const val CHECK_INTERVAL_MS = 60L * 60L * 1000L          // 1 hour
    private const val MAX_APK_BYTES = 200L * 1024L * 1024L           // 200 MB hard cap

    fun checkAndInstall(context: Context, baseUrl: String) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val now = System.currentTimeMillis()
        val last = prefs.getLong(KEY_LAST_CHECK_MS, 0L)
        if (now - last < CHECK_INTERVAL_MS) {
            Log.d(TAG, "Skipping update check; last ran ${(now - last) / 1000}s ago")
            return
        }
        prefs.edit().putLong(KEY_LAST_CHECK_MS, now).apply()

        val currentVersion = BuildConfig.VERSION_CODE
        thread {
            try {
                val url = URL("$baseUrl/api/cast/pair/android-tv/version")
                val conn = (url.openConnection() as HttpsURLConnection).apply {
                    sslSocketFactory = Discovery.trustAllSsl.socketFactory
                    hostnameVerifier = Discovery.acceptAllHosts
                    connectTimeout = 5000
                    readTimeout = 5000
                }

                if (conn.responseCode == 200) {
                    val response = conn.inputStream.bufferedReader().use { it.readText() }
                    val json = JSONObject(response)
                    val serverVersion = json.optInt("versionCode", -1)
                    val hasUpdate = json.optBoolean("hasUpdate", false)
                    val autoUpdate = json.optBoolean("autoUpdate", true)

                    if (autoUpdate && hasUpdate && serverVersion > currentVersion) {
                        Log.i(TAG, "Update found: version $serverVersion. Downloading...")
                        downloadAndInstall(context, baseUrl)
                    } else {
                        Log.i(TAG, "No update needed (server=$serverVersion, current=$currentVersion, auto=$autoUpdate)")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error checking for updates", e)
            }
        }
    }

    private fun downloadAndInstall(context: Context, baseUrl: String) {
        var tmpFile: File? = null
        try {
            val url = URL("$baseUrl/api/cast/pair/android-tv/download")
            val conn = (url.openConnection() as HttpsURLConnection).apply {
                sslSocketFactory = Discovery.trustAllSsl.socketFactory
                hostnameVerifier = Discovery.acceptAllHosts
                connectTimeout = 10000
                readTimeout = 60000
            }

            if (conn.responseCode != 200) {
                Log.w(TAG, "Download HTTP ${conn.responseCode}")
                return
            }

            val advertisedLen = conn.contentLengthLong
            if (advertisedLen > MAX_APK_BYTES) {
                Log.e(TAG, "Refusing download: Content-Length $advertisedLen exceeds cap $MAX_APK_BYTES")
                return
            }

            val updatesDir = File(context.getExternalFilesDir(null), "updates")
            if (!updatesDir.exists()) updatesDir.mkdirs()

            val apkFile = File(updatesDir, "augmentum-tv-update.apk")
            // Stream to a sibling temp file and atomically swap in only on success.
            // A torn download otherwise leaves a half-written APK that the OS
            // tries to install on next launch.
            tmpFile = File(updatesDir, "augmentum-tv-update.apk.tmp")
            if (tmpFile.exists()) tmpFile.delete()

            var written = 0L
            conn.inputStream.use { input ->
                FileOutputStream(tmpFile).use { output ->
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n <= 0) break
                        written += n
                        if (written > MAX_APK_BYTES) {
                            throw java.io.IOException("APK exceeded cap mid-stream at $written bytes")
                        }
                        output.write(buf, 0, n)
                    }
                }
            }

            if (apkFile.exists()) apkFile.delete()
            if (!tmpFile.renameTo(apkFile)) {
                Log.e(TAG, "Failed to rename ${tmpFile.absolutePath} → ${apkFile.absolutePath}")
                return
            }
            tmpFile = null  // ownership transferred; suppress cleanup

            Log.i(TAG, "Download complete ($written bytes). Starting installation...")
            installApk(context, apkFile)
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading update", e)
        } finally {
            try { tmpFile?.delete() } catch (_: Exception) {}
        }
    }

    private fun installApk(context: Context, apkFile: File) {
        try {
            // On Android 8+ the user must grant "Install unknown apps" for
            // this package specifically — the manifest permission alone is
            // not enough. If they haven't, leave the APK on disk and bail
            // (the next manual launch through Settings can pick it up).
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
                !context.packageManager.canRequestPackageInstalls()
            ) {
                Log.w(TAG, "REQUEST_INSTALL_PACKAGES not granted by user; skipping silent install. APK staged at ${apkFile.absolutePath}")
                return
            }

            val uri = FileProvider.getUriForFile(
                context,
                "${context.packageName}.fileprovider",
                apkFile
            )

            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "application/vnd.android.package-archive")
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION
            }

            context.startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Error installing update", e)
        }
    }
}
