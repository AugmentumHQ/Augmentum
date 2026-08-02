/*
 * ConfigActivity — first-launch and reconfigure URL entry.
 *
 * Single text field + Save button. Persists the base URL to
 * SharedPreferences and relaunches MainActivity. Designed for the
 * D-pad world: large input, large buttons, no fluff.
 *
 * The URL is the Augmentum HTTPS edge (Caddy) — e.g.
 * "https://192.168.1.42:6443" — NOT the cast-receiver path. We
 * append /ui/cast-receiver/ ourselves so the same APK can target
 * any Augmentum host.
 */
package com.augmentum.castreceiver

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class ConfigActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_config)

        val urlInput = findViewById<EditText>(R.id.config_url_input)
        val saveBtn = findViewById<Button>(R.id.config_save_button)
        val statusText = findViewById<TextView>(R.id.config_status)

        // Pre-fill if there's a previous value (e.g. user came back
        // here from a failed load).
        val prefs = getSharedPreferences(MainActivity.PREFS_NAME, MODE_PRIVATE)
        val existing = prefs.getString(MainActivity.KEY_BASE_URL, "") ?: ""
        if (existing.isNotEmpty()) {
            urlInput.setText(existing)
        } else {
            // No prior URL — try LAN auto-discovery. Scan the local
            // subnet for an Augmentum response signature on :6443 and
            // pre-fill the field if we find one. User can still edit
            // before hitting Save. Skipped entirely if the user has
            // already typed something (we check inside the post-back).
            statusText.text = getString(R.string.config_scanning)
            statusText.setTextColor(0xff8aa0ff.toInt())
            statusText.visibility = View.VISIBLE
            Thread(
                {
                    val found = Discovery.findFirstSync(this@ConfigActivity)
                    runOnUiThread {
                        if (found != null && urlInput.text.toString().isBlank()) {
                            urlInput.setText(found)
                            statusText.text = getString(R.string.config_scan_found, found)
                            statusText.setTextColor(0xff8ad57f.toInt())
                        } else if (found == null) {
                            statusText.text = getString(R.string.config_scan_none)
                            statusText.setTextColor(0xff888888.toInt())
                        } else {
                            statusText.visibility = View.GONE
                        }
                    }
                },
                "augmentum-discovery",
            ).start()
        }

        saveBtn.setOnClickListener {
            val raw = urlInput.text.toString().trim()
            if (raw.isEmpty()) {
                statusText.text = getString(R.string.config_url_required)
                statusText.visibility = View.VISIBLE
                return@setOnClickListener
            }
            val normalised = normaliseUrl(raw)
            if (normalised == null) {
                statusText.text = getString(R.string.config_url_invalid)
                statusText.visibility = View.VISIBLE
                return@setOnClickListener
            }
            prefs.edit()
                .putString(MainActivity.KEY_BASE_URL, normalised)
                .apply()
            startActivity(Intent(this, MainActivity::class.java))
            finish()
        }
    }

    /**
     * Accepts the common ways a user might type their Augmentum host —
     * with/without https, with/without port — and returns the
     * canonical form. Returns null when the input is unparseable.
     */
    private fun normaliseUrl(raw: String): String? {
        var working = raw
        if (!working.contains("://")) {
            working = "https://$working"
        }
        return try {
            val parsed = java.net.URI(working)
            // Drop trailing path; MainActivity adds /ui/cast-receiver/.
            val scheme = parsed.scheme ?: return null
            val host = parsed.host ?: return null
            val port = if (parsed.port > 0) ":${parsed.port}" else ""
            "$scheme://$host$port"
        } catch (_: Exception) {
            null
        }
    }
}
