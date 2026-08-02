/*
 * LAN auto-discovery for the Augmentum server.
 *
 * Computes the device's local IPv4 subnet (refusing anything wider than
 * /16 so we never carpet-bomb a corporate LAN) and probes every host on
 * port 6443 in parallel for the `/api/auth/status` response signature.
 * First responder wins; all in-flight probes are cancelled.
 *
 * The trust-all SSL context exists ONLY here, ONLY for the probe — every
 * home install ships with a self-signed Caddy cert and we're explicitly
 * looking for it. The WebView retains the platform's default trust policy.
 */
package com.augmentum.castreceiver

import java.net.HttpURLConnection
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.URL
import java.security.SecureRandom
import java.security.cert.X509Certificate
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager

object Discovery {

    private const val PORT = 6443
    private const val PROBE_PATH = "/api/auth/status"
    // Distinct enough to be Augmentum: only /api/auth/status returns
    // a body containing both keys together. False-positive surface ≈ 0.
    private const val SIG_A = "\"setup_required\""
    private const val SIG_B = "\"authenticated\""
    private const val PROBE_TIMEOUT_MS = 800
    private const val MDNS_TIMEOUT_MS = 2500L
    private const val OVERALL_TIMEOUT_MS = 6000L
    private const val MAX_PARALLEL = 64

    internal val trustAllSsl: SSLContext by lazy {
        SSLContext.getInstance("TLS").apply {
            init(null, arrayOf<TrustManager>(object : X509TrustManager {
                override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) {}
                override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {}
                override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
            }), SecureRandom())
        }
    }

    internal val acceptAllHosts = HostnameVerifier { _, _ -> true }

    /**
     * Blocking scan. Caller must invoke from a background thread.
     * Returns the canonical base URL of the first responder, or null
     * if nothing on the LAN answers within OVERALL_TIMEOUT_MS.
     */
    fun findFirstSync(context: android.content.Context): String? {
        // mDNS: gives the LAN ~MDNS_TIMEOUT_MS to answer. The Augmentum
        // server advertises an `_http._tcp.` service whose name contains
        // "augmentum" — filter on that so we don't probe every printer
        // and Hue bridge in range. The found URL is returned with the
        // resolved port (NOT a hardcoded one) so a non-standard install
        // still works as long as the SRV record is honest.
        val mdnsUrl = findViaMdns(context)
        if (mdnsUrl != null) return mdnsUrl

        // Subnet sweep fallback for when the LAN doesn't carry mDNS
        // (some enterprise networks, IPv6-only segments, or a server
        // build that pre-dates the mDNS advertisement).
        val hosts = localSubnetHosts() ?: return null
        if (hosts.isEmpty()) return null

        val pool = Executors.newFixedThreadPool(MAX_PARALLEL.coerceAtMost(hosts.size))
        val found = AtomicReference<String?>(null)
        val scanDone = CountDownLatch(1)
        try {
            for (ip in hosts) {
                pool.submit {
                    if (found.get() != null) return@submit
                    if (probe(ip, PORT) && found.compareAndSet(null, ip)) {
                        scanDone.countDown()
                    }
                }
            }
            scanDone.await(OVERALL_TIMEOUT_MS, TimeUnit.MILLISECONDS)
        } finally {
            pool.shutdownNow()
        }
        return found.get()?.let { "https://$it:$PORT" }
    }

    /**
     * mDNS leg. Returns the resolved ``https://host:port`` of the first
     * Augmentum service we both resolve AND probe successfully, or
     * null if nothing answers within MDNS_TIMEOUT_MS.
     *
     * NSD callbacks (``onServiceFound`` / ``onServiceResolved``) run
     * on a system handler thread — we MUST NOT block them with the
     * 800ms HTTPS probe. Probes are dispatched to a small worker pool
     * and the success path signals completion via the shared latch.
     */
    private fun findViaMdns(context: android.content.Context): String? {
        val nsdManager = context.getSystemService(android.content.Context.NSD_SERVICE)
            as? android.net.nsd.NsdManager ?: return null

        val found = AtomicReference<String?>(null)
        val done = CountDownLatch(1)
        val probePool = Executors.newFixedThreadPool(4)

        val resolveListener = object : android.net.nsd.NsdManager.ResolveListener {
            override fun onResolveFailed(serviceInfo: android.net.nsd.NsdServiceInfo, errorCode: Int) {}
            override fun onServiceResolved(serviceInfo: android.net.nsd.NsdServiceInfo) {
                val host = serviceInfo.host?.hostAddress ?: return
                val port = serviceInfo.port.takeIf { it > 0 } ?: PORT
                probePool.submit {
                    if (found.get() != null) return@submit
                    if (probe(host, port)) {
                        if (found.compareAndSet(null, "https://$host:$port")) {
                            done.countDown()
                        }
                    }
                }
            }
        }

        val discoveryListener = object : android.net.nsd.NsdManager.DiscoveryListener {
            override fun onDiscoveryStarted(regType: String) {}
            override fun onServiceFound(service: android.net.nsd.NsdServiceInfo) {
                if (service.serviceName.contains("augmentum", ignoreCase = true)) {
                    try {
                        nsdManager.resolveService(service, resolveListener)
                    } catch (_: Exception) {}
                }
            }
            override fun onServiceLost(service: android.net.nsd.NsdServiceInfo) {}
            override fun onDiscoveryStopped(serviceType: String) {}
            override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) {}
            override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) {}
        }

        try {
            nsdManager.discoverServices("_http._tcp.", android.net.nsd.NsdManager.PROTOCOL_DNS_SD, discoveryListener)
            done.await(MDNS_TIMEOUT_MS, TimeUnit.MILLISECONDS)
        } catch (_: Exception) {
        } finally {
            try { nsdManager.stopServiceDiscovery(discoveryListener) } catch (_: Exception) {}
            probePool.shutdownNow()
        }
        return found.get()
    }

    private fun probe(host: String, port: Int = PORT): Boolean {
        var conn: HttpURLConnection? = null
        return try {
            val url = URL("https", host, port, PROBE_PATH)
            conn = (url.openConnection() as HttpsURLConnection).apply {
                sslSocketFactory = trustAllSsl.socketFactory
                hostnameVerifier = acceptAllHosts
                connectTimeout = PROBE_TIMEOUT_MS
                readTimeout = PROBE_TIMEOUT_MS
                requestMethod = "GET"
                useCaches = false
                instanceFollowRedirects = false
            }
            if (conn.responseCode != 200) return false
            val body = conn.inputStream.bufferedReader().use { it.readText() }
            body.contains(SIG_A) && body.contains(SIG_B)
        } catch (_: Exception) {
            false
        } finally {
            try { conn?.disconnect() } catch (_: Exception) {}
        }
    }

    private fun localSubnetHosts(): List<String>? {
        val ifaces = try {
            NetworkInterface.getNetworkInterfaces() ?: return null
        } catch (_: Exception) {
            return null
        }
        for (iface in ifaces) {
            if (!iface.isUp || iface.isLoopback || iface.isVirtual) continue
            for (addr in iface.interfaceAddresses) {
                val ip = addr.address as? Inet4Address ?: continue
                if (ip.isLinkLocalAddress || ip.isAnyLocalAddress) continue
                val prefix = addr.networkPrefixLength.toInt()
                if (prefix < 16) continue  // refuse to scan /15 or wider
                return enumerateHosts(ip, prefix)
            }
        }
        return null
    }

    private fun enumerateHosts(local: Inet4Address, prefix: Int): List<String> {
        val ipInt = local.address.let {
            ((it[0].toInt() and 0xff) shl 24) or
            ((it[1].toInt() and 0xff) shl 16) or
            ((it[2].toInt() and 0xff) shl 8) or
            (it[3].toInt() and 0xff)
        }
        val mask = if (prefix == 0) 0 else (0xffffffff.toInt() shl (32 - prefix))
        val network = ipInt and mask
        val broadcast = network or mask.inv()

        val out = mutableListOf<String>()
        var addr = network + 1
        while (addr < broadcast) {
            if (addr != ipInt) out.add(intToIp(addr))
            if (out.size >= 1022) break  // safety cap (covers up to /22)
            addr += 1
        }
        return out
    }

    private fun intToIp(ip: Int): String =
        "${(ip ushr 24) and 0xff}.${(ip ushr 16) and 0xff}.${(ip ushr 8) and 0xff}.${ip and 0xff}"
}
