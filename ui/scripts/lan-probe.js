/**
 * lan-probe.js — Subnet inference for device discovery.
 *
 * The augmentum container often can't see the user's LAN — Docker
 * Desktop sticks it inside a VM whose multicast doesn't escape — but
 * outbound TCP from the container to a LAN IP DOES work (the host
 * routes via NAT). What the container can't easily figure out is
 * WHICH subnet the user is on, especially when the request arrives
 * via a reverse proxy (Caddy), where the source IP looks like the
 * proxy's container IP rather than the user's LAN address.
 *
 * The browser solves that. The browser is on the user's actual LAN.
 * We extract the subnet here and hand it to the server, which then
 * runs the TCP-based sweep from inside the container.
 *
 * Why we don't probe from the browser directly:
 *
 *   The augmentum app's CSP `connect-src` and `img-src` directives
 *   only allow requests to a small allowlist of origins. Cross-LAN
 *   fetches (`http://192.168.X.Y:port/...`) get refused by the
 *   browser before they leave the page. Loosening the CSP to enable
 *   probing would weaken every other route's security boundary just
 *   for one feature, which isn't worth it. WebRTC is a separate
 *   security plane that CSP doesn't gate, so the IP-discovery half
 *   still works fine — we just push the actual probing to the server.
 */


/* ------------------------------------------------------------------ *\
   Subnet inference
\* ------------------------------------------------------------------ */


function _isPrivateIPv4(ip) {
  if (!/^\d+\.\d+\.\d+\.\d+$/.test(ip)) return false;
  const [a, b] = ip.split('.').map(Number);
  if (a === 10) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  return false;
}


function _subnetFor(ip) {
  // Compute the /24 subnet for an IP in CIDR notation.
  const m = ip.match(/^(\d+\.\d+\.\d+)\.\d+$/);
  return m ? `${m[1]}.0/24` : '';
}


/**
 * Extract the browser's LAN IP via a WebRTC ICE candidate.
 *
 * Modern browsers anonymize candidates with `*.local` mDNS names by
 * default, but enough configurations still leak a real private IPv4
 * that this is the most reliable cross-browser path. Returns null on
 * timeout or when no usable candidate appears.
 */
async function _localIPViaWebRTC(timeoutMs = 1500) {
  if (typeof RTCPeerConnection === 'undefined') return null;
  return new Promise((resolve) => {
    const pc = new RTCPeerConnection({ iceServers: [] });
    let resolved = false;
    const finish = (val) => {
      if (resolved) return;
      resolved = true;
      try { pc.close(); } catch { /* ignore */ }
      resolve(val);
    };

    const timer = setTimeout(() => finish(null), timeoutMs);

    pc.onicecandidate = (e) => {
      if (!e.candidate) return;
      const cand = String(e.candidate.candidate || '');
      const match = cand.match(/(\d+\.\d+\.\d+\.\d+)/);
      if (!match) return;
      const ip = match[1];
      if (!_isPrivateIPv4(ip)) return;
      clearTimeout(timer);
      finish(ip);
    };

    pc.createDataChannel('lan-probe');
    pc.createOffer()
      .then(offer => pc.setLocalDescription(offer))
      .catch(() => finish(null));
  });
}


/**
 * Best-effort subnet detection. Returns a CIDR string like
 * `192.168.1.0/24` or empty string when no signal works.
 */
export async function detectLocalSubnet() {
  // 1. Hostname of the page — works when augmentum is on a LAN IP.
  const hostname = window.location.hostname;
  if (_isPrivateIPv4(hostname)) {
    return _subnetFor(hostname);
  }

  // 2. WebRTC ICE candidate — the only universal fallback.
  const webrtcIp = await _localIPViaWebRTC();
  if (webrtcIp) {
    return _subnetFor(webrtcIp);
  }

  return '';
}


/* ------------------------------------------------------------------ *\
   Public API
\* ------------------------------------------------------------------ */


/**
 * Run server-side TCP sweep for the user's local subnet.
 *
 * The browser provides the subnet hint (the only thing the server can't
 * easily figure out behind a reverse proxy); the server does the actual
 * probing via direct TCP, which crosses Docker NAT cleanly and isn't
 * subject to browser CSP rules.
 */
export async function discoverViaServerSweep({ signal } = {}) {
  const subnet = await detectLocalSubnet();
  const start = performance.now();
  const params = subnet ? `?subnet=${encodeURIComponent(subnet)}` : '';

  try {
    const resp = await fetch(`/api/devices/sweep${params}`, { signal });
    if (!resp.ok) {
      return {
        discovered: [],
        subnet,
        durationMs: Math.round(performance.now() - start),
        error: `HTTP ${resp.status}`,
      };
    }
    const body = await resp.json();
    return {
      discovered: body.discovered || [],
      subnet: body.subnet || subnet,
      inferredFrom: body.inferred_from || (subnet ? 'browser' : ''),
      errors: body.errors || {},
      durationMs: Math.round(performance.now() - start),
    };
  } catch (err) {
    return {
      discovered: [],
      subnet,
      durationMs: Math.round(performance.now() - start),
      error: String(err?.message || err),
    };
  }
}


/**
 * Backward-compat shim — the previous `discoverViaBrowserProbe` did
 * direct LAN fetches that CSP now blocks. Keep the same name pointing
 * at the server-sweep path so existing callers keep working.
 */
export const discoverViaBrowserProbe = discoverViaServerSweep;
