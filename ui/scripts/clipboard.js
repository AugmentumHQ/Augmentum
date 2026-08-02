// Clipboard helper. The async Clipboard API is only available in secure
// contexts (HTTPS or localhost) and is missing in private-mode Firefox
// and older Safari builds. Augmentum frequently runs over plain HTTP on
// LAN (e.g. http://homelab.local), so every copy site needs to fall
// back to the legacy `execCommand('copy')` path or the click silently
// throws. This helper centralises the dance.

/**
 * Copy ``text`` to the system clipboard. Returns a Promise that
 * resolves true on success and false on failure. Never throws.
 *
 * Prefers the modern async Clipboard API, falls back to a hidden
 * textarea + document.execCommand('copy'). The textarea is appended
 * off-screen, kept readonly to avoid the iOS keyboard popping up,
 * and removed in a finally block so a thrown selectionRange doesn't
 * leak DOM nodes.
 */
export async function copyToClipboard(text) {
  const value = text == null ? '' : String(text);
  if (navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(value);
      return true;
    } catch {
      // Permission denied / not focused / non-secure context — fall through
    }
  }
  return _execCommandFallback(value);
}

function _execCommandFallback(value) {
  if (typeof document === 'undefined' || !document.body) return false;
  const ta = document.createElement('textarea');
  ta.value = value;
  ta.setAttribute('readonly', '');
  // Off-screen but still focusable; iOS needs a real position to copy.
  ta.style.position = 'fixed';
  ta.style.top = '0';
  ta.style.left = '-9999px';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  let ok = false;
  try {
    ta.focus();
    ta.select();
    ta.setSelectionRange(0, value.length);
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  } finally {
    document.body.removeChild(ta);
  }
  return ok;
}
