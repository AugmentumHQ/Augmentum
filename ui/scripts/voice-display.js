// Shared formatting helpers for voice entries from /api/audio/voices.
//
// Voice rows now carry a ``sources`` array (added by audio_routes.list_voices
// when fabric peers advertise the same voice). The dedupe-by-name design
// keeps the dropdown one-entry-per-voice regardless of fleet size, and the
// badge here makes the redundancy visible without overwhelming the user.
//
// All consumers should route voice labels through these helpers so a future
// shape change (e.g. inline language badges, custom-voice indicators) lands
// in one place instead of N surfaces.

/**
 * Count how many fabric peers advertise this voice in addition to local.
 * Returns 0 when the voice is local-only or peer-only with a single source.
 */
export function peerSourceCount(voice) {
  const sources = voice?.sources;
  if (!Array.isArray(sources) || sources.length <= 1) return 0;
  return sources.filter(s => !s.is_local).length;
}

/**
 * Returns true when local advertises this voice (in addition to any peers).
 */
export function hasLocalSource(voice) {
  const sources = voice?.sources;
  if (!Array.isArray(sources) || sources.length === 0) {
    // Fallback for entries built before the sources field landed: infer from
    // the provider_id — local Kokoro/Pocket use bare ids, fabric peers use
    // "fabric:<node>:<provider>" handles.
    const pid = voice?.provider_id || '';
    return !pid.startsWith('fabric:');
  }
  return sources.some(s => s.is_local);
}

/**
 * Compact badge string for a voice. Returns:
 *   ""           — local-only (or single-source, nothing to indicate)
 *   "• 2"        — 2 peers also have it
 *   "[Box 2]"    — peer-only voice (no local source)
 *
 * Designed for plain <option> text where rich HTML isn't available. Use
 * ``voiceBadgeRich`` when you want a styled span.
 */
export function voiceBadge(voice) {
  const peers = peerSourceCount(voice);
  const local = hasLocalSource(voice);
  if (!peers && local) return '';
  if (!local && peers === 0) return '';
  if (!local) {
    // Peer-only: name the single peer (uses hostname when available).
    const peerSource = (voice.sources || []).find(s => !s.is_local);
    const host = peerSource?.hostname || peerSource?.node_id?.slice(0, 8) || 'peer';
    return `[${host}]`;
  }
  return `• ${peers}`;
}

/**
 * Voice label suitable for <option> text. Combines the voice name with
 * the badge so a single dropdown render captures availability info.
 *
 * Example outputs:
 *   "af_heart"                — local only
 *   "af_heart • 2"            — local + 2 peers
 *   "sky_clone (Box 3)"       — peer-only
 */
export function formatVoiceLabel(voice) {
  const name = voice?.name || voice?.voice_id || '';
  const badge = voiceBadge(voice);
  return badge ? `${name} ${badge}` : name;
}

/**
 * Multi-line tooltip describing every source of a voice. Use as the
 * ``title`` attribute on <option> or as the body of a rich popover.
 */
export function voiceSourcesTooltip(voice) {
  const sources = voice?.sources || [];
  if (sources.length === 0) {
    const pid = voice?.provider_id || '';
    return pid ? `Routes via: ${pid}` : '';
  }
  const lines = sources.map((s, idx) => {
    if (s.is_local) return `  • local${idx === 0 ? '  (preferred)' : ''}`;
    const host = s.hostname || s.node_id?.slice(0, 8) || 'peer';
    const icon = s.icon ? `${s.icon} ` : '';
    return `  • ${icon}${host}${idx === 0 ? '  (preferred)' : ''}`;
  });
  return `Available on ${sources.length} source${sources.length > 1 ? 's' : ''}:\n${lines.join('\n')}`;
}

/**
 * Build an HTML span carrying the badge with styling hooks. Surfaces that
 * render their own option rows (custom dropdowns, not native <select>) can
 * append this directly. Returns "" when no badge is warranted.
 */
export function voiceBadgeRich(voice) {
  const peers = peerSourceCount(voice);
  const local = hasLocalSource(voice);
  if (!local && peers === 0) return '';
  if (!local) {
    const peerSource = (voice.sources || []).find(s => !s.is_local);
    const host = peerSource?.hostname || peerSource?.node_id?.slice(0, 8) || 'peer';
    const icon = peerSource?.icon ? `${peerSource.icon} ` : '';
    return `<span class="voice-source-badge voice-source-peer" title="${voiceSourcesTooltip(voice).replace(/"/g, '&quot;')}">${icon}${host}</span>`;
  }
  if (!peers) return '';
  return `<span class="voice-source-badge voice-source-shared" title="${voiceSourcesTooltip(voice).replace(/"/g, '&quot;')}">• ${peers}</span>`;
}
