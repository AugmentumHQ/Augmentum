/**
 * grove-rotation.js — recently-played rotation for Grove's play-matching ladder.
 *
 * Repeat genre asks ("jazz" again, and again) must ROTATE the result rather than
 * replay one video forever. Grove's YouTube tier randomizes its pick, but when
 * the YouTube search times out the deterministic favourite/radio fallback served
 * the identical station every time. This tracks a small recently-played window
 * and biases each tier's pick away from it, while keeping the top-results bias
 * (quality) — so repeats cycle through the good options instead of sticking.
 *
 * Pure + dependency-free so it's unit-testable apart from grove.js's DOM.
 */
export function createRotation(cap = 6) {
  let recent = [];

  /** Remember an id (videoId / file id / station id) as just-played. */
  function note(id) {
    if (!id) return;
    recent = recent.filter((x) => x !== id);
    recent.push(id);
    while (recent.length > cap) recent.shift();
  }

  /** True if this id is within the recently-played window. */
  function isRecent(id) {
    return recent.includes(id);
  }

  /**
   * Pick from the top `topN` items, preferring ones not played recently.
   * Falls back to the full top slice when everything up there is recent (so it
   * never returns null just because the window is saturated). Returns null only
   * for an empty list.
   */
  function preferFresh(items, getId, topN = 4) {
    const top = (items || []).slice(0, Math.max(1, topN));
    if (!top.length) return null;
    const fresh = top.filter((it) => !recent.includes(getId(it)));
    const pool = fresh.length ? fresh : top;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  /** Test/inspection hook. */
  function snapshot() {
    return recent.slice();
  }

  return { note, isRecent, preferFresh, snapshot };
}
