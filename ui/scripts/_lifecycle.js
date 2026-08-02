/**
 * _lifecycle.js — small, framework-free lifetime tracker for mounted UI.
 *
 * Replaces the recurring pattern where each mounted module hand-rolled
 * a set of module-scope handler refs + a manual unmount() that had to
 * mirror every addEventListener / setInterval / observer it set up.
 * That pattern is correct when fully followed, but in practice loses
 * coverage piece by piece — the audit on 2026-06-09 found 7 distinct
 * listener leaks across becca-presence.js and companion-notes.js.
 *
 * Usage:
 *
 *   import { createLifetime } from './_lifecycle.js';
 *
 *   let _lifetime = null;
 *
 *   function mount() {
 *     _lifetime = createLifetime();
 *     _lifetime.addEventListener(window, 'resize', onResize);
 *     _lifetime.add(() => myObserver.disconnect());
 *     _lifetime.addInterval(() => poll(), 5000);
 *   }
 *
 *   function unmount() {
 *     _lifetime?.dispose();
 *     _lifetime = null;
 *   }
 *
 * Disposal runs teardowns in reverse-registration order (LIFO), which
 * mirrors stack unwind semantics: things registered later in mount are
 * usually downstream of earlier ones and want to tear down first.
 * Individual teardown failures are caught and logged, never thrown, so
 * one bad listener can't block the rest of unmount.
 */

const PREFIX = '[lifetime]';

/**
 * Create a lifetime scope. Each mounted module should own exactly one;
 * call ``dispose()`` once on unmount.
 *
 * @returns {{
 *   addEventListener: (target: EventTarget, type: string, fn: Function, opts?: object) => void,
 *   add: (teardown: () => void) => void,
 *   addInterval: (fn: () => void, ms: number) => number,
 *   addTimeout: (fn: () => void, ms: number) => number,
 *   addObserver: (observer: { disconnect: () => void }) => void,
 *   disposed: boolean,
 *   dispose: () => void,
 * }}
 */
export function createLifetime() {
  /** @type {Array<() => void>} */
  const teardowns = [];
  let disposed = false;

  const guard = () => {
    if (disposed) {
      // Registering against a disposed lifetime is almost always a bug
      // (a stale callback firing post-unmount). Surface it loud enough
      // to be greppable but don't throw — the calling path is usually
      // a setTimeout / fetch.then that we can't unwind.
      console.warn(`${PREFIX} register-after-dispose ignored`);
    }
    return disposed;
  };

  return {
    /**
     * Register a DOM event listener. Removed automatically on dispose.
     * Matches the native addEventListener signature so calls read
     * identically — only the target argument moves to the front.
     */
    addEventListener(target, type, fn, opts) {
      if (guard()) return;
      target.addEventListener(type, fn, opts);
      teardowns.push(() => {
        try { target.removeEventListener(type, fn, opts); } catch (_) { /* target gone */ }
      });
    },

    /** Register an arbitrary teardown function. */
    add(teardown) {
      if (guard()) return;
      teardowns.push(teardown);
    },

    /**
     * setInterval wrapper. Returns the interval id so the caller can
     * clear it earlier than dispose if they want.
     */
    addInterval(fn, ms) {
      if (guard()) return 0;
      const id = setInterval(fn, ms);
      teardowns.push(() => clearInterval(id));
      return id;
    },

    /**
     * setTimeout wrapper. Mostly useful for one-shot deferred work
     * that should be cancelled if the module unmounts before it fires.
     */
    addTimeout(fn, ms) {
      if (guard()) return 0;
      const id = setTimeout(fn, ms);
      teardowns.push(() => clearTimeout(id));
      return id;
    },

    /**
     * Register a MutationObserver / IntersectionObserver / ResizeObserver
     * (anything with a ``disconnect`` method).
     */
    addObserver(observer) {
      if (guard()) return;
      teardowns.push(() => {
        try { observer.disconnect(); } catch (_) { /* observer already gone */ }
      });
    },

    /** True once dispose has run; subsequent registrations are no-ops. */
    get disposed() { return disposed; },

    /**
     * Run every registered teardown in LIFO order, swallowing per-step
     * errors. Idempotent — calling twice is a no-op on the second pass.
     */
    dispose() {
      if (disposed) return;
      disposed = true;
      while (teardowns.length) {
        const t = teardowns.pop();
        try {
          t();
        } catch (err) {
          console.warn(`${PREFIX} teardown failed`, err);
        }
      }
    },
  };
}
