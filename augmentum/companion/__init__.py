"""The single object that *is* a companion.

Per the accumulation thesis (Step 2):
``docs/superpowers/specs/2026-05-23-accumulation-thesis.md``.

Until this module landed, "the companion" was a pattern across the
codebase: identity in ``companion_runtime.identity``, state in
``companion_runtime.state``, memory in ``companion_runtime.memory``,
the avatar VRM in ``ui/scripts/becca-presence.js``, the personality
doc in ``docs/superpowers/specs/2026-05-14-becca-personality.md``,
the journal in ``companion_journal``, the affect tracker in
``perception.user_affect``. There was no single thing you could
point at and say "this is her."

This module provides that single thing. :class:`Companion` composes
identity + state + memory + bus + user_affect + history into one
object. Every module that touches her should route through this
class rather than reaching into the constituent systems directly.

Phase 1 (this commit): thin façade. Every method delegates to
existing code. No behavior change. The point is to have *one place
to look* — so future PRs can migrate read/write sites one at a time
into routing through ``app.state.companions[name]`` without changing
behavior at each step.

Phase 2 (later): the class becomes load-bearing. It enforces
invariants (propagation, drift, signature contracts). It owns the
write paths. It's the only thing that knows how to construct a
prompt context.

Phase 3 (eventually): the filesystem layout under
``companions/<name>/`` becomes the source of truth. The class is
the runtime view over that filesystem.
"""

from __future__ import annotations

from augmentum.companion.companion import Companion, CompanionUserView

__all__ = ["Companion", "CompanionUserView"]
