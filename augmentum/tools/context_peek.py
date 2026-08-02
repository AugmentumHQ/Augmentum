"""context.peek — the pull door of the perception contract.

The companion's prompt names what exists at index/digest fidelity (the
perception block in ``companion_runtime/presence_context.py``); this
tool fetches the FULL detail behind any named slot, silently, into the
model's loop. Every branch reuses an existing substrate — AttentionStore,
notes store, device_play_history, the results ring, the ReferentCache —
peek deepens what the index says exists; it does not go fishing beyond
declared perception (orchestrator-bleed principle).

A successful peek also re-warms the matching ring entry, so the detail
she just pulled renders full again next turn instead of instantly
re-decaying.
"""

from __future__ import annotations

from typing import Any

from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_SLOTS = ("page", "note", "playing", "working", "recent", "referents",
          "abilities", "loaded")


def _clock(seconds: Any) -> str:
    """Seconds → '1h 58m' / '32m 10s' / '45s' — spoken-friendly."""
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "?"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class ContextPeekTool(Tool):
    """Pull full detail behind the perception index, headlessly."""

    def __init__(self, app_state: Any = None) -> None:
        self._app_state = app_state

    @property
    def surfaces(self) -> SurfaceExposure:
        # Conversational-surface tool — hidden from reasoning-flow
        # steps (see SurfaceExposure.flow).
        return SurfaceExposure(flow=False)

    @property
    def name(self) -> str:
        return "context_peek"

    @property
    def description(self) -> str:
        return (
            "Silently fetch the full detail behind something you can "
            "only see the name of — the open page's text, the note's "
            "full content, what's playing and how far in, the open "
            "code file's location, the full text they HANDED you to read "
            "(slot 'loaded' — the 'Read this' button), your recent "
            "results, current referents, or YOUR OWN full ability census "
            "(slot 'abilities' — use it for 'what can you do?'; your "
            "inline roster shows only a slice). Use before quoting, "
            "summarizing, or editing anything your context shows only "
            "as a title or digest."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "slot": {
                    "type": "string",
                    "enum": list(_SLOTS),
                    "description": (
                        "Which named context to deepen: 'page' (open "
                        "browse article's text), 'note' (the co-author "
                        "note's full content), 'playing' (position + "
                        "recent plays), 'working' (open code file "
                        "location), 'recent' (your recent tool results "
                        "in full), 'referents' (last image/url/file + "
                        "trail)."
                    ),
                },
            },
            "required": ["slot"],
        }

    @property
    def timeout(self) -> float:
        return 8.0

    @property
    def cacheable(self) -> bool:
        # Peeking twice is legitimate — the underlying state moves.
        return False

    async def execute(self, **kwargs) -> ToolResult:
        slot = str(kwargs.get("slot") or "").strip().lower()
        if slot not in _SLOTS:
            return ToolResult(
                success=False,
                error=f"Unknown slot '{slot}'. One of: {', '.join(_SLOTS)}",
            )
        user_id = self.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(success=False, error="No user context.")
        ctx = kwargs.get("_context") or {}
        session_id = str(ctx.get("session_id") or "") if isinstance(ctx, dict) else ""

        try:
            handler = getattr(self, f"_peek_{slot}")
            return await handler(user_id, session_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("context_peek_failed", slot=slot, error=str(exc))
            return ToolResult(success=False, error=f"peek {slot} failed: {exc}")

    # -- slot handlers ------------------------------------------------------

    def _refs(self, user_id: str, session_id: str) -> Any:
        if self._app_state is None:
            return None
        try:
            from augmentum.intent.dispatch import get_referent_cache
            return get_referent_cache(self._app_state, user_id, session_id)
        except Exception:  # noqa: BLE001
            return None

    def _rewarm(
        self, user_id: str, session_id: str, *,
        slot: str, label: str, digest: str, detail: str,
    ) -> None:
        """A peeked slot re-earns full fidelity for the coming turns."""
        refs = self._refs(user_id, session_id)
        if refs is None:
            return
        from augmentum.companion_runtime import ring
        ring.record(
            refs, kind="presence", slot=slot,
            label=label, digest=digest, detail=detail,
            refetch={"slot": slot.split(":", 1)[-1]},
        )

    async def _peek_page(self, user_id: str, session_id: str) -> ToolResult:
        from augmentum.companion_runtime.presence_context import ATTENTION
        page = ATTENTION.get(user_id, "page")
        if not page:
            return ToolResult(
                success=True,
                output=(
                    "No page is open on their screen right now (or it "
                    "went stale). If they mean something specific, ask."
                ),
            )
        label = page.get("label") or page.get("url") or "the page"
        excerpt = (page.get("excerpt") or "").strip()
        if not excerpt:
            return ToolResult(
                success=True,
                output=(
                    f'Open page: "{label}" ({page.get("url") or "no url"}) '
                    "— no extracted text is available for it. web_fetch "
                    "the url to read it."
                ),
            )
        self._rewarm(
            user_id, session_id, slot="presence:page",
            label=str(label), digest="open on their screen",
            detail=excerpt[:700],
        )
        from augmentum.security.untrusted import wrap_untrusted
        return ToolResult(
            success=True,
            output=wrap_untrusted(
                "browse/page",
                f'"{label}" ({page.get("url") or ""}):\n\n{excerpt}',
            ),
            metadata={"chars": len(excerpt)},
        )

    async def _peek_note(self, user_id: str, session_id: str) -> ToolResult:
        from augmentum.companion_runtime.presence_context import _active_note
        note = await _active_note(self._app_state, user_id, session_id)
        if note is None:
            return ToolResult(
                success=True,
                output="No note is open between you two right now.",
            )
        content = note["content"]
        capped = content[:4000]
        marker = " …(truncated)" if len(content) > 4000 else ""
        self._rewarm(
            user_id, session_id, slot="presence:note",
            label=note["title"], digest=f"{len(content)} chars",
            detail=content[-600:],
        )
        return ToolResult(
            success=True,
            output=f'Note "{note["title"]}" (full content{marker}):\n\n{capped}',
            metadata={"chars": len(content), "note_id": note["note_id"]},
        )

    async def _peek_playing(self, user_id: str, session_id: str) -> ToolResult:
        from augmentum.companion_runtime.presence_context import ATTENTION
        lines: list[str] = []

        # Receiver casts FIRST — server truth, and the index tier
        # already names them ("…on Living Room TV"), so a peek that
        # answered "Nothing is playing" while a cast ran contradicted
        # her own perception line (the gap this leg closes,
        # 2026-06-12). Live position/pause/volume come from the
        # capability snapshot — the session's cached state only
        # updates when WE invoke an action.
        reg = getattr(self._app_state, "device_registry", None)
        if reg is not None:
            try:
                sessions = await reg.list_sessions(user_id=user_id)
                for s in sessions:
                    if not str(getattr(s, "capability_id", "")).startswith("media."):
                        continue
                    extra = getattr(s, "extra", None) or {}
                    device_label = extra.get("device_label") or s.device_id
                    head = f"Casting to {device_label}: {s.title or '?'}"
                    who = extra.get("author") or extra.get("artist") or ""
                    if who:
                        head += f" by {who}"
                    lines.append(head)
                    snap = await reg.snapshot(
                        user_id=user_id, device_id=s.device_id,
                        capability=s.capability_id,
                    ) or {}
                    bits: list[str] = []
                    pos = snap.get("current_time_s")
                    dur = snap.get("duration_s")
                    if pos is not None and dur:
                        bits.append(f"{_clock(pos)} of {_clock(dur)}")
                    elif pos is not None:
                        bits.append(f"{_clock(pos)} in")
                    if snap.get("is_paused") is True:
                        bits.append("paused")
                    elif snap.get("is_paused") is False:
                        bits.append("playing")
                    if snap.get("is_muted"):
                        bits.append("muted")
                    elif snap.get("volume_level") is not None:
                        bits.append(f"volume {int(snap['volume_level'])}")
                    if bits:
                        lines.append("  " + ", ".join(bits))
            except Exception:  # noqa: BLE001
                log.debug("peek_playing_receiver_failed", exc_info=True)

        playing = ATTENTION.get(user_id, "playing")
        if playing:
            mins = int((playing.get("age_s") or 0) // 60)
            when = "right now" if mins < 5 else f"{mins} minutes ago"
            lines.append(
                f"Playing ({when}): {playing.get('label')} "
                f"({playing.get('kind') or 'media'})"
            )
        try:
            conn = getattr(
                getattr(self._app_state, "state_manager", None), "backend", None,
            )
            conn = getattr(conn, "conn", None)
            if conn is not None:
                from augmentum.architect.inference import query_play_history
                rows = await query_play_history(
                    conn, user_id, limit=3, favourites_first=False,
                )
                for row in rows or []:
                    lines.append(
                        f"- {row.get('content_label') or '?'} "
                        f"({row.get('capability_id') or 'media'}, "
                        f"last at {row.get('created_at') or '?'})"
                    )
        except Exception:  # noqa: BLE001
            log.debug("peek_playing_history_failed", exc_info=True)
        if not lines:
            return ToolResult(
                success=True,
                output="Nothing is playing, and no recent plays found.",
            )
        return ToolResult(success=True, output="\n".join(lines))

    async def _peek_abilities(self, user_id: str, session_id: str) -> ToolResult:
        """Full capability census, generated from the live registry.

        The inline roster is relevance-ranked under a char budget, so
        "what can you do?" — which scores weakly against everything —
        showed her maybe a dozen lines while 40+ verbs sat deferred
        and invisible. Her self-description understated reality
        (2026-06-12 audit). This census is the honest answer: every
        Tier-3 verb she can dispatch, grouped by family, each carrying
        its first sentence — generated, never hand-maintained, so it
        cannot drift from the registry (the doc_facts lesson).
        """
        lines: list[str] = []
        count = 0
        try:
            from augmentum.intent.registry import REGISTRY
            families: dict[str, list[str]] = {}
            for action in REGISTRY.all():
                if not action.fanout.tier3:
                    continue
                if action.surfaces and not (
                    "becca" in action.surfaces or "chat" in action.surfaces
                ):
                    continue
                fam = action.id.split(".", 1)[0]
                first = (action.summary or "").split(".")[0].strip()[:90]
                families.setdefault(fam, []).append(
                    f"  {action.id} — {first}"
                )
                count += 1
            for fam in sorted(families):
                lines.append(f"[{fam}]")
                lines.extend(sorted(families[fam]))
        except Exception:  # noqa: BLE001
            log.warning("peek_abilities_registry_failed", exc_info=True)
        try:
            from augmentum.companion_runtime.native_loop import CORE_TOOL_NAMES
            lines.append(
                "[always-on tools] " + ", ".join(CORE_TOOL_NAMES)
            )
        except Exception:  # noqa: BLE001
            log.debug("peek_abilities_pool_failed", exc_info=True)
        if not lines:
            return ToolResult(
                success=True,
                output="The action registry isn't available right now.",
            )
        header = (
            "Everything you can do right now — answer from THIS, not "
            "from your inline roster (it shows only a relevance-ranked "
            "slice). Group naturally when you speak; don't recite ids."
        )
        return ToolResult(
            success=True,
            output=header + "\n\n" + "\n".join(lines)[:6000],
            metadata={"verbs": count},
        )

    async def _peek_loaded(self, user_id: str, session_id: str) -> ToolResult:
        """Full body the user handed her via the 'Read this …' button."""
        from augmentum.companion_runtime.presence_context import LOADED
        item = LOADED.get_latest(user_id)
        if not item:
            return ToolResult(
                success=True,
                output=(
                    "They haven't handed you anything to read right now "
                    "(or it went stale). If they mean something specific, "
                    "ask or peek another slot."
                ),
            )
        content = (item.get("content") or "").strip()
        kind = item.get("kind") or "item"
        label = item.get("label") or kind
        if not content:
            return ToolResult(
                success=True,
                output=f'They handed you "{label}" but it came through empty.',
            )
        capped = content[:4000]
        marker = " …(truncated — peek again for more)" if len(content) > 4000 else ""
        self._rewarm(
            user_id, session_id, slot="loaded:current",
            label=f"{kind}: {label}", digest=f"full {kind}",
            detail=content[:900],
        )
        return ToolResult(
            success=True,
            output=f'Full {kind} "{label}"{marker}:\n\n{capped}',
            metadata={"chars": len(content)},
        )

    async def _peek_working(self, user_id: str, session_id: str) -> ToolResult:
        from augmentum.companion_runtime.presence_context import ATTENTION
        working = ATTENTION.get(user_id, "working")
        if not working:
            return ToolResult(
                success=True,
                output="No code file is open in the workspace right now.",
            )
        return ToolResult(
            success=True,
            output=(
                f"Open in the coder workspace: {working.get('label')} "
                f"(path: {working.get('path') or '?'}, workspace "
                f"{working.get('ref') or '?'}). You can't read its "
                "contents from here yet — discuss it by name, or "
                "suggest they ask in the coder surface."
            ),
        )

    async def _peek_recent(self, user_id: str, session_id: str) -> ToolResult:
        refs = self._refs(user_id, session_id)
        if refs is None:
            return ToolResult(success=True, output="No recent results.")
        from augmentum.companion_runtime import ring
        entries = ring.alive(refs, keep_turns=ring.DEFAULT_KEEP_TURNS)
        body: list[str] = []
        for e in reversed(entries):
            body.append(f"== {e.get('label')} ==")
            body.append(e.get("detail") or e.get("digest") or "(no detail kept)")
        if not body:
            return ToolResult(
                success=True,
                output="Nothing in your recent-results ring — it decayed.",
            )
        return ToolResult(success=True, output="\n".join(body)[:6000])

    async def _peek_referents(self, user_id: str, session_id: str) -> ToolResult:
        refs = self._refs(user_id, session_id)
        if refs is None:
            return ToolResult(success=True, output="No referents tracked.")
        bits: list[str] = []
        for attr, label in (
            ("last_url", "last url"),
            ("last_image_title", "last image"),
            ("last_image_prompt", "last image prompt"),
            ("last_file_id", "last file id"),
            ("last_played_track", "last played"),
            ("active_note_title", "active note"),
            ("last_dispatch_summary", "last action"),
        ):
            val = getattr(refs, attr, None)
            if val:
                bits.append(f"- {label}: {str(val)[:160]}")
        trail = list(getattr(refs, "trail", None) or [])[-5:]
        if trail:
            bits.append("- trail (most recent last):")
            bits += [
                f"    {t.get('kind')}: {t.get('label')}" for t in trail
            ]
        if not bits:
            return ToolResult(
                success=True, output="No referents tracked yet this session.",
            )
        return ToolResult(success=True, output="\n".join(bits))
