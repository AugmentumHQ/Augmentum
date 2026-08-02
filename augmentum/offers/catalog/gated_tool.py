"""Gated-tool offers — heavy tools the model PROPOSES; the user confirms.

``image_generation`` and ``build_application`` are costly / long-running. In
Auto mode the model emits the ``[[tool:NAME]]`` marker; SSOS turns that into one
of these offers instead of firing the tool. The chip renders inline with
Accept / Not now / Never, so the user always has an exit when the intent was
inferred wrong. On Accept the real tool launches and its OWN step-by-step
progress UI (the build project card, the image gallery/events) shows every
stage — the confirmation is step 0, the native UI is steps 1..n.

See docs/superpowers/specs/2026-06-02-offer-substrate-design.md and the SSOS
``kind == "gated"`` capabilities in modes/passthrough/orchestrator.py.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from augmentum.offers.catalog.base import (
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

log = get_logger(__name__)

KIND = "gated_tool"


# target_id is the registry tool name. Each carries the chip framing + an
# honest "what happens after Accept" line pointing at the tool's native UI.
_GATED: list[dict[str, str]] = [
    {
        "tool": "image_generation",
        "verb": "Generate image",
        "icon": "🎨",
        "hint": "Creates an image from the prompt — uses GPU time.",
        "started_msg": "Generating your image — it'll appear here when it's ready.",
    },
    {
        "tool": "build_application",
        "verb": "Build the app",
        "icon": "🏗️",
        "hint": "Spins up a workspace and builds it — runs for a few minutes.",
        "started_msg": "Building — follow the live steps in the project card.",
    },
    # Structured creators — the offer carries a planned outline (in extra);
    # Accept runs the tool with the full structure, not a single string.
    {
        "tool": "create_ebook",
        "verb": "Write the ebook",
        "icon": "📕",
        "hint": "Writes and illustrates the whole ebook from the outline.",
        "started_msg": "Writing and illustrating your ebook — it'll land in your library.",
    },
    {
        "tool": "create_presentation",
        "verb": "Build the deck",
        "icon": "📊",
        "hint": "Builds the slide deck from the outline.",
        "started_msg": "Building your deck — it'll appear in your library.",
    },
    {
        "tool": "create_document",
        "verb": "Write the document",
        "icon": "📄",
        "hint": "Writes the structured document from the outline.",
        "started_msg": "Writing your document — it'll appear in your library.",
    },
]


def _read_args(payload: dict[str, Any]) -> tuple[str, str]:
    extra = payload.get("extra") or {}
    return str(extra.get("args") or "").strip(), str(extra.get("primary_arg") or "").strip()


def _read_structured(payload: dict[str, Any]) -> dict | None:
    """Planned tools carry the tool's full structured input under
    extra['structured'] — the outline the user confirmed."""
    extra = payload.get("extra") or {}
    s = extra.get("structured")
    return s if isinstance(s, dict) and s else None


def _make_entry(spec: dict[str, str]) -> CatalogEntry:
    async def _preview(_target_id: str, _user_id: str) -> OfferPreview | None:
        # The concrete prompt/description lives in payload.extra (not available
        # at preview time); the chip body's ``reason`` carries the specifics.
        return OfferPreview(
            label=spec["verb"],
            hint=spec["hint"],
            details={"tool": spec["tool"], "confirm": "true"},
        )

    async def _accept(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        registry = getattr(request.app.state, "tool_registry", None)
        tool = registry.get(spec["tool"]) if registry else None
        if tool is None:
            return {"ok": False, "error": "tool_unavailable",
                    "detail": f"{spec['tool']} isn't available right now."}

        user = request.scope.get("user")
        user_id = getattr(user, "id", "") if user is not None else ""
        ctx = {"user_id": user_id, "source": "gated_offer"}
        # Thread the originating chat session (stored at propose time) so the
        # tool associates its output with that chat — and so image/artifact
        # tools satisfy their session_id wiring instead of landing only in the
        # gallery.
        extra = payload.get("extra") or {}
        session_id = str(extra.get("session_id") or "") if isinstance(extra, dict) else ""
        if session_id:
            ctx["session_id"] = session_id

        # build_application is the coder-workspace builder (run_build), shared
        # with build mode + the Library button. Route it through the common
        # start_app_build seam rather than the tool's retired quickjs
        # execute(), and hand the build_id back so the chat attaches its live
        # build card (offer-chip.js).
        if spec["tool"] == "build_application":
            objective, _primary = _read_args(payload)
            if not objective:
                return {"ok": False, "error": "missing_args",
                        "detail": "The proposal carried no build description."}
            model = str(extra.get("model") or "") if isinstance(extra, dict) else ""
            from augmentum.builds.dispatch import start_app_build
            ack = await start_app_build(
                request.app.state,
                objective=objective, user_id=user_id,
                session_id=session_id, model=model,
            )
            if not ack.get("ok"):
                return {"ok": False, "error": ack.get("error", "build_failed"),
                        "detail": ack.get("detail", "Couldn't start the build.")}
            log.info("gated_tool_accepted", tool=spec["tool"], user_id=user_id, mode="build")
            return {
                "ok": True, "started": True, "tool": spec["tool"],
                "build_id": ack["build_id"], "name": ack.get("name", ""),
                "next_step": spec["started_msg"],
            }

        structured = _read_structured(payload)
        if structured is not None:
            # Planned creator — run with the confirmed outline (title + sections).
            kwargs = {**structured, "_context": ctx}
        else:
            args, primary = _read_args(payload)
            if not args or not primary:
                return {"ok": False, "error": "missing_args",
                        "detail": "The proposal carried no prompt/description."}
            kwargs = {primary: args, "_context": ctx}

        # Delivery policy:
        #   * DETACHED — long-running (build_application) or planned creators
        #     (ebook/deck/doc). A multi-minute build must not block the Accept
        #     response, and each already delivers to its OWN native home
        #     (build → project card via ACTIVE_BUILDS; the creators → the
        #     library). We return a "started" ack; the chip names the home.
        #   * INLINE — fast result tools (image_generation, seconds). Await the
        #     result and hand it straight back as a ``deliverable`` so the chat
        #     appends it into the originating session's last assistant message,
        #     persisted in the tree like the pre-offer-substrate inline tool
        #     card. Without this the image lands only in the gallery and the
        #     user has to hunt for it — the regression this restores.
        detached = structured is not None or getattr(tool, "long_running", False)
        if detached:
            async def _run() -> None:
                try:
                    await tool.execute(**kwargs)
                except Exception:
                    log.warning(
                        "gated_tool_run_failed", tool=spec["tool"], exc_info=True,
                    )

            asyncio.create_task(_run())
            log.info(
                "gated_tool_accepted", tool=spec["tool"],
                user_id=user_id, mode="detached",
            )
            return {
                "ok": True,
                "started": True,
                "tool": spec["tool"],
                "next_step": spec["started_msg"],
            }

        # Inline: await the fast tool, surface its result for in-chat delivery.
        # If the request is cancelled (client/proxy timeout on a very slow GPU)
        # the queued job may still finish into the gallery, so the worst case
        # degrades to the old "it's in the gallery" behavior — never data loss.
        try:
            result = await tool.execute(**kwargs)
        except Exception:
            log.warning("gated_tool_run_failed", tool=spec["tool"], exc_info=True)
            return {"ok": False, "error": "tool_failed",
                    "detail": f"{spec['tool']} didn't finish — check the gallery."}
        if getattr(result, "success", False) is not True:
            return {"ok": False, "error": "tool_failed",
                    "detail": (getattr(result, "error", "") or
                               f"{spec['tool']} didn't finish.")}
        meta = getattr(result, "metadata", None) or {}
        url = str(meta.get("url") or "").strip()
        log.info(
            "gated_tool_accepted", tool=spec["tool"],
            user_id=user_id, mode="inline", delivered=bool(url),
        )
        return {
            "ok": True,
            "tool": spec["tool"],
            # The chat consumer (mountOfferFeed onAfterAction) appends this into
            # the originating session; None when no URL came back.
            "deliverable": (
                {"kind": "image", "url": url, "session_id": session_id}
                if url else None
            ),
            "next_step": "Here's your image." if url else spec["started_msg"],
        }

    return CatalogEntry(
        kind=KIND,
        target_id=spec["tool"],
        title=spec["verb"] + "?",
        scope="user",
        build_preview=_preview,
        accept=_accept,
        icon=spec["icon"],
    )


ENTRIES: list[CatalogEntry] = [_make_entry(s) for s in _GATED]

if ENTRIES:
    register_kind(KIND, ENTRIES)
