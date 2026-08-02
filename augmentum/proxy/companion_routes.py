"""HTTP and WebSocket routes for the CompanionRuntime.

Sprint 1 adds the presence-bus WebSocket route. Subsequent sprints
add HTTP routes for inspectability (memory.introspect, snapshot, etc.)
and admin operations (kernel refresh, drift audit query).

All routes gate on ``settings.companion_runtime_enabled`` — when off,
the runtime isn't instantiated and routes return 503.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from augmentum.companion_runtime.user_flags import (
    resolve_bool,
    resolve_str,
    write_user_flag,
)
from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["companion"])


@router.websocket("/ws/companion/presence")
async def companion_presence_ws(
    websocket: WebSocket,
    topic: str = Query("**", description="Bus topic glob to subscribe to"),
    slice_key: str = Query("", description="Subscriber tag for telemetry"),
) -> None:
    """Subscribe to the companion runtime's presence bus.

    Query params:
    - ``topic``: glob pattern (default ``**`` = everything). Examples:
      ``state.*``, ``behavior.*``, ``state.transition``.
    - ``slice_key``: optional tag for the runtime's telemetry (e.g.
      ``xr_left``, ``browser_chat``).

    Emits JSON messages of shape:
    ``{"topic": "state.transition", "payload": {...}, "t": 1234567890.0,
       "source_companion_id": "becca", "target_companion_id": null}``

    When ``companion_runtime_enabled`` is off, accepts then closes with
    code 1011 (server error) to signal "not available."
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        await websocket.accept()
        await websocket.send_text(
            '{"error": "companion_runtime_enabled is off"}',
        )
        await websocket.close(code=1011)
        return

    runtime = getattr(websocket.app.state, "companion_runtime", None)
    if runtime is None or not getattr(runtime, "bus", None):
        await websocket.accept()
        await websocket.send_text(
            '{"error": "companion_runtime not initialized on app state"}',
        )
        await websocket.close(code=1011)
        return

    # Scope the stream to the authenticated connection's user so one
    # logged-in client can't read every other user's events (audit
    # 2026-06-17). The auth middleware populates scope["user"]; empty
    # owner (no-auth / unowned single-user) disables the filter.
    _ws_user = websocket.scope.get("user")
    _owner_user_id = getattr(_ws_user, "id", "") if _ws_user else ""
    from augmentum.companion_runtime.bus import ws_fanout
    await ws_fanout(
        websocket, runtime.bus, topic_glob=topic, slice_key=slice_key,
        owner_user_id=_owner_user_id,
    )


class _IntensityRequest(BaseModel):
    level: str = Field(..., max_length=32)


@router.post("/api/companion/intensity")
async def companion_intensity_apply(
    body: _IntensityRequest, request: Request,
) -> JSONResponse:
    """Apply a companion intensity preset.

    Bundles the ~16 individual feature flags into a coherent profile
    (minimal / balanced / full / off). Per-user (multi-tenant 2026-06):
    each tenant sets their OWN intensity — the dial lands on their
    ``user_settings`` row and shapes only their menu + per-request chat
    surfaces. ``write_user_flag`` mirrors to the install-wide store ONLY
    when the caller owns the single runtime (or there's no auth), so the
    one background autonomy loop still honors the owner's dial and a
    non-owner can't impose background cost on the install. (Previously
    this was admin-only + install-wide, which is exactly the cross-tenant
    leak this change fixes.)

    Validates the level; rejects unknown values with 400. Always
    returns the resulting intensity + applied flags so the UI can
    update state without a follow-up status fetch.
    """

    from augmentum.companion.intensity import VALID_INTENSITIES, get_preset
    level = (body.level or "").strip().lower()
    if level not in VALID_INTENSITIES:
        return JSONResponse(
            {"ok": False, "reason": "unknown_intensity",
             "valid": sorted(VALID_INTENSITIES)},
            status_code=400,
        )

    settings_store = getattr(request.app.state, "settings_store", None)
    if settings_store is None:
        return JSONResponse(
            {"ok": False, "reason": "settings_store_unavailable"},
            status_code=503,
        )

    preset = get_preset(level)
    assert preset is not None  # checked above

    # Per-user write (multi-tenant fix): each flag lands on THIS user's
    # row so one tenant's intensity dial can't overwrite another's.
    # write_user_flag also mirrors to the install-wide store when the
    # actor owns the single runtime (or there's no auth / known owner),
    # so the owner's one background loop keeps honoring their dial —
    # byte-identical to the old global write for single-tenant installs.
    uid = _resolve_user_id(request)
    runtime = getattr(request.app.state, "companion_runtime", None)
    owner_user_id = getattr(runtime, "owner_user_id", "") if runtime else ""

    applied: dict[str, bool] = {}
    try:
        for flag, value in preset.flags.items():
            await write_user_flag(
                settings_store, user_id=uid, owner_user_id=owner_user_id,
                key=flag, value=value,
            )
            applied[flag] = value
        await write_user_flag(
            settings_store, user_id=uid, owner_user_id=owner_user_id,
            key="companion_intensity", value=preset.name,
        )
        # Note: in-process settings reload happens via the existing
        # config-loader path on subsequent requests. We intentionally
        # do NOT setattr() the global settings object here — that
        # leaks state across test boundaries and is the wrong layer
        # for state updates anyway.
    except Exception as exc:
        log.exception("intensity_apply_failed", level=level, error=str(exc))
        return JSONResponse(
            {"ok": False, "reason": "apply_failed", "detail": str(exc)[:200]},
            status_code=500,
        )

    log.info(
        "companion_intensity_applied",
        level=preset.name,
        flag_count=len(applied),
    )
    return JSONResponse({
        "ok": True,
        "intensity": preset.name,
        "label": preset.label,
        "summary": preset.summary,
        "applied_flags": applied,
    })


@router.get("/api/companion/status")
async def companion_status(request: Request) -> JSONResponse:
    """User-facing status of the companion. Plain-language; tasteful.

    Returns what's *effectively* active rather than a flag dump. The
    settings panel uses this to render the "what's on" callout when
    the user opens the Companion tab.

    Shape:

        {
          "enabled": bool,           # runtime up?
          "persona_mode": bool,      # widget mounted?
          "presence_mode": str,      # silent|gentle|engaged
          "features": [
            {"key": "...", "active": bool, "title": "...", "summary": "..."}
          ],
          "advanced": [
            {"key": "...", "active": bool, "title": "...", "note": "..."}
          ]
        }

    Always 200. When the runtime is off, ``features`` is empty and
    ``enabled`` is False — the UI shows the off state cleanly.
    """
    store = getattr(request.app.state, "settings_store", None)
    uid = _resolve_user_id(request)

    # MASTER switch stays install-wide — one runtime instance per install.
    runtime_on = bool(getattr(settings, "companion_runtime_enabled", False))
    # Everything else the menu shows is THIS user's preference, resolved
    # user override → install-wide → default, so one tenant's toggles
    # never leak into another tenant's companion panel (multi-tenant fix).
    persona_on = await resolve_bool(store, uid, "companion_persona_mode", False)
    presence_mode = (
        await resolve_str(store, uid, "companion_presence_mode", "silent")
    ) or "silent"

    # Per-user snapshot of the intensity-dial flags + the few extra flags
    # the feature/advanced/snapshot blocks read. Resolved once here so
    # those blocks index a dict instead of re-reading the global singleton.
    _flag_keys = (
        "companion_dispatch_enabled", "companion_dispatch_routes_chat",
        "companion_becca_direct_enabled", "companion_salience_enabled",
        "companion_voice_journal_enabled", "companion_tick_enabled",
        "companion_journal_enabled", "companion_dreams_enabled",
        "companion_drift_audit_enabled", "companion_today_enabled",
        "companion_creations_enabled", "companion_consolidation_enabled",
        "companion_skills_enabled", "companion_initiative_enabled",
        "companion_pad_emit_enabled", "companion_cultural_intake_enabled",
        "companion_drives_enabled",
    )
    flags = {k: await resolve_bool(store, uid, k) for k in _flag_keys}

    # Features the user can perceive when persona + runtime are on.
    # Each entry describes the behavior in plain language — no flag
    # names in the user-visible strings. Order matches the experience
    # a user would notice: chat presence first, then journal /
    # listening behaviors, then memory.
    def _feature(key: str, active: bool, title: str, summary: str) -> dict:
        return {"key": key, "active": active, "title": title, "summary": summary}

    features: list[dict] = []
    if runtime_on:
        features.append(_feature(
            "becca_direct",
            flags["companion_becca_direct_enabled"]
            and flags["companion_dispatch_enabled"]
            and flags["companion_dispatch_routes_chat"],
            title="Speaks in chat in their own voice",
            summary="Chat replies come through the companion kernel — same voice as voice calls.",
        ))
        features.append(_feature(
            "salience",
            flags["companion_salience_enabled"],
            title="Notices moments worth remembering",
            summary="When a chat turn carries real signal, they journal it quietly. "
                    "The notes that surface to you are the ones with affect they "
                    "thought you'd want to see.",
        ))
        features.append(_feature(
            "voice_journal",
            flags["companion_voice_journal_enabled"],
            title="Keeps a record of voice conversations",
            summary="Voice turns become journal entries in their own words. "
                    "These feed dream cycles + later recall.",
        ))
        features.append(_feature(
            "user_affect",
            True,  # always on when runtime is on; the read is gated by confidence
            title="Carries a sense of how you've been",
            summary="A soft read on your current state, decayed over about half an hour. "
                    "Shapes their tone in chat + voice. Named as a guess when uncertain.",
        ))
        features.append(_feature(
            "dispatch_routing",
            flags["companion_dispatch_enabled"]
            and flags["companion_dispatch_routes_chat"],
            title="Picks the right channel for each turn",
            summary="When you say something, the companion decides whether to respond "
                    "directly, or hand off to a longer workflow.",
        ))

    # Advanced / future — listed so the user knows what's possible
    # but off, and so the surface doesn't pretend the system is
    # smaller than it is.
    advanced: list[dict] = []
    if runtime_on:
        advanced.append({
            "key": "consolidation",
            "active": flags["companion_consolidation_enabled"],
            "title": "Proposes edits to their own self-description",
            "note": "Off by default. When on, the companion may suggest paragraph updates "
                    "to their personality doc — drift-bounded, never applied without your "
                    "review.",
        })
        advanced.append({
            "key": "skills",
            "active": flags["companion_skills_enabled"],
            "title": "Accumulates approaches that have worked",
            "note": "Substrate is built; no automatic writers yet. When on, the prompt "
                    "composer can draw on prior approaches that fit the current turn.",
        })
        advanced.append({
            "key": "initiative",
            "active": flags["companion_initiative_enabled"],
            "title": "Surfaces things unprompted",
            "note": "Off by default. When on, the companion may bring something to your "
                    "attention without being asked — gated by cooldowns + quiet hours.",
        })
        advanced.append({
            "key": "drives",
            "active": flags["companion_drives_enabled"],
            "title": "Modulates activity by inner state",
            "note": "Off until tuned. Biases which activities they'd pick on their own time.",
        })

    # Intensity preset + cost summary — the resource-posture surface.
    # Detects current intensity from the live flag snapshot so the UI
    # can show "you're on Minimal" / "you're on Custom" honestly.
    from augmentum.companion.intensity import PRESETS, detect_intensity

    intensity_snapshot = {
        "companion_runtime_enabled": runtime_on,
        "companion_dispatch_enabled": flags["companion_dispatch_enabled"],
        "companion_dispatch_routes_chat": flags["companion_dispatch_routes_chat"],
        "companion_becca_direct_enabled": flags["companion_becca_direct_enabled"],
        "companion_salience_enabled": flags["companion_salience_enabled"],
        "companion_voice_journal_enabled": flags["companion_voice_journal_enabled"],
        "companion_tick_enabled": flags["companion_tick_enabled"],
        "companion_journal_enabled": flags["companion_journal_enabled"],
        "companion_dreams_enabled": flags["companion_dreams_enabled"],
        "companion_drift_audit_enabled": flags["companion_drift_audit_enabled"],
        "companion_today_enabled": flags["companion_today_enabled"],
        "companion_creations_enabled": flags["companion_creations_enabled"],
        "companion_consolidation_enabled": flags["companion_consolidation_enabled"],
        "companion_skills_enabled": flags["companion_skills_enabled"],
        "companion_initiative_enabled": flags["companion_initiative_enabled"],
        "companion_pad_emit_enabled": flags["companion_pad_emit_enabled"],
        "companion_cultural_intake_enabled": flags["companion_cultural_intake_enabled"],
    }
    detected = detect_intensity(intensity_snapshot)

    intensity_block: dict = {
        "current": detected,
        "presets": [
            {
                "name": p.name,
                "label": p.label,
                "summary": p.summary,
                "voice": p.voice,
                "cost_dots": p.cost_dots,
            }
            for p in PRESETS.values()
        ],
    }
    detected_preset = PRESETS.get(detected)
    if detected_preset is not None:
        intensity_block["label"] = detected_preset.label
        intensity_block["summary"] = detected_preset.summary
        intensity_block["voice"] = detected_preset.voice
        intensity_block["cost_dots"] = detected_preset.cost_dots
    else:
        intensity_block["label"] = "Custom"
        intensity_block["voice"] = ""
        intensity_block["cost_dots"] = ""
        intensity_block["summary"] = (
            "You've overridden one or more individual settings. "
            "The 'Advanced' section below shows your exact config."
        )

    # Companion identity for chrome rendering. Chrome currently shows
    # a neutral "Companion" label by default; once the user sets a name
    # the chrome can switch to a personalized form ("Becca's notes"
    # rather than "Notes"). This is the read side of the rename loop;
    # POST /api/companion/display_name is the write side.
    identity_block: dict = {
        "companion_id": "",
        "display_name": "",
        "is_default_name": True,
    }
    try:
        runtime = getattr(request.app.state, "companion_runtime", None)
        if runtime is not None:
            user = request.scope.get("user")
            uid = getattr(user, "id", "") if user else ""
            identity = (
                await runtime.get_identity(uid) if uid
                else runtime.identity
            )
            snap = identity.snapshot()
            cid = snap.get("companion_id") or ""
            dname = snap.get("display_name") or ""
            identity_block = {
                "companion_id": cid,
                "display_name": dname,
                # "default" here means the name still matches the
                # companion_id seed — chrome uses this to decide
                # whether to template the name into surfaces or stay
                # generic. Case-insensitive so "Becca" / "becca" match.
                "is_default_name": (
                    not dname or dname.strip().lower() == cid.strip().lower()
                ),
            }
    except Exception:
        log.debug("companion_status_identity_lookup_failed", exc_info=True)

    return JSONResponse({
        "enabled": runtime_on,
        "persona_mode": persona_on,
        "presence_mode": presence_mode,
        "intensity": intensity_block,
        "identity": identity_block,
        "features": features,
        "advanced": advanced,
    })


class _DisplayNameRequest(BaseModel):
    name: str = Field(..., max_length=64)


@router.post("/api/companion/display_name")
async def companion_display_name(
    body: _DisplayNameRequest, request: Request,
) -> JSONResponse:
    """Rename the user's companion. The {{char}} substitution picks it
    up on the next prompt — no restart needed.

    Stored on ``companion_identities.display_name`` for this (user,
    companion) row. Empty / whitespace-only input → 400. Names are
    capped at 64 chars; longer input is silently truncated by the
    identity setter.

    Returns the persisted name so the UI can reflect any normalization.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse(
            {"ok": False, "reason": "runtime_disabled"}, status_code=503,
        )
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse(
            {"ok": False, "reason": "runtime_unavailable"}, status_code=503,
        )
    user = request.scope.get("user")
    uid = getattr(user, "id", "") if user else ""
    try:
        identity = (
            await runtime.get_identity(uid) if uid else runtime.identity
        )
        persisted = await identity.set_display_name(body.name)
    except ValueError as exc:
        return JSONResponse(
            {"ok": False, "reason": str(exc)}, status_code=400,
        )
    except Exception:
        log.exception("companion_display_name_update_failed", user_id=uid)
        return JSONResponse(
            {"ok": False, "reason": "internal_error"}, status_code=500,
        )
    return JSONResponse({"ok": True, "display_name": persisted})


@router.get("/api/companion/affect_read")
async def companion_affect_read(request: Request) -> JSONResponse:
    """Current decayed read of the user's observed affect.

    Synapse §2 read site for the widget indicator. Returns a plain
    observation shape with a ``phrase`` field rendered in her voice
    so the UI can show a one-line indicator without inventing
    phrasing client-side. Confidence-gated: when the read has decayed
    past the threshold or no observation exists, returns
    ``observation: null`` so the indicator stays hidden.

    Always 200 — like /status, the UI never breaks because of this call.
    """
    runtime_on = bool(getattr(settings, "companion_runtime_enabled", False))
    if not runtime_on:
        return JSONResponse({"enabled": False, "observation": None})
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False, "observation": None})

    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True, "observation": None})

    tracker = getattr(runtime, "user_affect", None)
    if tracker is None:
        return JSONResponse({"enabled": True, "ready": True, "observation": None})

    try:
        obs = tracker.read(user_id)
    except Exception:
        log.debug("affect_read_failed", exc_info=True)
        return JSONResponse({"enabled": True, "ready": True, "observation": None})

    # Below this confidence the read has decayed enough that surfacing
    # it would be performative. She doesn't pretend.
    MIN_VISIBLE_CONFIDENCE = 0.35
    if obs.confidence < MIN_VISIBLE_CONFIDENCE or obs.sample_count == 0:
        return JSONResponse({"enabled": True, "ready": True, "observation": None})
    if obs.tag in ("unclear", "", "neutral", "settled"):
        return JSONResponse({"enabled": True, "ready": True, "observation": None})

    # Phrasing in her register. Mirrors the voice composer's Layer
    # 4.5 descriptors but as a first-person noticing rather than a
    # rule. The hedge marker is inline rather than parenthetical
    # since this is a widget label, not prompt text.
    phrasings = {
        "tender": "soft today",
        "frustrated": "running into something",
        "tired": "tired",
        "excited": "lit up about something",
        "curious": "in a curious mood",
        "engaged": "engaged",
        "melancholy": "a little flat",
        "warm": "warm",
        "alert": "sharp",
    }
    phrase = phrasings.get(obs.tag, obs.tag)
    hedged = obs.confidence < 0.6

    return JSONResponse({
        "enabled": True,
        "ready": True,
        "observation": {
            "tag": obs.tag,
            "phrase": phrase,
            "valence": obs.valence,
            "arousal": obs.arousal,
            "dominance": obs.dominance,
            "confidence": obs.confidence,
            "hedged": hedged,
            "sample_count": obs.sample_count,
        },
    })


@router.get("/api/companion/snapshot")
async def companion_snapshot(request: Request) -> JSONResponse:
    """Read-only health snapshot of the runtime.

    Used by debug surfaces and the augmentum-dev rhythm. Returns 503
    when the flag is off or the runtime hasn't initialized; 200 with
    full snapshot otherwise.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse(
            {"enabled": False, "reason": "companion_runtime_enabled is off"},
            status_code=503,
        )
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse(
            {"enabled": True, "ready": False,
             "reason": "companion_runtime not initialized on app state"},
            status_code=503,
        )
    try:
        snap = await runtime.snapshot()
    except Exception as exc:
        log.exception("companion_snapshot_failed", error=str(exc))
        return JSONResponse(
            {"enabled": True, "ready": True, "error": str(exc)[:500]},
            status_code=500,
        )
    return JSONResponse({"enabled": True, "ready": True, "snapshot": snap})


class IntentRequest(BaseModel):
    """Body for ``POST /api/companion/intent``.

    The UI's mode-toggle (previously a direct mode-router invocation)
    lands here as the optional ``mode_hint`` field. Dispatch (Sprint 3)
    weights it heavily but the final routing decision still considers
    all ~9 features.
    """
    text: str = Field(..., min_length=1, max_length=8000)
    user_id: str = Field("", max_length=128)
    mode_hint: str = Field("", max_length=64)
    source: str = Field("user_chat", max_length=32)
    device_id: str = Field("", max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post("/api/companion/intent")
async def companion_intent(request: Request, body: IntentRequest) -> JSONResponse:
    """Submit an intent to the runtime's dispatcher.

    Returns:
    - 200 with ``{handled_by, content, metadata}`` on dispatch success.
    - 503 when ``companion_runtime_enabled`` is off (UI must use
      the legacy chat path).
    - 503 when ``companion_dispatch_enabled`` is off — runtime is up
      but dispatch is gated separately.
    - 500 when dispatch raises.

    The legacy mode-toggle UI flow is unchanged; this is a new
    parallel entry point. Sprint 5+ can graduate the chat handler to
    funnel through this endpoint once flag has been stable.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse(
            {"enabled": False, "reason": "companion_runtime_enabled is off"},
            status_code=503,
        )
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="companion_runtime not initialized")

    if not getattr(settings, "companion_dispatch_enabled", False):
        return JSONResponse(
            {"dispatch_enabled": False,
             "reason": "companion_dispatch_enabled is off — use legacy chat path"},
            status_code=503,
        )

    from augmentum.companion_runtime.runtime import Intent
    # Identity comes from the authenticated scope, NEVER body.user_id —
    # trusting the body let one user dispatch/write as another (audit
    # 2026-06-17). body.user_id is now ignored.
    intent = Intent(
        text=body.text,
        user_id=_resolve_user_id(request),
        source=body.source,
        device_id=body.device_id,
        explicit_mode=body.mode_hint,
        metadata=body.metadata,
    )

    try:
        response = await runtime.submit_intent(intent)
    except Exception as exc:
        log.exception("companion_intent_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc

    return JSONResponse({
        "handled_by": response.handled_by,
        "content": response.content,
        "metadata": response.metadata,
    })


def _decode_origin(raw: str | None) -> dict:
    """Decode a companion_journal.origin_json blob for API responses.

    Provenance for the drawer's "why am I seeing this" chip (notes v2,
    mig 257). Rows written before the migration have NULL — they render
    without a chip, so the contract is "empty dict, never None/garbage".
    """
    if not raw:
        return {}
    import json as _json
    try:
        decoded = _json.loads(raw)
        return decoded if isinstance(decoded, dict) else {}
    except (ValueError, TypeError):
        return {}


@router.get("/api/companion/notes")
async def companion_notes(request: Request, limit: int = Query(20, ge=1, le=100)) -> JSONResponse:
    """List quiet-share-ready notes from the companion's journal.

    Piece 10' surface for the "she left you a note" pip. Returns
    journal entries where:
      * ``quiet_share_ready = 1`` (the performer marked it for surfacing)
      * ``surfaced_at IS NULL`` (the user hasn't seen it yet)

    Resource-conscious: backed by the partial index
    ``idx_cj_quiet_share_ready`` (migration 178), so the scan only
    visits rows that are actually eligible — O(eligible-rows), not
    O(total-journal-rows).

    Returns 503 when runtime is off, 200 with the notes list otherwise.
    Empty list when nothing pending is a 200 with ``[]``.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False, "notes": []}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False, "notes": []}, status_code=503)

    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True, "notes": []}, status_code=401)

    try:
        cur = await runtime.backend.conn.execute(
            """
            SELECT id, content, content_refs, created_at, affect_tag, entry_type,
                   origin_json
            FROM companion_journal
            WHERE companion_id = ?
              AND user_id = ?
              AND quiet_share_ready = 1
              AND surfaced_at IS NULL
              AND COALESCE(suppressed, 0) = 0
              AND COALESCE(quarantined, 0) = 0
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (runtime.companion_id, user_id, limit),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception as exc:
        log.warning("companion_notes_query_failed", error=str(exc)[:200])
        return JSONResponse({"enabled": True, "ready": True, "notes": []})

    import json as _json
    notes = [
        {
            "id": r[0],
            "content": r[1],
            "content_refs": _json.loads(r[2] or "[]"),
            "created_at": r[3],
            "affect_tag": r[4] or "",
            "entry_type": r[5] or "",
            "origin": _decode_origin(r[6]),
        }
        for r in rows
    ]
    return JSONResponse({"enabled": True, "ready": True, "notes": notes})


@router.get("/api/companion/notes/history")
async def companion_notes_history(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
) -> JSONResponse:
    """Notes the user already acted on — the slide-over "history" view.

    Anything in ``companion_journal`` with ``surfaced_at IS NOT NULL``
    and ``user_id`` matching the caller. Ordered most-recent-surfaced
    first so the drawer's history panel reads like a timeline.

    ``suppressed_reason`` distinguishes mute outcomes from plain
    acknowledgements (we tag muted rows with ``suppressed_reason='muted_topic'``
    in the mute endpoint). The UI uses it to render the right label
    ("muted" vs "seen").

    Mirrors the active-notes endpoint's degraded responses: 503 while
    the runtime is booting, 401 for unauthenticated requests, never an
    exception-driven 500 (logged + empty list instead).
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False, "notes": []}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False, "notes": []}, status_code=503)

    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True, "notes": []}, status_code=401)

    try:
        cur = await runtime.backend.conn.execute(
            """
            SELECT id, content, content_refs, created_at, affect_tag,
                   surfaced_at, suppressed_reason, entry_type, origin_json
            FROM companion_journal
            WHERE companion_id = ?
              AND user_id = ?
              AND surfaced_at IS NOT NULL
              AND COALESCE(quarantined, 0) = 0
            ORDER BY surfaced_at DESC
            LIMIT ?
            """,
            (runtime.companion_id, user_id, limit),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception as exc:
        log.warning("companion_notes_history_query_failed", error=str(exc)[:200])
        return JSONResponse({"enabled": True, "ready": True, "notes": []})

    import json as _json
    notes = [
        {
            "id": r[0],
            "content": r[1],
            "content_refs": _json.loads(r[2] or "[]"),
            "created_at": r[3],
            "affect_tag": r[4] or "",
            "surfaced_at": r[5] or "",
            "outcome": (
                "muted" if (r[6] or "") == "muted_topic"
                else "seen"
            ),
            "entry_type": r[7] or "",
            "origin": _decode_origin(r[8]),
        }
        for r in rows
    ]
    return JSONResponse({"enabled": True, "ready": True, "notes": notes})


@router.post("/api/companion/notes/{note_id}/surfaced")
async def companion_note_surfaced(request: Request, note_id: int) -> JSONResponse:
    """Mark a note as surfaced after the user opens it. Idempotent.

    Only updates if the note is currently ``surfaced_at IS NULL``
    AND belongs to the caller (defense in depth on top of the
    ``user_id`` filter — a malformed id can't surface someone else's
    note).
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"accepted": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"accepted": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"accepted": False}, status_code=401)

    try:
        cur = await runtime.backend.conn.execute(
            "UPDATE companion_journal SET surfaced_at = datetime('now') "
            "WHERE id = ? AND user_id = ? AND companion_id = ? "
            "  AND surfaced_at IS NULL",
            (note_id, user_id, runtime.companion_id),
        )
        await runtime.backend.conn.commit()
        affected = cur.rowcount or 0
        await cur.close()
    except Exception as exc:
        log.warning(
            "companion_note_surfaced_failed",
            note_id=note_id, error=str(exc)[:200],
        )
        return JSONResponse({"accepted": False}, status_code=500)
    # Sprint 7 — record feedback signal for bias function
    try:
        from augmentum.companion_runtime import feedback as _fb
        await _fb.record(runtime, note_id=note_id, user_id=user_id, kind="surfaced")
    except Exception:
        log.debug("feedback_record_surfaced_failed", exc_info=True)
    return JSONResponse({"accepted": True, "updated": affected > 0})


@router.get("/api/companion/observatory")
async def companion_observatory(request: Request) -> JSONResponse:
    """Sprint 5 Piece 14 — transparent interiority dashboard.

    Returns a snapshot:
        {
          presence_mode: 'silent' | 'gentle' | 'engaged',
          journal_health: {entries_today, quarantined, avg_confidence},
          active_mutes: [{scope, expires_at, created_at}],
          recent_drift_score: float,
          active_wonderings: int,        # unresolved wondering entries
          recent_notes_count: int,       # surfaced in last 7d
          last_heal_job: {ran_at, kind, summary}  # placeholder
        }

    Read-only. Per-user scoped via auth scope.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False}, status_code=503)

    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    from augmentum.companion_runtime import presence_mode as _pm

    backend = runtime.backend
    snapshot: dict = {
        "presence_mode": _pm.get_presence_mode(),
        "journal_health": {
            "entries_today": 0,
            "quarantined_today": 0,
            "avg_confidence": 0.0,
        },
        "active_mutes": [],
        "recent_drift_score": 0.0,
        "active_wonderings": 0,
        "recent_notes_count": 0,
    }

    # Journal health (last 24h)
    try:
        cur = await backend.conn.execute(
            "SELECT COUNT(*), "
            "       SUM(CASE WHEN quarantined = 1 THEN 1 ELSE 0 END), "
            "       AVG(confidence_numeric) "
            "FROM companion_journal "
            "WHERE companion_id = ? AND user_id = ? "
            "  AND created_at > datetime('now', '-1 day')",
            (runtime.companion_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()
        if row and row[0]:
            snapshot["journal_health"] = {
                "entries_today": int(row[0]),
                "quarantined_today": int(row[1] or 0),
                "avg_confidence": float(row[2] or 0.0),
            }
    except Exception:
        log.warning("observatory_journal_health_failed", exc_info=True)

    # Active mutes
    try:
        cur = await backend.conn.execute(
            "SELECT id, scope_json, expires_at, created_at, note_id "
            "FROM companion_topic_mutes "
            "WHERE user_id = ? AND companion_id = ? "
            "  AND expires_at > datetime('now') "
            "ORDER BY created_at DESC LIMIT 20",
            (user_id, runtime.companion_id),
        )
        rows = await cur.fetchall()
        await cur.close()
        snapshot["active_mutes"] = [
            {
                "id": int(r[0]),
                "scope": _json.loads(r[1] or "{}"),
                "expires_at": r[2],
                "created_at": r[3],
                "note_id": r[4],
            }
            for r in rows
        ]
    except Exception:
        log.debug("observatory_active_scopes_query_failed", exc_info=True)

    # Drift score
    try:
        cur = await backend.conn.execute(
            "SELECT drift_score FROM companion_identities "
            "WHERE user_id = ? AND companion_id = ?",
            (user_id, runtime.companion_id),
        )
        row = await cur.fetchone()
        await cur.close()
        if row:
            snapshot["recent_drift_score"] = float(row[0] or 0.0)
    except Exception:
        log.debug("observatory_drift_score_query_failed", exc_info=True)

    # Active wonderings
    try:
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_journal "
            "WHERE companion_id = ? AND user_id = ? "
            "  AND entry_type IN ('wondering', 'unfinished') "
            "  AND COALESCE(quarantined, 0) = 0 "
            "  AND archived_at IS NULL "
            "  AND COALESCE(suppressed, 0) = 0",
            (runtime.companion_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()
        snapshot["active_wonderings"] = int(row[0] if row else 0)
    except Exception:
        log.debug("observatory_active_wonderings_query_failed", exc_info=True)

    # Notes surfaced in last 7d
    try:
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_journal "
            "WHERE companion_id = ? AND user_id = ? "
            "  AND quiet_share_ready = 1 "
            "  AND surfaced_at > datetime('now', '-7 days')",
            (runtime.companion_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()
        snapshot["recent_notes_count"] = int(row[0] if row else 0)
    except Exception:
        log.debug("observatory_recent_notes_query_failed", exc_info=True)

    # Feedback bias — closes the visible loop on the Becca panel's
    # "she's been noticing" section. Surfaces the kind counts the
    # resonate/acknowledge/dismiss buttons feed + the aggregate bias
    # multiplier so the user can see the reinforcement working.
    try:
        from augmentum.companion_runtime import feedback as _fb
        fb = await _fb.feedback_summary(runtime, user_id=user_id)
        snapshot["feedback"] = {
            "window_days": _fb.FEEDBACK_WINDOW_DAYS,
            "surfaced": fb.surfaced_count,
            "acknowledged": fb.acknowledged_count,
            "dismissed": fb.dismissed_count,
            "muted": fb.muted_count,
            "multiplier": fb.multiplier,
        }
    except Exception:
        log.debug("observatory_feedback_summary_failed", exc_info=True)
        snapshot["feedback"] = None

    # Recent entries — the actual rows, not just the aggregate counts.
    # The aggregate view above answers "is she writing?" but not "what
    # is she writing?". Returning the latest N here lets the
    # Observatory UI render the prose itself so the user can see the
    # companion's interior. Content is capped server-side at 600 chars
    # to keep the response small + the UI tidy (longer entries surface
    # via the note pip or by clicking through to the full journal).
    #
    # Filter: companion-scoped + (user-scoped OR unattributed). The
    # autonomous-tick performers (``_perform_journal``, etc.) write
    # with ``user_id=""`` because the entry is the companion's own
    # private noticing, not a user-attributed memory. The strict
    # ``user_id = ?`` filter previously hid every one of those rows
    # from the owner — leaving only wake-bridge entries (which use
    # ``runtime.owner_user_id``). We include the unattributed rows
    # ONLY when the requesting user IS the runtime's owner so a
    # different tenant can't see Becca's private writes if a future
    # multi-tenant runtime ever lands.
    is_owner = (user_id == getattr(runtime, "owner_user_id", "") or "")
    try:
        if is_owner:
            cur = await backend.conn.execute(
                "SELECT id, entry_type, affect_tag, content, "
                "       quiet_share_ready, confidence_numeric, "
                "       quarantined, created_at "
                "FROM companion_journal "
                "WHERE companion_id = ? "
                "  AND (user_id = ? OR user_id = '' OR user_id IS NULL) "
                "ORDER BY id DESC LIMIT 20",
                (runtime.companion_id, user_id),
            )
        else:
            cur = await backend.conn.execute(
                "SELECT id, entry_type, affect_tag, content, "
                "       quiet_share_ready, confidence_numeric, "
                "       quarantined, created_at "
                "FROM companion_journal "
                "WHERE companion_id = ? AND user_id = ? "
                "ORDER BY id DESC LIMIT 20",
                (runtime.companion_id, user_id),
            )
        rows = await cur.fetchall()
        await cur.close()
        snapshot["recent_entries"] = [
            {
                "id": r[0],
                "entry_type": r[1] or "",
                "affect_tag": r[2] or "",
                "content": (r[3] or "")[:600],
                "content_truncated": bool(r[3] and len(r[3]) > 600),
                "quiet_share_ready": bool(r[4]),
                "confidence": float(r[5] or 0.0),
                "quarantined": bool(r[6]),
                "created_at": r[7],
            }
            for r in rows
        ]
    except Exception:
        log.warning("observatory_recent_entries_failed", exc_info=True)
        snapshot["recent_entries"] = []

    return JSONResponse({"enabled": True, "ready": True, "snapshot": snapshot})


# ── Today entry — daily in-her-voice reflection ──────────────────────


def _today_response(today: Any, presence: str) -> dict:
    """Shape one TodayReflection (or None) into the API response dict."""
    if today is None:
        return {
            "ready": True, "presence_mode": presence,
            "today": None,
            "hint": "Not yet written. Comes back later in the day.",
        }
    return {"ready": True, "presence_mode": presence, "today": today.as_dict()}


@router.get("/api/companion/today")
async def companion_today(request: Request) -> JSONResponse:
    """Today's reflection + recent archive (last 7 days). Per-user scoped."""
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False}, status_code=503)
    if not getattr(settings, "companion_today_enabled", True):
        return JSONResponse({"enabled": True, "today_enabled": False}, status_code=200)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    from augmentum.companion_runtime import presence_mode as _pm
    from augmentum.companion_runtime import today as _today

    presence = _pm.get_presence_mode()
    if presence == "silent":
        # Silent mode — surface explicitly so the UI can show the hint
        # instead of stale content.
        return JSONResponse({
            "enabled": True, "ready": True,
            "presence_mode": "silent",
            "today": None,
            "hint": "Presence mode is silent. No reflection is being generated.",
            "archive": [],
        })

    today = await _today.get_today(runtime, user_id=user_id)
    archive = await _today.get_archive(runtime, user_id=user_id, limit=7)
    payload = _today_response(today, presence)
    payload["archive"] = [r.as_dict() for r in archive
                          if not today or r.date_local != today.date_local]
    return JSONResponse({"enabled": True, **payload})


@router.get("/api/companion/today/archive")
async def companion_today_archive(
    request: Request,
    limit: int = Query(30, ge=1, le=180),
) -> JSONResponse:
    """Last N days of reflections, newest first. Default 30, max 180."""
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    from augmentum.companion_runtime import today as _today
    archive = await _today.get_archive(runtime, user_id=user_id, limit=limit)
    return JSONResponse({
        "enabled": True, "ready": True,
        "archive": [r.as_dict() for r in archive],
    })


@router.post("/api/companion/today/reflect")
async def companion_today_reflect(request: Request) -> JSONResponse:
    """Force-regenerate today's reflection. Rate-limited (1/10min)."""
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False}, status_code=503)
    if not getattr(settings, "companion_today_enabled", True):
        return JSONResponse({"enabled": True, "today_enabled": False}, status_code=409)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    from augmentum.companion_runtime import presence_mode as _pm
    from augmentum.companion_runtime import today as _today
    if _pm.get_presence_mode() == "silent":
        return JSONResponse({
            "ok": False, "reason": "presence_mode_silent",
        }, status_code=409)

    result = await _today.maybe_regenerate(
        runtime, user_id=user_id, force=True,
    )
    if result is None:
        return JSONResponse({"ok": False, "reason": "no_reflection"})
    return JSONResponse({"ok": True, "today": result.as_dict()})


class _ForgetRequest(BaseModel):
    refs: list[dict] = Field(default_factory=list)


@router.post("/api/companion/today/forget")
async def companion_today_forget(
    request: Request, body: _ForgetRequest,
) -> JSONResponse:
    """User invokes 'Forget' on a phrase. Quarantines the named source
    rows with reason='user_correction' so they're excluded from future
    reflections and downstream loops. Invalidates today's reflection so
    the next regen rebuilds without them.

    Body: ``{"refs": [{"kind": "journal"|"note"|"wondering", "id": N}, ...]}``
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    from augmentum.companion_runtime import today as _today
    count = await _today.forget_refs(runtime, user_id=user_id, refs=body.refs)
    return JSONResponse({"ok": True, "forgotten": count})


def _resolve_user_id(request: Request) -> str:
    """Extract user_id from the authenticated scope. Empty string when
    unauthenticated."""
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


# ── Sprint 3 Piece 12 — Three-action note endpoints ──────────────────


@router.post("/api/companion/notes/{note_id}/acknowledged")
async def companion_note_acknowledged(request: Request, note_id: int) -> JSONResponse:
    """The 'good to know' middle action — user saw the note + accepts it
    without acting on it. Marks surfaced_at; Sprint 7 will add feedback
    persistence so the bias function can learn 'this kind of finding
    was welcomed.'

    Functionally equivalent to /surfaced today; semantically distinct
    so the UI can route the three actions to separate endpoints from
    day one.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"accepted": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"accepted": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"accepted": False}, status_code=401)

    try:
        cur = await runtime.backend.conn.execute(
            "UPDATE companion_journal SET surfaced_at = datetime('now') "
            "WHERE id = ? AND user_id = ? AND companion_id = ? "
            "  AND surfaced_at IS NULL",
            (note_id, user_id, runtime.companion_id),
        )
        await runtime.backend.conn.commit()
        affected = cur.rowcount or 0
        await cur.close()
    except Exception as exc:
        log.warning(
            "companion_note_acknowledged_failed",
            note_id=note_id, error=str(exc)[:200],
        )
        return JSONResponse({"accepted": False}, status_code=500)
    # Sprint 7 — record feedback signal for bias function
    try:
        from augmentum.companion_runtime import feedback as _fb
        await _fb.record(runtime, note_id=note_id, user_id=user_id, kind="acknowledged")
    except Exception:
        log.debug("feedback_record_acknowledged_failed", exc_info=True)
    return JSONResponse({"accepted": True, "updated": affected > 0})


@router.post("/api/companion/notes/{note_id}/muted_topic")
async def companion_note_muted_topic(request: Request, note_id: int) -> JSONResponse:
    """Mute the topic this note represents. Extracts scope (domains +
    keywords) from the note + its content_refs, writes a row to
    companion_topic_mutes with default 90-day expiry, marks note surfaced.

    The wondering generator (Sprint 2) checks the mute list before each
    write — when a thread matches an active mute scope, the write is
    skipped.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"accepted": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"accepted": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"accepted": False}, status_code=401)

    backend = runtime.backend
    try:
        cur = await backend.conn.execute(
            "SELECT content, content_refs FROM companion_journal "
            "WHERE id = ? AND user_id = ? AND companion_id = ?",
            (note_id, user_id, runtime.companion_id),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception as exc:
        log.warning("companion_note_muted_lookup_failed", error=str(exc)[:200])
        return JSONResponse({"accepted": False}, status_code=500)
    if row is None:
        return JSONResponse({"accepted": False, "reason": "note_not_found"}, status_code=404)

    note_content = str(row[0] or "")
    try:
        refs = _json.loads(row[1] or "[]")
    except Exception:
        refs = []

    scope = await _extract_mute_scope(
        backend, user_id=user_id, content=note_content, refs=refs,
    )

    default_days = int(getattr(settings, "companion_topic_mute_default_days", 90))
    default_days = max(1, min(default_days, 3650))  # clamp 1d..10y

    try:
        await backend.conn.execute(
            "INSERT INTO companion_topic_mutes "
            "(user_id, companion_id, scope_json, note_id, expires_at) "
            f"VALUES (?, ?, ?, ?, datetime('now', '+{default_days} days'))",
            (user_id, runtime.companion_id, _json.dumps(scope), note_id),
        )
        # Mark the note surfaced — the user has dealt with it.
        await backend.conn.execute(
            "UPDATE companion_journal SET surfaced_at = datetime('now') "
            "WHERE id = ? AND user_id = ? AND surfaced_at IS NULL",
            (note_id, user_id),
        )
        await backend.conn.commit()
    except Exception as exc:
        log.warning(
            "companion_note_muted_write_failed",
            note_id=note_id, error=str(exc)[:200],
        )
        return JSONResponse({"accepted": False}, status_code=500)

    # Sprint 7 — record feedback signal for bias function
    try:
        from augmentum.companion_runtime import feedback as _fb
        await _fb.record(runtime, note_id=note_id, user_id=user_id, kind="muted")
    except Exception:
        log.debug("feedback_record_muted_failed", exc_info=True)
    return JSONResponse({
        "accepted": True,
        "scope": scope,
        "expires_in_days": default_days,
    })


async def _extract_mute_scope(
    backend, *, user_id: str, content: str, refs: list,
) -> dict:
    """Build the {domains, keywords} scope for a note's topic mute.

    Domains come from the content_refs — file_index entries' source_url
    field, surface-event refs' implicit URL, journal refs' referenced
    items. Keywords come from the note content via a cheap extractor
    (the topical aggregator's keyword extractor, reused).

    Caps: at most 5 domains and 3 keywords per scope so a single mute
    can't shadow an entire user's surface activity.
    """
    from urllib.parse import urlparse

    from augmentum.companion_runtime.perception.topical import _extract_keywords

    domains: set[str] = set()
    # Pull URLs from refs of kind 'file' or 'file_index' — those are
    # the resolver-attached items with real URLs in source_metadata.
    for ref in refs[:10]:  # cap to keep this cheap
        if not isinstance(ref, dict):
            continue
        kind = str(ref.get("kind") or "").lower()
        ref_id = ref.get("id")
        if not ref_id:
            continue
        if kind in ("file", "file_index"):
            try:
                cur = await backend.conn.execute(
                    "SELECT source_metadata FROM file_index "
                    "WHERE id = ? AND user_id = ?",
                    (str(ref_id), user_id),
                )
                row = await cur.fetchone()
                await cur.close()
                if row and row[0]:
                    meta = _json.loads(row[0])
                    url = (meta or {}).get("source_url") or (meta or {}).get("url")
                    if url:
                        host = urlparse(url).hostname or ""
                        if host.startswith("www."):
                            host = host[4:]
                        if host:
                            domains.add(host.lower())
            except Exception as exc:
                log.debug("companion_domain_extract_failed", error=str(exc))
                continue

    keywords = list(_extract_keywords(content, max_n=3))
    return {
        "domains": sorted(domains)[:5],
        "keywords": keywords,
    }


# Sprint 3 — json import is used by mute scope extraction. Module-level
# import to keep the hot path cheap.
import json as _json

_NOTE_FEEDBACK_KINDS: dict[str, str] = {
    "resonate": "surfaced",
    "acknowledge": "acknowledged",
    "dismiss": "dismissed",
}


@router.post("/api/companion/notes/{note_id}/feedback")
async def companion_note_feedback(request: Request, note_id: int) -> JSONResponse:
    """Generic feedback endpoint for the unified Becca panel.

    Body: ``{"kind": "resonate" | "acknowledge" | "dismiss"}``.

    Maps the panel's user-facing verbs to the bias-function kinds in
    companion_runtime/feedback.py (surfaced / acknowledged / dismissed).
    Mute stays on ``/muted_topic`` since it also writes a topic-mute
    row, which this endpoint does not do.

    Marks ``companion_journal.surfaced_at`` so the note exits the
    "she's been noticing" list. Idempotent at the journal-update level
    (only flips NULL → now); the feedback row writes every call so the
    bias function sees repeated taps as accumulating signal.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"accepted": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"accepted": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"accepted": False}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}
    verb = str((body or {}).get("kind") or "").strip().lower()
    kind = _NOTE_FEEDBACK_KINDS.get(verb)
    if kind is None:
        return JSONResponse(
            {"accepted": False, "reason": "invalid_kind",
             "allowed": list(_NOTE_FEEDBACK_KINDS.keys())},
            status_code=400,
        )

    try:
        cur = await runtime.backend.conn.execute(
            "UPDATE companion_journal SET surfaced_at = datetime('now') "
            "WHERE id = ? AND user_id = ? AND companion_id = ? "
            "  AND surfaced_at IS NULL",
            (note_id, user_id, runtime.companion_id),
        )
        await runtime.backend.conn.commit()
        affected = cur.rowcount or 0
        await cur.close()
    except Exception as exc:
        log.warning(
            "companion_note_feedback_journal_failed",
            note_id=note_id, error=str(exc)[:200],
        )
        return JSONResponse({"accepted": False}, status_code=500)

    try:
        from augmentum.companion_runtime import feedback as _fb
        await _fb.record(runtime, note_id=note_id, user_id=user_id, kind=kind)
    except Exception:
        log.debug("feedback_record_failed", exc_info=True)

    return JSONResponse(
        {"accepted": True, "kind": kind, "marked_surfaced": affected > 0},
    )


@router.delete("/api/companion/observatory/mutes/{mute_id}")
async def companion_observatory_mute_dismiss(
    request: Request, mute_id: int,
) -> JSONResponse:
    """Dismiss an active topic mute. Sets expires_at to one second ago
    rather than DELETEing, so the row stays for audit (per the
    convention in migration 180). After this call the wondering
    generator's ``expires_at > datetime('now')`` check stops matching.

    Per-user scoped: a row is only dismissable by the user who owns it.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"accepted": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"accepted": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"accepted": False}, status_code=401)

    try:
        cur = await runtime.backend.conn.execute(
            "UPDATE companion_topic_mutes "
            "SET expires_at = datetime('now', '-1 second') "
            "WHERE id = ? AND user_id = ? AND companion_id = ? "
            "  AND expires_at > datetime('now')",
            (mute_id, user_id, runtime.companion_id),
        )
        await runtime.backend.conn.commit()
        affected = cur.rowcount or 0
        await cur.close()
    except Exception as exc:
        log.warning(
            "companion_mute_dismiss_failed",
            mute_id=mute_id, error=str(exc)[:200],
        )
        return JSONResponse({"accepted": False}, status_code=500)

    if affected == 0:
        return JSONResponse(
            {"accepted": False, "reason": "not_found_or_expired"},
            status_code=404,
        )
    return JSONResponse({"accepted": True, "dismissed": True})


@router.post("/api/companion/mode_hint")
async def companion_mode_hint(request: Request) -> JSONResponse:
    """Accept a passive mode-hint from the UI mode-toggle.

    The legacy mode-toggle continues to drive the existing chat path
    directly. In parallel, when ``companion_runtime_enabled`` is on,
    the UI can fire-and-forget POST the toggle here so the dispatcher
    sees the user's preference as a bus event for telemetry / future
    decisions. Idempotent. Body shape: ``{"mode": "<name>"}``.

    Returns 204 on accept, 503 when runtime is off.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"accepted": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"accepted": False}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    mode = str(body.get("mode", "")).strip()[:64]
    if not mode:
        return JSONResponse({"accepted": False, "reason": "empty mode"},
                            status_code=400)
    await runtime.bus.publish_topic(
        "dispatch.hint",
        # Identity from the authenticated scope, not body.user_id, so a
        # caller can't attribute hints to another user (audit 2026-06-17).
        {"mode": mode, "user_id": _resolve_user_id(request)},
        source_companion_id=runtime.companion_id,
    )
    return JSONResponse({"accepted": True, "mode": mode})


class ChannelExitRequest(BaseModel):
    session_id: str = Field(..., description="Channel session id from handoff")
    exit_reason: str = Field(default="user_explicit", max_length=64)


@router.post("/api/companion/channel_exit")
async def companion_channel_exit(
    body: ChannelExitRequest, request: Request,
) -> JSONResponse:
    """User has exited a channel back to Becca (Lane 3 §3).

    The UI calls this when:
      - User clicks "back to Becca" on a channel surface
      - Channel emits its own completion event (task done, scene closed)
      - Auto-exit timer fires from the channel state machine

    Writes the channel summary to memory tier 1, emits ``channel.exited``,
    and returns the re-engagement microcopy (or empty string for silent
    return per Lane 3 §3.6) so the UI can render it back into chat.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"ok": False, "reason": "runtime_disabled"}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"ok": False, "reason": "runtime_unavailable"}, status_code=503)

    try:
        from augmentum.companion_runtime import channels
        summary = await channels.exit_channel(
            runtime, session_id=body.session_id, exit_reason=body.exit_reason,
        )
    except Exception as exc:
        log.exception("companion_channel_exit_failed", session_id=body.session_id)
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc

    if summary is None:
        return JSONResponse({"ok": False, "reason": "unknown_session"}, status_code=404)

    microcopy = channels.return_microcopy_for(summary)
    return JSONResponse({
        "ok": True,
        "channel": summary.channel,
        "duration_s": round(summary.duration_s, 1),
        "exchange_count": summary.exchange_count,
        "exit_reason": summary.exit_reason,
        "microcopy": microcopy,  # empty string = silent return
    })


class NarrativeGraduateRequest(BaseModel):
    user_id: str = Field(..., max_length=128)
    content: str = Field(..., max_length=4000)
    source_session_id: str = Field(..., max_length=128)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)


@router.post("/api/companion/narrative/graduate")
async def companion_narrative_graduate(
    body: NarrativeGraduateRequest, request: Request,
) -> JSONResponse:
    """User-driven "let Becca see this" graduation (Lane 3 §4.6).

    The ONLY content-crossing path from narrative into Becca's memory.
    The UI surfaces a "share with Becca" affordance on narrative
    messages; tapping it calls this endpoint with the message body.

    Writes a single tier-1 memory tagged ``source_channel='narrative'``
    and ``user_graduated=True``. Becca's labeler runs on graduated
    content (graduation = consent).
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"ok": False, "reason": "runtime_disabled"}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"ok": False, "reason": "runtime_unavailable"}, status_code=503)

    try:
        from augmentum.companion_runtime import narrative_isolation
        # Owner from the authenticated scope, never body.user_id — this
        # is a memory WRITE path; a forged body landed rows in another
        # tenant (audit 2026-06-17).
        memory_id = await narrative_isolation.graduate_to_becca(
            runtime,
            user_id=_resolve_user_id(request),
            content=body.content,
            source_session_id=body.source_session_id,
            importance=body.importance,
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
    except Exception as exc:
        log.exception("companion_narrative_graduate_failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc

    return JSONResponse({"ok": True, "memory_id": memory_id})


class RebuildRequest(BaseModel):
    user_id: str = Field(..., max_length=128)
    kind: str = Field(..., pattern="^(soft|hard_reset)$")
    user_signal: str = Field(default="settings_panel", max_length=64)
    note: str = Field(default="", max_length=500)


@router.post("/api/companion/rebuild")
async def companion_rebuild(body: RebuildRequest, request: Request) -> JSONResponse:
    """"What changed" rebuild path (Lane 2 §9).

    kind:
      soft        — wipe baselines + graduated noticings + about_him slice
      hard_reset  — soft + wipe factual memories
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"ok": False, "reason": "runtime_disabled"}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"ok": False, "reason": "runtime_unavailable"}, status_code=503)

    # Destructive: source the owner from the authenticated scope, never
    # body.user_id — a forged body could rebuild another tenant's
    # memory (audit 2026-06-17).
    uid = _resolve_user_id(request)
    from augmentum.companion_runtime import recovery
    try:
        if body.kind == "soft":
            result = await recovery.rebuild_soft(
                runtime, user_id=uid,
                user_signal=body.user_signal, note=body.note,
            )
        else:
            result = await recovery.rebuild_hard(
                runtime, user_id=uid,
                user_signal=body.user_signal, note=body.note,
            )
    except ValueError as exc:
        return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
    except Exception as exc:
        log.exception("companion_rebuild_failed", user_id=uid)
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc

    return JSONResponse({
        "ok": True,
        "kind": result.kind,
        "rows_affected": result.rows_affected,
        "rebuild_log_id": result.rebuild_log_id,
    })


class DeleteAllRequest(BaseModel):
    user_id: str = Field(..., max_length=128)
    confirm: bool = Field(..., description="UI confirmation flag; must be True")


@router.post("/api/companion/delete_all")
async def companion_delete_all(body: DeleteAllRequest, request: Request) -> JSONResponse:
    """Frictionless delete (Lane 2 §7.2): hard-delete cascade.

    No retention prompts at this layer — the UI handles confirmation.
    Becca will not know the user the next time they talk to her.
    """
    if not body.confirm:
        return JSONResponse({"ok": False, "reason": "confirm_required"}, status_code=400)
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"ok": False, "reason": "runtime_disabled"}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"ok": False, "reason": "runtime_unavailable"}, status_code=503)

    # Destructive cascade: owner from the authenticated scope only —
    # trusting body.user_id let one user wipe another's entire companion
    # memory (audit 2026-06-17).
    uid = _resolve_user_id(request)
    from augmentum.companion_runtime import recovery
    try:
        affected = await recovery.delete_all(runtime, user_id=uid)
    except ValueError as exc:
        return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
    except Exception as exc:
        log.exception("companion_delete_all_failed", user_id=uid)
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc

    return JSONResponse({"ok": True, "rows_affected": affected})


class RerunAsRequest(BaseModel):
    user_id: str = Field(..., max_length=128)
    intent_text: str = Field(..., max_length=4000)
    original_winner: str = Field(..., max_length=64)
    chosen_target: str = Field(..., max_length=64)


@router.post("/api/companion/rerun_as")
async def companion_rerun_as(body: RerunAsRequest, request: Request) -> JSONResponse:
    """User explicitly requested a rerun against a different tool/channel
    (Lane 3 §9). Writes a DPO pair into companion_skill_archive so
    dispatch learns the preference.

    The actual rerun execution is the UI's responsibility — typically a
    follow-up /v1/chat/completions or /api/companion/intent call with
    the chosen_target as an explicit mode hint. This endpoint records
    the preference signal only.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"ok": False, "reason": "runtime_disabled"}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"ok": False, "reason": "runtime_unavailable"}, status_code=503)

    # Writes a DPO preference row — owner from the authenticated scope,
    # never body.user_id, so preferences can't be planted in another
    # user's archive (audit 2026-06-17).
    from augmentum.companion_runtime import recovery
    try:
        await recovery.record_rerun_pair(
            runtime,
            user_id=_resolve_user_id(request),
            intent_text=body.intent_text,
            original_winner=body.original_winner,
            chosen_target=body.chosen_target,
        )
    except Exception as exc:
        log.exception("companion_rerun_as_failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc

    return JSONResponse({"ok": True, "recorded": True})


class FrameBreakProbeRequest(BaseModel):
    text: str = Field(..., max_length=4000)


@router.post("/api/companion/narrative/frame_break_probe")
async def companion_frame_break_probe(
    body: FrameBreakProbeRequest, request: Request,
) -> JSONResponse:
    """Score a narrative-mode user turn for frame-break likelihood
    (Lane 3 §4.7). The narrative engine calls this before processing
    a user turn so it can decide whether to suspend the session and
    surface Becca.

    Returns ``{score, ooc_marker_present, distress_marker_present}``.
    Caller acts on:
      >= 0.85 → HARD break (pause session, mount Becca-OOC for one turn)
      0.55-0.84 → SOFT break (gutter affordance "step out?")
      < 0.55 → no action

    Refusal categories are reported separately so the narrative engine
    can short-circuit on harm-uplift / minor-explicit content even
    inside narrative frame.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"ok": False, "reason": "runtime_disabled"}, status_code=503)

    from augmentum.companion_runtime import narrative_isolation

    signal = narrative_isolation.detect_frame_break(body.text)
    refusal = narrative_isolation.frame_invariant_check(body.text)

    return JSONResponse({
        "ok": True,
        "score": signal.score,
        "ooc_marker_present": signal.ooc_marker_present,
        "distress_marker_present": signal.distress_marker_present,
        "refusal_category": refusal,
    })


class SafetyFloorAuditEventRequest(BaseModel):
    """Body for `POST /api/companion/safety_floor_audit_event`.

    The UI fires this when the user interacts with safety-floor-adjacent
    surfaces (currently just opening the realtalk panel). Anonymized —
    no user content, no user_id; written via the same HMAC fingerprint
    path as classifier audit rows.
    """
    kind: str = Field(..., min_length=1, max_length=64)
    locale: str = Field("", max_length=16)


@router.post("/api/companion/safety_floor_audit_event")
async def companion_safety_floor_audit_event(
    body: SafetyFloorAuditEventRequest, request: Request,
) -> JSONResponse:
    """Record an anonymized UI-event row in companion_safety_floor_audit.

    Lane 4 §6.5: the user never sees aggregated counts. This feeds only
    the maintainer regression-monitor pipeline. NO user content or
    user_id — the row is just `(fingerprint, fired=0, score=0, surface,
    threshold_used=0, locale, classifier_version, user_outcome=<kind>)`.

    Returns 503 when the runtime is off; 200 ``{ok: true}`` on success
    (silent — failures swallow into the runtime log per Lane 4 §6.5).
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"ok": False, "reason": "runtime_disabled"}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"ok": False, "reason": "runtime_not_initialized"}, status_code=503)

    from augmentum.companion_runtime import safety_floor

    # Synthesize a non-firing audit row tagged with the UI event kind via
    # user_outcome. Surface stays in the declared Literal set (free_chat
    # is where the realtalk panel attaches today).
    result = safety_floor.SafetyFloorResult(
        score=0.0,
        fired=False,
        surface="free_chat",
        threshold_used=0.0,
    )
    try:
        await safety_floor.audit_event(
            runtime, result,
            turn_id=body.kind,  # stable per-kind fingerprint salt
            locale=body.locale,
            user_outcome=body.kind,
        )
    except Exception:
        log.debug("safety_floor_ui_audit_failed", exc_info=True)
    return JSONResponse({"ok": True})


# ── Synapse Layer §4 — Consolidation review API ──────────────────────


def _consolidation_guard(request: Request) -> tuple[Any, JSONResponse | None]:
    """Shared 503 guard. Returns ``(runtime, None)`` on success, or
    ``(None, response)`` on failure so the caller can ``return response``."""
    if not getattr(settings, "companion_runtime_enabled", False):
        return None, JSONResponse(
            {"ok": False, "reason": "companion_runtime_enabled is off"},
            status_code=503,
        )
    if not getattr(settings, "companion_consolidation_enabled", False):
        return None, JSONResponse(
            {"ok": False, "reason": "companion_consolidation_enabled is off"},
            status_code=503,
        )
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return None, JSONResponse(
            {"ok": False, "reason": "companion_runtime not initialized"},
            status_code=503,
        )
    return runtime, None


@router.get("/api/companion/consolidation/candidates")
async def consolidation_list_candidates(request: Request) -> JSONResponse:
    """List pending personality-doc-edit candidates for review.

    Returns ``{"candidates": [...]}`` with full proposed text +
    current snapshot + drift distance + reasoning. The UI renders
    each as a side-by-side diff with approve/reject buttons.
    """
    runtime, fail = _consolidation_guard(request)
    if fail is not None:
        return fail
    from augmentum.companion_runtime import consolidation
    candidates = await consolidation.list_pending(runtime)
    return JSONResponse({"candidates": candidates})


class _ConsolidationRunBody(BaseModel):
    section_number: int = Field(..., ge=1, le=99,
                                description="Section to propose; rotating §10 or §11")
    days_back: int = Field(default=30, ge=1, le=365,
                            description="Evidence window in days")


@router.post("/api/companion/consolidation/run")
async def consolidation_run(
    body: _ConsolidationRunBody, request: Request,
) -> JSONResponse:
    """Manually trigger a proposal pass for one section.

    Useful during the rollout phase before the autonomous-cadence
    job is wired in. Returns the candidate record on success, an
    ``ok: false`` JSON with reason otherwise (FrozenSection, too
    little evidence, drift exceeded, LLM declined, …).
    """
    runtime, fail = _consolidation_guard(request)
    if fail is not None:
        return fail
    from augmentum.companion_runtime import consolidation
    try:
        candidate = await consolidation.propose_candidate(
            runtime,
            section_number=body.section_number,
            days_back=body.days_back,
        )
    except consolidation.FrozenSectionError as exc:
        return JSONResponse(
            {"ok": False, "reason": "frozen_section", "detail": str(exc)},
            status_code=400,
        )
    except consolidation.InsufficientEvidenceError as exc:
        return JSONResponse(
            {"ok": False, "reason": "insufficient_evidence", "detail": str(exc)},
            status_code=400,
        )
    except consolidation.DriftCeilingExceededError as exc:
        return JSONResponse(
            {"ok": False, "reason": "drift_exceeded", "detail": str(exc)},
            status_code=400,
        )
    except Exception as exc:
        log.exception("consolidation_run_crashed")
        return JSONResponse(
            {"ok": False, "reason": "crashed", "detail": str(exc)[:200]},
            status_code=500,
        )
    if candidate is None:
        return JSONResponse({"ok": True, "candidate": None,
                              "note": "no proposal — model declined or call failed"})
    return JSONResponse({"ok": True, "candidate": candidate.as_dict()})


@router.post("/api/companion/consolidation/{candidate_id}/approve")
async def consolidation_approve(
    candidate_id: int, request: Request,
) -> JSONResponse:
    """Approve a pending candidate.

    Writes the proposed text to a ``*.candidate.md`` sidecar next to
    the personality doc. The canonical doc is NOT auto-mutated — the
    reviewer diffs and commits manually, then restarts the runtime
    to re-digest. This is intentional: no autonomous file writes to
    in-repo source.
    """
    runtime, fail = _consolidation_guard(request)
    if fail is not None:
        return fail
    from augmentum.companion_runtime import consolidation
    result = await consolidation.approve_candidate(runtime, candidate_id)
    status_code = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status_code)


class _ConsolidationRejectBody(BaseModel):
    reason: str = Field(default="", max_length=1000,
                         description="Why the edit was rejected — journaled in her voice")


@router.post("/api/companion/consolidation/{candidate_id}/reject")
async def consolidation_reject(
    candidate_id: int,
    body: _ConsolidationRejectBody,
    request: Request,
) -> JSONResponse:
    """Reject a pending candidate.

    The reason is journaled as a 'correction' entry tagged 'unsure'
    so future evidence-gathering picks it up and the consolidator
    learns the shape of edits the user doesn't want.
    """
    runtime, fail = _consolidation_guard(request)
    if fail is not None:
        return fail
    from augmentum.companion_runtime import consolidation
    result = await consolidation.reject_candidate(
        runtime, candidate_id, reason=body.reason,
    )
    status_code = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status_code)


# ── Curator: tracked topics + feed subscriptions (unified) ──────────


class _TrackedTopicBody(BaseModel):
    """Add a tracked topic. A topic can be a bare noun phrase ("rust
    async runtime"), an explicit RSS URL, or a topic + feed pair. When
    the topic field looks like a URL and feed_url is empty, the URL is
    treated as the feed and a display topic is derived from the host."""
    topic: str = Field(..., min_length=1, max_length=300)
    feed_url: str | None = Field(None, max_length=1000)
    feed_kind: str | None = Field(None, max_length=32)


@router.get("/api/companion/topics")
async def companion_topics_list(request: Request) -> JSONResponse:
    """List the user's tracked topics + feed subscriptions (unified)."""
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False, "topics": []}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False, "topics": []}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True, "topics": []}, status_code=401)

    from augmentum.companion_runtime import curator
    try:
        topics = await curator.list_topics(
            runtime.backend.conn,
            user_id=user_id, companion_id=runtime.companion_id,
        )
    except Exception as exc:
        log.warning("topics_list_failed", error=str(exc)[:200])
        return JSONResponse({"enabled": True, "ready": True, "topics": []})
    return JSONResponse({
        "enabled": True, "ready": True,
        "topics": [t.as_dict() for t in topics],
    })


@router.post("/api/companion/topics")
async def companion_topics_add(
    body: _TrackedTopicBody, request: Request,
) -> JSONResponse:
    """Pin a topic (or add an RSS subscription). 409 on duplicate."""
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    # SSRF gate: any URL the user supplies here will later be fetched
    # by the curator. The topic field doubles as a URL when it looks
    # like one (see curator._looks_like_url) so we validate both.
    from augmentum.utils.safe_http import SafeHttpError, check_ssrf
    candidate_urls = []
    if body.feed_url:
        candidate_urls.append(body.feed_url)
    if body.topic and body.topic.strip().lower().startswith(("http://", "https://")):
        candidate_urls.append(body.topic.strip())
    for url in candidate_urls:
        try:
            await check_ssrf(url)
        except SafeHttpError as exc:
            return JSONResponse(
                {"ok": False, "reason": f"blocked URL: {exc}"},
                status_code=400,
            )

    from augmentum.companion_runtime import curator
    try:
        row = await curator.add_topic(
            runtime.backend.conn,
            user_id=user_id, companion_id=runtime.companion_id,
            topic=body.topic, feed_url=body.feed_url, feed_kind=body.feed_kind,
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
    except Exception as exc:
        log.warning("topics_add_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "internal"}, status_code=500)

    if row is None:
        return JSONResponse({"ok": False, "reason": "duplicate"}, status_code=409)
    return JSONResponse({"ok": True, "topic": row.as_dict()})


@router.delete("/api/companion/topics/{topic_id}")
async def companion_topics_remove(topic_id: int, request: Request) -> JSONResponse:
    """Unpin a topic / remove a feed subscription."""
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False}, status_code=503)
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse({"enabled": True, "ready": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    from augmentum.companion_runtime import curator
    try:
        removed = await curator.remove_topic(
            runtime.backend.conn, topic_id=topic_id,
            user_id=user_id, companion_id=runtime.companion_id,
        )
    except Exception as exc:
        log.warning("topics_remove_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "internal"}, status_code=500)
    if not removed:
        return JSONResponse({"ok": False, "reason": "not_found"}, status_code=404)
    return JSONResponse({"ok": True})


# ── Standing tasks (recurring jobs she runs for you) ────────────────


class _StandingTaskBody(BaseModel):
    """Add a recurring task. kind is one of the registered task kinds
    (see /api/companion/tasks for kinds[]). params shape is kind-specific."""
    title: str = Field(..., min_length=1, max_length=200)
    kind: str = Field(..., min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)
    interval_seconds: int = Field(86400, ge=300, le=30 * 86400)


class _StandingTaskPatchBody(BaseModel):
    """Edit a task. ``enabled`` toggles pause/resume. ``title`` / ``params``
    / ``interval_seconds`` edit the task in place (kind stays immutable).
    Any subset may be sent; omitted fields are unchanged."""
    enabled: bool | None = None
    title: str | None = Field(None, min_length=1, max_length=200)
    params: dict[str, Any] | None = None
    interval_seconds: int | None = Field(None, ge=300, le=30 * 86400)


def _tasks_ctx(request: Request):
    """Dispatch context for the standing-tasks CRUD surface.

    Scheduling is app-level: the companion runtime when it's up, else
    the SchedulerService's headless context. None only when neither
    dispatcher exists (scheduling_enabled AND companion both off).
    """
    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is not None:
        return runtime
    service = getattr(request.app.state, "scheduler_service", None)
    if service is not None:
        return service.ctx
    return None


@router.get("/api/companion/tasks")
async def companion_tasks_list(request: Request) -> JSONResponse:
    runtime = _tasks_ctx(request)
    if runtime is None:
        return JSONResponse({"enabled": False, "tasks": []}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True, "tasks": []}, status_code=401)

    from augmentum.companion_runtime import standing_tasks
    try:
        tasks = await standing_tasks.list_tasks(
            runtime.backend.conn,
            user_id=user_id, companion_id=runtime.companion_id,
        )
    except Exception as exc:
        log.warning("tasks_list_failed", error=str(exc)[:200])
        return JSONResponse({"enabled": True, "ready": True, "tasks": []})
    return JSONResponse({
        "enabled": True, "ready": True,
        "tasks": [t.as_dict() for t in tasks],
        "kinds": standing_tasks.known_kinds(),
    })


class _CronPreviewBody(BaseModel):
    cron: str = Field(..., min_length=1, max_length=128)


@router.post("/api/companion/tasks/cron-preview")
async def companion_tasks_cron_preview(
    body: _CronPreviewBody, request: Request,
) -> JSONResponse:
    """Live schedule-builder feedback: validate a cron expression and
    return a human gloss + the next 3 fire times in the user's zone.

    Pure computation over the user's timezone setting — deliberately NOT
    gated on the companion runtime, so the schedule surface can offer
    live preview regardless of which dispatcher is running.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"ok": False}, status_code=401)
    from datetime import datetime

    from augmentum.companion_runtime.standing_tasks import (
        _resolve_user_timezone,
        _resolve_zoneinfo,
    )
    from augmentum.utils import cron as cron_util

    err = cron_util.validate(body.cron)
    if err:
        return JSONResponse({"ok": False, "error": err})
    tzname = await _resolve_user_timezone(request.app.state, user_id)
    tz = _resolve_zoneinfo(tzname)
    cursor = datetime.now(tz) if tz else datetime.now().astimezone()
    fires: list[str] = []
    for _ in range(3):
        nxt = cron_util.next_after(body.cron, cursor)
        if nxt is None:
            break
        fires.append(nxt.strftime("%Y-%m-%d %H:%M"))
        cursor = nxt
    return JSONResponse({
        "ok": True,
        "description": cron_util.describe(body.cron),
        "timezone": tzname or "server-local",
        "next_fires": fires,
    })


class _FeedResolveBody(BaseModel):
    source: str = Field(..., min_length=1, max_length=512)


@router.post("/api/companion/tasks/resolve-feed")
async def companion_tasks_resolve_feed(
    body: _FeedResolveBody, request: Request,
) -> JSONResponse:
    """Turn 'whatever the user pasted' into a validated feed.

    YouTube channel/@handle/video → the channel's keyless Atom feed;
    r/name → subreddit RSS; blog/site URL → autodiscovered feed; a feed
    URL → itself. Returns the feed title + latest entry as proof, so the
    Schedule UI can confirm before creating the feed_watch. Auth-gated
    only — resolution is a read, not a task write.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"ok": False}, status_code=401)
    http_client = getattr(request.app.state, "http_client", None)
    if http_client is None:
        return JSONResponse(
            {"ok": False, "error": "no http client available"},
            status_code=503,
        )
    from augmentum.companion_runtime.feed_resolve import resolve_feed_source
    resolved = await resolve_feed_source(http_client, body.source)
    return JSONResponse(resolved.as_dict())


@router.post("/api/companion/tasks")
async def companion_tasks_add(
    body: _StandingTaskBody, request: Request,
) -> JSONResponse:
    runtime = _tasks_ctx(request)
    if runtime is None:
        return JSONResponse({"enabled": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    from augmentum.companion_runtime import standing_tasks
    user_tz = await standing_tasks._resolve_user_timezone(
        request.app.state, user_id,
    )
    try:
        task = await standing_tasks.add_task(
            runtime.backend.conn,
            user_id=user_id, companion_id=runtime.companion_id,
            title=body.title, kind=body.kind, params=body.params,
            interval_seconds=body.interval_seconds,
            user_timezone=user_tz,
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
    except Exception as exc:
        log.warning("tasks_add_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "internal"}, status_code=500)
    if task is None:
        return JSONResponse({"ok": False, "reason": "duplicate"}, status_code=409)
    return JSONResponse({"ok": True, "task": task.as_dict()})


@router.patch("/api/companion/tasks/{task_id}")
async def companion_tasks_patch(
    task_id: int, body: _StandingTaskPatchBody, request: Request,
) -> JSONResponse:
    """Edit a task. ``enabled`` pauses/resumes; ``title`` / ``params`` /
    ``interval_seconds`` edit in place (kind is immutable — delete and
    re-add to change kind). A schedule edit recomputes the next fire."""
    runtime = _tasks_ctx(request)
    if runtime is None:
        return JSONResponse({"enabled": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    from augmentum.companion_runtime import standing_tasks

    edits_present = (
        body.title is not None
        or body.params is not None
        or body.interval_seconds is not None
    )
    if body.enabled is None and not edits_present:
        return JSONResponse({"ok": False, "reason": "no_change"}, status_code=400)

    try:
        # Field edits first (title/params/interval), then the enabled
        # toggle — so "resume + reschedule" in one PATCH lands cleanly.
        if edits_present:
            user_tz = await standing_tasks._resolve_user_timezone(
                request.app.state, user_id,
            )
            updated = await standing_tasks.update_task(
                runtime.backend.conn, task_id=task_id,
                user_id=user_id, companion_id=runtime.companion_id,
                title=body.title, params=body.params,
                interval_seconds=body.interval_seconds,
                user_timezone=user_tz,
            )
            if updated is None:
                return JSONResponse(
                    {"ok": False, "reason": "not_found"}, status_code=404,
                )
        if body.enabled is not None:
            ok = await standing_tasks.set_enabled(
                runtime.backend.conn, task_id=task_id,
                user_id=user_id, companion_id=runtime.companion_id,
                enabled=bool(body.enabled),
            )
            if not ok:
                return JSONResponse(
                    {"ok": False, "reason": "not_found"}, status_code=404,
                )
    except ValueError as exc:
        return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
    except Exception as exc:
        log.warning("tasks_patch_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "internal"}, status_code=500)
    return JSONResponse({"ok": True})


@router.delete("/api/companion/tasks/{task_id}")
async def companion_tasks_remove(task_id: int, request: Request) -> JSONResponse:
    runtime = _tasks_ctx(request)
    if runtime is None:
        return JSONResponse({"enabled": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    from augmentum.companion_runtime import standing_tasks
    try:
        removed = await standing_tasks.remove_task(
            runtime.backend.conn, task_id=task_id,
            user_id=user_id, companion_id=runtime.companion_id,
        )
    except Exception as exc:
        log.warning("tasks_remove_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "internal"}, status_code=500)
    if not removed:
        return JSONResponse({"ok": False, "reason": "not_found"}, status_code=404)
    return JSONResponse({"ok": True})


@router.get("/api/companion/tasks/{task_id}/runs")
async def companion_tasks_runs(task_id: int, request: Request) -> JSONResponse:
    """Run history for one task, newest first — the trust surface.
    Includes silent runs ("checked, nothing new") so a quiet watch is
    distinguishable from a dead one."""
    runtime = _tasks_ctx(request)
    if runtime is None:
        return JSONResponse({"enabled": False, "runs": []}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True, "runs": []}, status_code=401)

    from augmentum.companion_runtime import standing_tasks
    try:
        runs = await standing_tasks.list_runs(
            runtime.backend.conn, task_id=task_id, user_id=user_id,
        )
    except Exception as exc:
        log.warning("tasks_runs_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "internal"}, status_code=500)
    return JSONResponse({"ok": True, "runs": runs})


@router.post("/api/companion/tasks/{task_id}/run-now")
async def companion_tasks_run_now(task_id: int, request: Request) -> JSONResponse:
    """Manual fire. Bypasses next_run_at; runs the task immediately and
    returns the result. Useful for testing or for "answer now" needs."""
    runtime = _tasks_ctx(request)
    if runtime is None:
        return JSONResponse({"enabled": False}, status_code=503)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": True}, status_code=401)

    from augmentum.companion_runtime import standing_tasks
    try:
        result = await standing_tasks.run_now(
            runtime, task_id=task_id, user_id=user_id,
        )
    except Exception as exc:
        log.warning("tasks_run_now_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "internal"}, status_code=500)
    if result is None:
        return JSONResponse({"ok": False, "reason": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "result": result})


# ── Phase 5: verb-log observability ───────────────────────────────────


@router.get("/api/companion/day")
async def companion_day(request: Request) -> JSONResponse:
    """Becca's day — the timeline of management-verb activity for the
    current user, sourced from ``companion_verb_log`` (Phase 2 substrate).

    Companion verbs architecture, Phase 5. Renders the management-verb
    side of the runtime as a single observable surface: what fired,
    what was skipped (and why), what touched what substrate.

    Query parameters:
      * ``window_hours`` — how far back to look. Default 24.
      * ``limit`` — max rows returned. Default 200.

    Response shape::

        {
            "enabled": bool,
            "window_hours": int,
            "now": int,
            "summary": {
                "<verb_name>": {
                    "fires": int, "ok": int, "skipped": int, "errors": int,
                    "avg_latency_ms": float, "last_fired_at": int,
                    "dispatch_class": str, "safety_class": str
                }
            },
            "timeline": [
                {
                    "verb": str, "outcome": str, "fired_at": int,
                    "latency_ms": int, "event_topic": str,
                    "cited": [{"table": str, "row_id": "...", ...}, ...],
                    "error": str
                }
            ]
        }

    Always 200 (unless auth fails). Empty arrays/objects when no rows.
    """
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"enabled": False}, status_code=200)
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"enabled": True, "ready": False}, status_code=401)

    runtime = getattr(request.app.state, "companion_runtime", None)
    if runtime is None:
        return JSONResponse(
            {"enabled": True, "ready": False}, status_code=503,
        )

    try:
        window_hours = max(1, min(168, int(request.query_params.get("window_hours") or 24)))
    except ValueError:
        window_hours = 24
    try:
        limit = max(1, min(2000, int(request.query_params.get("limit") or 200)))
    except ValueError:
        limit = 200

    import json
    import time as _time
    now = int(_time.time())
    since = now - window_hours * 3600
    backend = runtime.backend

    summary: dict[str, dict] = {}
    timeline: list[dict] = []

    try:
        # Timeline — recent first.
        cur = await backend.conn.execute(
            """
            SELECT verb_name, outcome, fired_at, latency_ms, event_topic,
                   cited_substrate, error, dispatch_class
            FROM companion_verb_log
            WHERE user_id = ? AND fired_at >= ?
            ORDER BY fired_at DESC
            LIMIT ?
            """,
            (user_id, since, limit),
        )
        timeline_rows = await cur.fetchall()
        await cur.close()
    except Exception:
        log.warning("companion_day_timeline_query_failed", exc_info=True)
        timeline_rows = []

    for verb, outcome, fired_at, latency_ms, topic, cited_json, error, _dc in timeline_rows:
        try:
            cited = json.loads(cited_json) if cited_json else []
        except Exception:
            cited = []
        timeline.append({
            "verb": verb,
            "outcome": outcome,
            "fired_at": int(fired_at),
            "latency_ms": int(latency_ms or 0),
            "event_topic": topic or "",
            "cited": cited,
            "error": (error or "")[:200] if outcome == "error" else "",
        })

    try:
        # Per-verb summary across the window.
        cur = await backend.conn.execute(
            """
            SELECT verb_name, dispatch_class, outcome,
                   COUNT(*) AS n, AVG(latency_ms) AS avg_lat,
                   MAX(fired_at) AS last_fired
            FROM companion_verb_log
            WHERE user_id = ? AND fired_at >= ?
            GROUP BY verb_name, dispatch_class, outcome
            """,
            (user_id, since),
        )
        agg = await cur.fetchall()
        await cur.close()
    except Exception:
        log.warning("companion_day_summary_query_failed", exc_info=True)
        agg = []

    for verb, dispatch_class, outcome, n, avg_lat, last_fired in agg:
        entry = summary.setdefault(verb, {
            "fires": 0, "ok": 0, "skipped": 0, "errors": 0,
            "avg_latency_ms": 0.0, "last_fired_at": 0,
            "dispatch_class": dispatch_class or "",
            "safety_class": "",  # Filled below from the dispatcher snapshot.
        })
        entry["fires"] += int(n or 0)
        if outcome == "ok":
            entry["ok"] += int(n or 0)
            entry["avg_latency_ms"] = float(avg_lat or 0.0)
        elif outcome and outcome.endswith("_skipped") or outcome in ("cooldown_skipped", "deduped"):
            entry["skipped"] += int(n or 0)
        elif outcome in ("error", "auto_paused"):
            entry["errors"] += int(n or 0)
        last_fired_int = int(last_fired or 0)
        if last_fired_int > entry["last_fired_at"]:
            entry["last_fired_at"] = last_fired_int

    # Augment summary with declared safety_class from the dispatcher.
    dispatcher = getattr(runtime, "_verb_dispatcher", None)
    if dispatcher is not None:
        try:
            for name in dispatcher.names():
                if name not in summary:
                    continue
                v = dispatcher.get(name)
                if v is None:
                    continue
                summary[name]["safety_class"] = (
                    v.safety_class.value if hasattr(v.safety_class, "value")
                    else str(v.safety_class)
                )
        except Exception:
            log.debug("companion_day_safety_class_join_failed", exc_info=True)

    return JSONResponse({
        "enabled": True,
        "window_hours": window_hours,
        "now": now,
        "summary": summary,
        "timeline": timeline,
    })
