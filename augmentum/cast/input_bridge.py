"""In-RAM registry routing gamepad input between phone WS clients and
the in-container ``cast-input-bridge.py`` daemon.

The substrate mirrors :mod:`augmentum.cast.receiver_registry`: one
process-wide registry holds every open connection, user-scoped on every
read, no persistence. Lifecycle is bound to the game-stream session id
that the streaming runtime mints — when that session ends, both sides
detach.

Two attachment shapes live here:

* **Container side** — ``cast-input-bridge.py`` inside the agsp
  container dials augmentum and attaches as a ``ConnectedContainer``.
  Exactly one per session. Owns up to 4 UInput virtual pads.
* **Phone side** — the cast-control UI on a phone holds the user's
  Bluetooth/USB gamepad and attaches as a ``ConnectedPhone``. Multiple
  phones (or multiple pads on one phone) can attach to the same
  session; the registry's slot-claim logic decides which UInput each
  one drives.

Routing direction:
  phone frame → registry.route_input → container WS (with assigned
    ``slot`` field added so the container knows which UInput to write).
  container frame → registry.route_rumble → phone WS owning the slot
    (so ``vibrationActuator.playEffect`` can fire on the right pad).

Slot assignment honours the per-system ``pad_routing`` strategy from
:class:`augmentum.controllers.service.ControllerService`:

* ``index`` — slot follows the phone-reported ``pad_index`` (pad 0 →
  slot 0, pad 1 → slot 1). Deterministic; ideal when one phone holds
  every controller.
* ``firstpress`` — slot is claimed on the first non-zero button frame
  the phone sends. Ergonomic for couch co-op where multiple phones
  pair to one session and the order of arrival shouldn't matter.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import WebSocket

log = get_logger(__name__)


MAX_PADS_PER_SESSION = 4

# Routing strategies. Match the literal strings stored on
# controller_remaps.pad_routing — see ControllerService.resolve().
ROUTING_INDEX = "index"
ROUTING_FIRSTPRESS = "firstpress"

# Sentinel slot value for phones that haven't claimed yet under
# firstpress. Container drops frames carrying this slot so an idle
# phone can't pollute a P1's input stream.
SLOT_UNCLAIMED = -1

# Resilience #3: how long a slot is held warm after a phone WS
# detaches, so the same guest reconnecting (auto-reconnect on WS
# close, screen-wake from background, network handoff) reclaims
# THEIR slot rather than landing in a fresh one or seeing "all slots
# taken." Set per-guest-profile-id; anon detaches free the slot
# immediately since we can't disambiguate.
WARM_SLOT_TTL_S = 30.0


@dataclass(slots=True)
class ConnectedContainer:
    """A game-stream container's bridge WS."""

    session_id: str
    user_id: str
    ws: WebSocket
    pad_routing: str = ROUTING_INDEX
    system_id: str = ""
    connected_at: float = 0.0


@dataclass(slots=True)
class ConnectedPhone:
    """A phone-side controller WS for a specific game-stream session."""

    attachment_id: str
    session_id: str
    user_id: str
    ws: WebSocket
    pad_index: int = 0
    # -1 until the phone claims a slot (firstpress strategy). For
    # ``index`` routing, claimed at attach time from pad_index.
    slot: int = SLOT_UNCLAIMED
    connected_at: float = 0.0
    last_frame_at: float = 0.0
    # Phase 2: optional guest profile identity. Empty for the host's
    # own phone (the host's input legally counts as themselves) and
    # for anonymous Phase-1 joins. Populated when a guest came in
    # via the named-claim flow.
    guest_profile_id: str = ""
    guest_display_name: str = ""
    guest_color: str = ""
    # Resilience #4: rolling RTT stats. Phones stamp each frame with
    # t_send (perf-now); the proxy stamps t_recv (server unix ms) on
    # arrival and echoes back. The phone diffs the round-trip and
    # reports rtt_ms in subsequent frames so the server has a phone-
    # local view of how lossy the link is. Used by per-phone latency
    # logs + (future) admission feedback to the producer.
    rtt_ms_sum: float = 0.0
    rtt_ms_count: int = 0
    rtt_ms_last: float = 0.0
    # Server-stamp of last forwarded frame's t_send → diff against
    # arrival gives one-way delay sampling.
    last_owd_ms: float = 0.0
    # Frames dropped because they arrived out-of-order (stale t_send).
    stale_frames: int = 0
    # Browser-cast routing target. Empty string means "container target"
    # (the legacy path — route via session_id to a UInput pad in an
    # AGSP container). Non-empty means "route to this receiver's WS as
    # a CMD_INPUT_GAMEPAD command instead." Set at attach time from the
    # phone WS query params and never mutated; switching targets means
    # detaching + reattaching with a different param.
    target_receiver_id: str = ""


class CastInputRegistry:
    """Routes input frames + rumble events between phones and containers.

    Thread/task safety: like ReceiverRegistry, single-event-loop access
    only. No locks. Detach is idempotent.
    """

    def __init__(self) -> None:
        # session_id → container
        self._containers: dict[str, ConnectedContainer] = {}
        # attachment_id → phone
        self._phones: dict[str, ConnectedPhone] = {}
        # session_id → {slot: attachment_id} so we can look up the
        # owning phone for a rumble event without walking all phones.
        self._slot_owners: dict[str, dict[int, str]] = {}
        # Resilience: post-detach warm-slot reservations. A guest's
        # named profile holds onto its slot for WARM_SLOT_TTL_S after
        # WS close so a reconnect inside that window reclaims THEIR
        # slot rather than entering the firstpress pool from scratch.
        # Keyed (session_id, guest_profile_id) → {slot, expires_at}.
        self._warm_slots: dict[tuple[str, str], dict[str, Any]] = {}

    # ── Container attach / detach ─────────────────────────────────────

    def attach_container(
        self,
        *,
        ws: WebSocket,
        session_id: str,
        user_id: str,
        pad_routing: str = ROUTING_INDEX,
        system_id: str = "",
    ) -> ConnectedContainer:
        """Register a freshly-connected container WS.

        Replaces any prior container on the same session_id (a
        container restart inside a single game-stream lifecycle will
        re-dial; the old WS is closed). The caller has already
        accepted the WS handshake.
        """
        existing = self._containers.get(session_id)
        if existing is not None:
            log.info(
                "cast_input_container_replaced",
                session_id=session_id, user_id=user_id,
            )
            # Drop the old record; the old WS will close itself when
            # it next tries to read/write. We do NOT actively close it
            # here because the same event loop is running both the
            # read loop of the old WS and this attach call — closing
            # synchronously could deadlock. The route handler's
            # detach() in its finally block is the canonical cleanup.
            self._containers.pop(session_id, None)

        normalised_routing = (
            pad_routing if pad_routing in (ROUTING_INDEX, ROUTING_FIRSTPRESS)
            else ROUTING_INDEX
        )
        container = ConnectedContainer(
            session_id=session_id,
            user_id=user_id,
            ws=ws,
            pad_routing=normalised_routing,
            system_id=system_id,
            connected_at=time.time(),
        )
        self._containers[session_id] = container
        self._slot_owners.setdefault(session_id, {})
        log.info(
            "cast_input_container_attached",
            session_id=session_id, user_id=user_id,
            pad_routing=normalised_routing, system_id=system_id,
        )
        return container

    def detach_container(self, session_id: str) -> bool:
        container = self._containers.pop(session_id, None)
        if container is None:
            return False
        slots = self._slot_owners.pop(session_id, {})
        # Session is over — clear any warm-hold reservations for it
        # so dead sessions don't accrete in _warm_slots.
        to_drop = [
            k for k in self._warm_slots
            if k[0] == session_id
        ]
        for k in to_drop:
            self._warm_slots.pop(k, None)
        log.info(
            "cast_input_container_detached",
            session_id=session_id, user_id=container.user_id,
            slots_held=len(slots),
            warm_slots_cleared=len(to_drop),
            duration_s=round(time.time() - container.connected_at, 2),
        )
        return True

    def get_container(self, session_id: str) -> ConnectedContainer | None:
        return self._containers.get(session_id)

    # ── Phone attach / detach ─────────────────────────────────────────

    def attach_phone(
        self,
        *,
        ws: WebSocket,
        session_id: str,
        user_id: str,
        pad_index: int = 0,
        guest_profile_id: str = "",
        guest_display_name: str = "",
        guest_color: str = "",
        target_receiver_id: str = "",
    ) -> ConnectedPhone:
        """Register a phone WS for a session.

        Slot stays ``SLOT_UNCLAIMED`` until :meth:`route_input` sees a
        frame AND a container is attached. Deferred claim is the only
        safe path: when a phone arrives before the container's bridge
        has dialled in (typical — phone dials right after POST /start
        while the container is still booting), the registry doesn't
        yet know whether to apply ``index`` or ``firstpress`` routing.
        """
        attachment_id = f"cip_{secrets.token_hex(10)}"
        clamped_pad = max(0, min(int(pad_index), MAX_PADS_PER_SESSION - 1))

        # Resilience: check warm-slot reservation. If THIS guest
        # profile detached recently from THIS session, reclaim their
        # slot immediately rather than entering firstpress fresh.
        reclaimed_slot = SLOT_UNCLAIMED
        if guest_profile_id:
            key = (session_id, guest_profile_id)
            warm = self._warm_slots.get(key)
            if warm is not None and warm["expires_at"] > time.time():
                reclaimed_slot = int(warm["slot"])
                # Claim the slot under this new attachment_id.
                slots = self._slot_owners.setdefault(session_id, {})
                # The warm slot might be occupied by another guest
                # who took it during the grace window — only reclaim
                # if free.
                if reclaimed_slot not in slots:
                    slots[reclaimed_slot] = attachment_id
                else:
                    reclaimed_slot = SLOT_UNCLAIMED
            # Whether we reclaimed or not, the warm record is consumed.
            self._warm_slots.pop(key, None)

        phone = ConnectedPhone(
            attachment_id=attachment_id,
            session_id=session_id,
            user_id=user_id,
            ws=ws,
            pad_index=clamped_pad,
            slot=reclaimed_slot,
            connected_at=time.time(),
            guest_profile_id=guest_profile_id,
            guest_display_name=guest_display_name,
            guest_color=guest_color,
            target_receiver_id=target_receiver_id,
        )
        self._phones[attachment_id] = phone
        log.info(
            "cast_input_phone_attached",
            attachment_id=attachment_id, session_id=session_id,
            user_id=user_id, pad_index=clamped_pad,
            guest_profile_id=guest_profile_id or "anon",
            reclaimed_slot=reclaimed_slot if reclaimed_slot >= 0 else "none",
        )
        return phone

    async def detach_phone(self, attachment_id: str) -> bool:
        phone = self._phones.pop(attachment_id, None)
        if phone is None:
            return False
        slots = self._slot_owners.get(phone.session_id)
        slot = phone.slot
        owned = (
            slots is not None
            and slot != SLOT_UNCLAIMED
            and slots.get(slot) == attachment_id
        )
        if owned:
            slots.pop(slot, None)
            # Resilience: warm-hold the slot for this guest profile
            # if there is one. Anon detaches (no profile id) skip the
            # warm hold because we can't disambiguate which phone is
            # reconnecting later.
            if phone.guest_profile_id:
                self._warm_slots[(phone.session_id, phone.guest_profile_id)] = {
                    "slot": slot,
                    "expires_at": time.time() + WARM_SLOT_TTL_S,
                }
            # Send a neutral-state frame to the container so any buttons
            # that were pressed when the phone vanished get released in
            # the emulator. Without this, a controller yanked mid-press
            # leaves the slot's last-applied state stuck (the daemon's
            # delta diff would never see a transition back to zero).
            container = self._containers.get(phone.session_id)
            if container is not None:
                neutral = {
                    "seq": -1,
                    "t_send": 0.0,
                    "slot": slot,
                    "event": {
                        "kind": "gamepad_state",
                        "pad_index": phone.pad_index,
                        "buttons": [0] * 17,
                        "axes": [0.0] * 4,
                    },
                }
                try:
                    await container.ws.send_json(neutral)
                except Exception as exc:
                    log.warning(
                        "cast_input_release_send_failed",
                        attachment_id=attachment_id,
                        session_id=phone.session_id,
                        slot=slot, error=str(exc)[:160],
                    )
        avg_rtt_ms = (
            round(phone.rtt_ms_sum / phone.rtt_ms_count, 1)
            if phone.rtt_ms_count > 0 else 0
        )
        log.info(
            "cast_input_phone_detached",
            attachment_id=attachment_id, session_id=phone.session_id,
            user_id=phone.user_id, slot=slot,
            duration_s=round(time.time() - phone.connected_at, 2),
            avg_rtt_ms=avg_rtt_ms,
            last_rtt_ms=round(phone.rtt_ms_last, 1),
            rtt_samples=phone.rtt_ms_count,
            stale_frames=phone.stale_frames,
        )
        return True

    def get_phone(self, attachment_id: str) -> ConnectedPhone | None:
        return self._phones.get(attachment_id)

    def list_phones_for_session(self, session_id: str) -> list[ConnectedPhone]:
        return [p for p in self._phones.values() if p.session_id == session_id]

    def roster_for_session(self, session_id: str) -> list[dict[str, Any]]:
        """Snapshot of currently-attached phones for a session.

        Returns a list of ``{slot, name, color, status}`` dicts —
        one per claimed slot — sorted by slot number. The host (P1)
        isn't in the registry directly (their input goes through the
        same WS but isn't tracked here), so the receiver's UI strip
        renders host separately and combines with this list.

        Used by the cast routes to push ``invite_slot_update`` events
        to the receiver as phones come and go.
        """
        slots = self._slot_owners.get(session_id) or {}
        out: list[dict[str, Any]] = []
        for slot in sorted(slots.keys()):
            phone = self._phones.get(slots[slot])
            if phone is None:
                continue
            out.append({
                "slot": slot,
                "name": phone.guest_display_name or "Guest",
                "color": phone.guest_color,
                "status": "active",
            })
        return out

    # ── Routing ───────────────────────────────────────────────────────

    async def route_input(
        self,
        *,
        attachment_id: str,
        frame: dict[str, Any],
        receiver_registry: Any = None,
    ) -> bool:
        """Forward a phone-side input frame to its routing target.

        Routing target depends on the phone's attach-time configuration:

        * **Container** (legacy / default) — frame is sent to the AGSP
          container's WS for the same ``session_id``. Returns False if
          the container hasn't dialled in yet (frames in that window
          weren't going anywhere useful anyway).
        * **Receiver browser surface** (browser-cast) — frame is fanned
          out as a ``CMD_INPUT_GAMEPAD`` command on the receiver's WS.
          Requires ``receiver_registry`` to be passed; callers route
          via the cast routes which own the registry on app.state.

        Under firstpress routing, a frame with a non-zero button claims
        the next free slot for the phone. Frames carrying only neutral
        state are forwarded with slot=-1 which the container drops.
        Browser-cast targets always use ``index`` routing — there's no
        firstpress concept when the target is a single browser iframe.
        """
        phone = self._phones.get(attachment_id)
        if phone is None:
            return False
        # Browser-cast branch — short-circuits the container-bound path.
        # No slot-claim logic (single iframe consumer per receiver), no
        # rumble fan-out (browser games can request vibration via the
        # standard Gamepad API but it returns to whichever phone holds
        # the slot via the existing rumble path in route_rumble — not
        # implemented for receiver targets yet; tracked as a follow-up).
        if phone.target_receiver_id:
            return await self._route_to_receiver(
                phone=phone,
                frame=frame,
                receiver_registry=receiver_registry,
            )
        container = self._containers.get(phone.session_id)
        if container is None:
            return False

        # Lazy slot claim — once the container is attached we know the
        # routing strategy. Under ``index`` we claim immediately on the
        # first frame (matches the phone's pad_index); under
        # ``firstpress`` we only claim when a non-zero button arrives.
        if phone.slot == SLOT_UNCLAIMED:
            event = frame.get("event") if isinstance(frame, dict) else None
            if container.pad_routing == ROUTING_INDEX:
                claimed = self._claim_slot_index(
                    phone.session_id, phone.pad_index, attachment_id,
                )
                if claimed != SLOT_UNCLAIMED:
                    phone.slot = claimed
                    log.info(
                        "cast_input_phone_claimed_slot",
                        attachment_id=attachment_id,
                        session_id=phone.session_id,
                        slot=claimed, strategy="index",
                    )
            elif (
                isinstance(event, dict)
                and container.pad_routing == ROUTING_FIRSTPRESS
                and _has_active_button(event)
            ):
                claimed = self._claim_slot_firstpress(
                    phone.session_id, attachment_id,
                )
                if claimed != SLOT_UNCLAIMED:
                    phone.slot = claimed
                    log.info(
                        "cast_input_phone_claimed_slot",
                        attachment_id=attachment_id,
                        session_id=phone.session_id,
                        slot=claimed, strategy="firstpress",
                    )

        # Latency #4: drop frames that arrive AFTER a fresher one
        # already landed. With phone clock-drift this is conservative
        # (clamp by abs delta to avoid false positives) but the common
        # case — a TCP retransmit landing late behind a newer frame —
        # is correctly suppressed.
        now = time.time()
        frame_t_send = frame.get("t_send") if isinstance(frame, dict) else None
        if (
            isinstance(frame_t_send, int | float)
            and phone.last_frame_at > 0
            and frame_t_send < phone.last_owd_ms - 200  # 200ms tolerance
        ):
            phone.stale_frames += 1
            # Don't forward stale frames — they'd overwrite fresher state.
            return False

        phone.last_frame_at = now
        if isinstance(frame_t_send, int | float):
            phone.last_owd_ms = float(frame_t_send)

        # Latency #4: client-reported RTT measurement. The phone has
        # been diffing its own t_send with our echoed t_recv and
        # putting the result in frame.rtt_ms. We aggregate so per-
        # phone "is this controller laggy?" is diagnosable from logs.
        client_rtt = frame.get("rtt_ms") if isinstance(frame, dict) else None
        if isinstance(client_rtt, int | float) and client_rtt >= 0:
            phone.rtt_ms_last = float(client_rtt)
            phone.rtt_ms_sum += float(client_rtt)
            phone.rtt_ms_count += 1

        # Server-side stamp of the slot so the container daemon doesn't
        # have to trust phone-reported values. -1 means "neutral; drop".
        outbound = dict(frame)
        outbound["slot"] = phone.slot

        # Echo back a t_recv stamp so the producer can compute RTT.
        # Cheap — adds 1 field to the inbound JSON, no extra send.
        # We piggyback the next outbound rumble event for the echo
        # (see route_rumble) so we don't add a third stream, OR we
        # send a tiny ack here when the phone marked the frame as
        # echo-eligible. Default: skip if not requested.
        if frame.get("echo"):
            try:
                await phone.ws.send_json({
                    "kind": "echo",
                    "seq": frame.get("seq"),
                    "t_send": frame.get("t_send"),
                    "t_recv": now * 1000.0,  # ms wall clock
                })
            except Exception:
                pass  # Best-effort; no need to fail the input path.

        try:
            await container.ws.send_json(outbound)
            return True
        except Exception as exc:
            log.warning(
                "cast_input_forward_failed",
                attachment_id=attachment_id,
                session_id=phone.session_id, error=str(exc)[:160],
            )
            # The container WS is broken — drop it; the route's finally
            # block on the container side will fire when the read loop
            # eventually notices.
            self._containers.pop(phone.session_id, None)
            return False

    async def _route_to_receiver(
        self,
        *,
        phone: ConnectedPhone,
        frame: dict[str, Any],
        receiver_registry: Any,
    ) -> bool:
        """Forward a gamepad frame to a kiosk play surface on a receiver.

        Implementation note: we don't claim a slot here — browser-cast
        is one phone per pad slot per receiver (no UInput multiplexing),
        so ``slot`` is just the phone's reported ``pad_index``. The
        receiver's universal-input-adapter loader (gamepad_api shim)
        keys its virtual gamepad array by pad_index directly. Slot
        bookkeeping stays empty for these phones (``_slot_owners`` is
        unused for browser-cast).
        """
        if receiver_registry is None:
            log.warning(
                "cast_input_receiver_registry_missing",
                attachment_id=phone.attachment_id,
            )
            return False
        event = frame.get("event") if isinstance(frame, dict) else None
        if not isinstance(event, dict) or event.get("kind") != "gamepad_state":
            # Echo-mode pings and non-state frames don't fan out to the
            # iframe — only state deltas are useful there.
            return False
        # Reuse the existing receiver_protocol command shape so the
        # receiver shell can dispatch off the same `type=cmd / cmd=...`
        # switch the rest of the WS protocol uses. Args carry the
        # minimal gamepad state the shim needs to populate the virtual
        # navigator.getGamepads() array.
        from augmentum.cast.receiver_protocol import (
            CMD_INPUT_GAMEPAD,
            ReceiverCmd,
        )
        cmd = ReceiverCmd(
            cmd=CMD_INPUT_GAMEPAD,
            args={
                "slot": phone.pad_index,
                "pad_index": phone.pad_index,
                "buttons": event.get("buttons") or [],
                "axes": event.get("axes") or [],
            },
        )
        try:
            ok = await receiver_registry.send(phone.target_receiver_id, cmd)
        except Exception as exc:
            log.warning(
                "cast_input_receiver_forward_failed",
                attachment_id=phone.attachment_id,
                receiver_id=phone.target_receiver_id,
                error=str(exc)[:160],
            )
            return False
        if not ok:
            # Receiver is gone — drop the phone so it can re-attach
            # cleanly against a fresh receiver if the user re-casts.
            log.info(
                "cast_input_receiver_offline",
                attachment_id=phone.attachment_id,
                receiver_id=phone.target_receiver_id,
            )
            self._phones.pop(phone.attachment_id, None)
            return False
        return True

    async def route_rumble(
        self,
        *,
        session_id: str,
        frame: dict[str, Any],
    ) -> int:
        """Forward a container-side rumble event to the owning phone.

        The frame shape ``{"kind": "rumble", "slot": N, ...}`` is sent
        as-is to each phone holding that slot. Returns the count of
        successful sends.
        """
        slot_raw = frame.get("slot") if isinstance(frame, dict) else None
        try:
            slot = int(slot_raw) if slot_raw is not None else SLOT_UNCLAIMED
        except (TypeError, ValueError):
            slot = SLOT_UNCLAIMED
        if slot == SLOT_UNCLAIMED:
            return 0
        slots = self._slot_owners.get(session_id) or {}
        attachment_id = slots.get(slot, "")
        if not attachment_id:
            return 0
        phone = self._phones.get(attachment_id)
        if phone is None:
            return 0
        try:
            await phone.ws.send_json(frame)
            return 1
        except Exception as exc:
            log.warning(
                "cast_input_rumble_send_failed",
                attachment_id=attachment_id,
                session_id=session_id, slot=slot, error=str(exc)[:160],
            )
            self._phones.pop(attachment_id, None)
            return 0

    # ── Slot bookkeeping ──────────────────────────────────────────────

    def _claim_slot_index(
        self, session_id: str, pad_index: int, attachment_id: str,
    ) -> int:
        """Reserve ``pad_index`` as the slot for this phone.

        If the slot is already taken (rare — two phones reporting
        pad_index 0 to the same session), fall through to the next
        free slot. Returns SLOT_UNCLAIMED when all 4 slots are full.
        """
        slots = self._slot_owners.setdefault(session_id, {})
        if pad_index not in slots:
            slots[pad_index] = attachment_id
            return pad_index
        for s in range(MAX_PADS_PER_SESSION):
            if s not in slots:
                slots[s] = attachment_id
                return s
        return SLOT_UNCLAIMED

    def _claim_slot_firstpress(
        self, session_id: str, attachment_id: str,
    ) -> int:
        """Reserve the next-lowest free slot for this phone."""
        slots = self._slot_owners.setdefault(session_id, {})
        for s in range(MAX_PADS_PER_SESSION):
            if s not in slots:
                slots[s] = attachment_id
                return s
        return SLOT_UNCLAIMED

    # ── Shutdown ──────────────────────────────────────────────────────

    async def close_all(self) -> None:
        """Close every WS gracefully at app shutdown."""
        containers = list(self._containers.values())
        phones = list(self._phones.values())
        self._containers.clear()
        self._phones.clear()
        self._slot_owners.clear()
        # Containers first so phones see the back-end go away before
        # they try to send their final frame.
        for c in containers:
            try:
                await c.ws.close(code=1001, reason="server shutdown")
            except Exception:
                log.debug("cast_input_container_close_failed", exc_info=True)
        for p in phones:
            try:
                await p.ws.close(code=1001, reason="server shutdown")
            except Exception:
                log.debug("cast_input_phone_close_failed", exc_info=True)


def _has_active_button(event: dict[str, Any]) -> bool:
    """Did this gamepad_state event press any button?

    A "button" here is any of the digital buttons OR a trigger axis
    deflected past a deadzone. Stick deflection alone does NOT claim
    a slot — pre-claim ergonomics matter: the user should have to
    actively press something, not just rest a thumb on the analog stick.
    """
    if event.get("kind") != "gamepad_state":
        return False
    buttons = event.get("buttons")
    if isinstance(buttons, list):
        for b in buttons:
            try:
                if float(b) > 0.5:
                    return True
            except (TypeError, ValueError):
                continue
    return False
