"""Default notification-channel catalog.

Channels live in Python rather than seed data so that renames,
deletions, and reorderings don't need follow-on migrations. The store
falls back to this catalog when no per-user row exists for a given
``channel_id``; the row materializes lazily the first time a user
customizes the channel (e.g. mutes it).

The importance ladder mirrors Android's NotificationChannel scale,
which the research turned up as the most user-comprehensible model:

* ``MIN``      — feed-only, never sound, hides on the status bar
* ``LOW``      — feed entry, no sound, no toast
* ``DEFAULT``  — feed entry + toast
* ``HIGH``     — feed entry + toast + sound
* ``CRITICAL`` — persistent banner + sound, pierces Do-Not-Disturb

Sources: Android NotificationChannel importance constants
(``IMPORTANCE_MIN``..``IMPORTANCE_HIGH``, plus ``IMPORTANCE_MAX`` for
critical-tier behavior). Apple's ``UNNotificationInterruptionLevel``
maps onto the same shape (passive / active / time-sensitive /
critical) and informed the persistence semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


IMPORTANCE_MIN = 0
IMPORTANCE_LOW = 1
IMPORTANCE_DEFAULT = 2
IMPORTANCE_HIGH = 3
IMPORTANCE_CRITICAL = 4

# Sentinel for unrecognised values — used by the store when reading
# back rows whose importance was written by a future version that
# extended the ladder. Treat as DEFAULT for behavioral decisions
# rather than crashing.
_UNKNOWN_IMPORTANCE_FALLBACK = IMPORTANCE_DEFAULT


def normalise_importance(value: int) -> int:
    """Clamp + sanitise an importance integer.

    Anything outside ``0..4`` is mapped to ``DEFAULT``. We don't
    error out on bad values — they're more likely to come from a
    forward-compat client than from a bug — but we also don't pass
    them through, because UI code keys on the ladder.
    """

    if not isinstance(value, int):
        return _UNKNOWN_IMPORTANCE_FALLBACK
    if value < IMPORTANCE_MIN or value > IMPORTANCE_CRITICAL:
        return _UNKNOWN_IMPORTANCE_FALLBACK
    return value


@dataclass(frozen=True)
class ChannelTemplate:
    """A system-default channel.

    Per-user customizations (importance change, mute) land in the
    ``notification_channels`` table and shadow the template at read
    time. The template stays authoritative for description + sound
    + name unless the user overrides those too.
    """

    channel_id: str
    name: str
    description: str
    importance: int
    default_sound: str = ""


# Default channel catalog. Order is the canonical surface ordering
# (used by UIs that list channels for muting/configuration).
DEFAULT_CHANNELS: tuple[ChannelTemplate, ...] = (
    # ── Connect ────────────────────────────────────────────────────
    ChannelTemplate(
        channel_id="connect.call.incoming",
        name="Incoming call",
        description=(
            "Someone is calling you. Critical so it pierces "
            "do-not-disturb and rings until accepted, declined, or "
            "timed out."
        ),
        importance=IMPORTANCE_CRITICAL,
        default_sound="ring",
    ),
    ChannelTemplate(
        channel_id="connect.call.missed",
        name="Missed call",
        description="Summary of a call you didn't pick up.",
        importance=IMPORTANCE_DEFAULT,
    ),
    ChannelTemplate(
        channel_id="connect.message",
        name="New message",
        description="A message arrived in a Connect thread.",
        importance=IMPORTANCE_DEFAULT,
        default_sound="ping",
    ),
    # ── Coder ──────────────────────────────────────────────────────
    ChannelTemplate(
        channel_id="coder.run.complete",
        name="Coder run finished",
        description="A coder run reached a terminal state successfully.",
        importance=IMPORTANCE_DEFAULT,
    ),
    ChannelTemplate(
        channel_id="coder.run.failed",
        name="Coder run failed",
        description=(
            "A coder run errored or needs review. Higher importance "
            "than 'complete' because it usually wants attention now."
        ),
        importance=IMPORTANCE_HIGH,
    ),
    # ── Companion ──────────────────────────────────────────────────
    ChannelTemplate(
        channel_id="companion.initiative",
        name="Companion initiative",
        description="Becca surfaced a proactive thought.",
        importance=IMPORTANCE_LOW,
    ),
    ChannelTemplate(
        channel_id="companion.observation",
        name="Companion observation",
        description="Becca noticed a pattern worth flagging.",
        importance=IMPORTANCE_LOW,
    ),
    ChannelTemplate(
        channel_id="companion.tasks",
        name="Scheduled tasks & briefings",
        description=(
            "Results of standing tasks you scheduled — daily briefings, "
            "reminders, watches. The channel default is a quiet ping; "
            "individual fires escalate per kind (briefings and reminders "
            "buzz every device, passive watches only buzz when no tab is "
            "open)."
        ),
        importance=IMPORTANCE_DEFAULT,
        default_sound="chime",
    ),
    # ── Timers ─────────────────────────────────────────────────────
    ChannelTemplate(
        channel_id="time.timer",
        name="Timers & reminders",
        description=(
            "A timer you set finished — including the result of any "
            "action it ran on completion."
        ),
        importance=IMPORTANCE_HIGH,
        default_sound="ping",
    ),
    # ── Home alerts (direct sources) ───────────────────────────────
    ChannelTemplate(
        channel_id="alerts.home",
        name="Home-area alerts",
        description=(
            "Severe weather warnings and significant earthquakes near "
            "your saved home location. Individual events escalate to "
            "critical when the issuing service marks them Extreme."
        ),
        importance=IMPORTANCE_HIGH,
    ),
    # ── Background work ────────────────────────────────────────────
    ChannelTemplate(
        channel_id="jobs.complete",
        name="Background job finished",
        description="A queued background job completed.",
        importance=IMPORTANCE_LOW,
    ),
    ChannelTemplate(
        channel_id="jobs.failed",
        name="Background job failed",
        description="A queued background job errored.",
        importance=IMPORTANCE_HIGH,
    ),
    # ── Models ─────────────────────────────────────────────────────
    ChannelTemplate(
        channel_id="models.load.complete",
        name="Model ready",
        description="A model finished loading and is available.",
        importance=IMPORTANCE_LOW,
    ),
    # ── System ─────────────────────────────────────────────────────
    ChannelTemplate(
        channel_id="system.update",
        name="System update",
        description="An update or maintenance notice for Augmentum itself.",
        importance=IMPORTANCE_DEFAULT,
    ),
    ChannelTemplate(
        channel_id="system.offer",
        name="Suggestion",
        description=(
            "Offers from the assistant — install an MCP server, switch "
            "mode, save a memory, etc. Each requires explicit Accept; "
            "nothing changes silently. Dismiss with Snooze (30 days) "
            "or Never (permanent, undoable from Settings)."
        ),
        importance=IMPORTANCE_DEFAULT,
    ),
    # ── External services ───────────────────────────────────────────
    ChannelTemplate(
        channel_id="service.alert",
        name="Service alert",
        description=(
            "A connected service (Uptime Kuma, Beszel, changedetection.io) "
            "reported an incident — a site is down, a page changed, a "
            "threshold was crossed. High importance so it surfaces above "
            "routine notifications."
        ),
        importance=IMPORTANCE_HIGH,
        default_sound="ping",
    ),
)


_CATALOG_BY_ID: dict[str, ChannelTemplate] = {
    ch.channel_id: ch for ch in DEFAULT_CHANNELS
}


def catalog_channel(channel_id: str) -> ChannelTemplate | None:
    """Look up a template by id. None if not registered."""

    return _CATALOG_BY_ID.get(channel_id)


def catalog_channel_ids() -> list[str]:
    """All registered template ids, in canonical order."""

    return [ch.channel_id for ch in DEFAULT_CHANNELS]
