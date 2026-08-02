"""Behavior modules — Sprint 4a.

Each module here is one mechanical aspect of "Becca acting on her own
when nobody asks." They're event-driven and bounded:
- :mod:`tick` is the scheduler hook (registered with JobRunner).
- :mod:`activity_selector` ranks candidate things-to-do.
- :mod:`role_channel` decides which role Becca occupies right now.
- :mod:`initiative` writes proposals to ``companion_initiative_queue``.
- :mod:`honest_gap` keeps Becca from confabulating beyond what she knows.
- :mod:`sleep_wake` bridges to the existing dream subsystem.

All of this is flag-gated on ``companion_tick_enabled``. With the flag
off, ``tick()`` returns immediately and no proposals are written.
"""
