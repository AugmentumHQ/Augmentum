"""App-level scheduling substrate.

Timed actions (standing tasks) are a platform capability — briefings,
reminders, watches, deferred requests, scheduled verbs — created from
any surface (chat, voice, the Schedule UI) and dispatched whether or
not the companion runtime is enabled. The companion is one *entrypoint*
and one *dispatcher* among others, not the owner of the substrate.

See :mod:`augmentum.scheduling.service` for the dispatcher and
:mod:`augmentum.companion_runtime.standing_tasks` for the engine
(kept in place — every existing import path stays valid).
"""

from augmentum.scheduling.service import SchedulerService

__all__ = ["SchedulerService"]
