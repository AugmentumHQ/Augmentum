"""Driver implementations for the device substrate.

Each module under this package implements one wire protocol and conforms
to the ``DeviceDriver`` Protocol defined in ``augmentum.devices.driver``.
Drivers are registered against a ``DeviceRegistry`` at app startup.
"""

from __future__ import annotations

from augmentum.devices.drivers.cast_custom import CastCustomDriver
from augmentum.devices.drivers.dlna import DlnaDriver
from augmentum.devices.drivers.emby_remote import EmbyRemoteDriver

__all__ = ["CastCustomDriver", "DlnaDriver", "EmbyRemoteDriver"]
