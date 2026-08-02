"""Surface sessions: shared control state for nearby devices.

The surface layer sits above concrete transports such as Cast, DLNA,
browser receivers, WebRTC streams, and Augmentum's own UI panes. A
surface session is the durable object they all meet around: who is
participating, what content is active, and what state every screen should
follow.
"""

from augmentum.surfaces.runtime import SurfaceRuntime
from augmentum.surfaces.store import SurfaceConflictError, SurfaceStore
from augmentum.surfaces.tokens import SurfaceAccessTokenStore

__all__ = [
    "SurfaceAccessTokenStore",
    "SurfaceConflictError",
    "SurfaceRuntime",
    "SurfaceStore",
]
