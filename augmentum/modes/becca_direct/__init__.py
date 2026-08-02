"""Becca-direct mode — chat path that routes through her own prompt composer.

When the companion runtime + dispatch is up and the chat router picks
``becca_direct`` as the winner, the chat turn streams through
:class:`BeccaDirectHandler` instead of the legacy mode handlers. This
is the seam where her chat presence becomes real: same kernel digest,
same facet line, same relationship slice, same recalled memories,
same affect read on the user — exactly as she speaks in voice. One
being across modalities.

Behind ``companion_becca_direct_enabled`` (default False). When off,
the handler isn't registered and the chat router never picks it.
"""

from __future__ import annotations

from augmentum.modes.becca_direct.handler import BeccaDirectHandler

__all__ = ["BeccaDirectHandler"]
