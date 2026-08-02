"""Auth / rate-limit / session-isolation settings — the security
perimeter of the proxy.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting

_AUTH = ("auth", "security")
_AUTH_ADV = ("auth", "security", "advanced")
_RATE = ("rate-limit", "security")


def register(r: SettingsRegistry) -> None:
    # ============== Auth lockouts ==============
    r.register(
        Setting(
            key="auth_session_ttl_hours",
            kind="int",
            default=720,  # 30 days
            label="Session TTL (hours)",
            description=(
                "How long an authenticated session stays valid before "
                "requiring re-login. Default 720 = 30 days."
            ),
            section="auth.session",
            min_value=1,
            max_value=8760,
            tags=_AUTH,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="auth_lockout_threshold",
            kind="int",
            default=5,
            label="Login lockout threshold",
            description=(
                "Failed login attempts on a single user before the account "
                "is temporarily locked."
            ),
            section="auth.lockout",
            min_value=1,
            max_value=100,
            tags=_AUTH,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="auth_lockout_minutes",
            kind="int",
            default=15,
            label="Login lockout duration (min)",
            description=(
                "How long an account stays locked after exceeding the "
                "failed-login threshold."
            ),
            section="auth.lockout",
            min_value=1,
            max_value=1440,
            tags=_AUTH,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="auth_ip_lockout_threshold",
            kind="int",
            default=10,
            label="IP lockout threshold",
            description=(
                "Failed login attempts from a single IP (across any user) "
                "before the IP is temporarily blocked."
            ),
            section="auth.lockout",
            min_value=1,
            max_value=1000,
            tags=_AUTH,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="auth_ip_lockout_minutes",
            kind="int",
            default=60,
            label="IP lockout duration (min)",
            description=(
                "How long an IP stays blocked after exceeding the IP "
                "failed-login threshold."
            ),
            section="auth.lockout",
            min_value=1,
            max_value=1440,
            tags=_AUTH,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="auth_ws_ticket_ttl_seconds",
            kind="int",
            default=30,
            label="WebSocket ticket TTL (s)",
            description=(
                "How long a short-lived WebSocket authentication ticket is "
                "valid. Tickets are single-use; lower = tighter window."
            ),
            section="auth.ws",
            min_value=5,
            max_value=300,
            tags=_AUTH_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="auth_max_sessions_per_user",
            kind="int",
            default=10,
            label="Max sessions per user",
            description=(
                "Concurrent active sessions cap per user. Oldest sessions "
                "are evicted when the cap is exceeded."
            ),
            section="auth.session",
            min_value=1,
            max_value=100,
            tags=_AUTH,
            trust_tier="admin_only",
        )
    )

    # ============== Rate limits ==============
    r.register(
        Setting(
            key="rate_limit_enabled",
            kind="bool",
            default=False,
            label="Rate limiting",
            description=(
                "Enforce per-IP request-rate limits across chat / image / "
                "voice. Off by default — opt-in for public deployments."
            ),
            section="security.rate_limit",
            tags=_RATE,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="rate_limit_chat_rpm",
            kind="int",
            default=30,
            label="Chat RPM",
            description=(
                "Per-IP chat-request rate limit (requests per minute) when "
                "rate limiting is enabled."
            ),
            section="security.rate_limit",
            min_value=1,
            max_value=300,
            tags=_RATE,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="rate_limit_image_rpm",
            kind="int",
            default=10,
            label="Image RPM",
            description=(
                "Per-IP image-generation rate limit. Image is more expensive "
                "than chat — keep this tighter."
            ),
            section="security.rate_limit",
            min_value=1,
            max_value=100,
            tags=_RATE,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="rate_limit_voice_rpm",
            kind="int",
            default=5,
            label="Voice RPM",
            description=(
                "Per-IP voice-pipeline rate limit. Voice is even more "
                "expensive than image — keep this tightest."
            ),
            section="security.rate_limit",
            min_value=1,
            max_value=50,
            tags=_RATE,
            trust_tier="admin_only",
        )
    )

    # ============== Session isolation ==============
    r.register(
        Setting(
            key="session_client_isolation",
            kind="bool",
            default=False,
            label="Client-isolated sessions",
            description=(
                "Scope chat sessions by client identity (IP / header) so "
                "the same user on different devices doesn't share session "
                "state. Off = sessions are user-scoped only."
            ),
            section="auth.session",
            tags=_AUTH_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
