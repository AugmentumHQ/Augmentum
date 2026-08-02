"""Auth data models."""

from __future__ import annotations

from dataclasses import dataclass

# ─── Reserved usernames ────────────────────────────────────────────────────
#
# Names that must not be registered to any real account because:
#   (a) the substrate may treat them as elevation tokens in future work
#       (an attacker registering ``internal-tool`` could silently match
#       any ``user.username == "internal-tool"`` admin check — see the
#       Odysseus reference threat-model entry on `internal-tool` loopback
#       elevation; Augmentum doesn't currently do this kind of check,
#       but reserving the name now closes the footgun before it can be
#       introduced);
#   (b) the substrate already provisions service identities under stable
#       prefixes (fabric peer service users live under ``fabric:<id>``
#       and ``fabric_peer_<hex>`` — see ``session_manager.py:278``);
#   (c) common admin/role names that operators would expect to belong to
#       the system rather than a real user (``admin``, ``system``,
#       ``root``, ``superuser``).
#
# Match is case-insensitive on the canonical (casefolded) username AND
# applies to prefixes for namespaces (fabric_*). Add new entries here,
# not by editing the scattered route handlers — the centralised list is
# the security contract.

# Exact reserved names (case-insensitive match against canonical form).
RESERVED_USERNAMES: frozenset[str] = frozenset({
    # System / role names
    "system", "root", "superuser", "admin",
    # Service identities
    "api", "service", "daemon", "bot",
    # Internal loopback / elevation tokens (defensive — close the door
    # before it's even built)
    "internal", "internal-tool", "internal_tool", "internaltool",
    # Brand / persona names that operators would expect to belong to
    # the install, not a real user
    "augmentum", "becca",
    # Anonymous / placeholder
    "anonymous", "guest", "nobody", "unknown",
    # Test / demo naming the operator might assume is internal
    "demo", "test",
})

# Reserved prefixes — any username starting with these (case-insensitive)
# is rejected. Used for namespaces the substrate provisions
# automatically.
RESERVED_USERNAME_PREFIXES: tuple[str, ...] = (
    # Fabric peer service users — provisioned by FabricPeerMiddleware
    # via ``SessionManager.get_or_create_fabric_peer_user``. A real
    # account squatting one of these would block legitimate peer auth.
    "fabric_peer_",
    "fabric:",
    # Future-proof: usr_ is the DB primary-key prefix on user.id, NOT
    # username; but reserving it as a username prefix prevents an
    # operator from confusing the two columns.
    "usr_",
)


def is_reserved_username(name: str) -> bool:
    """Return True if ``name`` (any casing) is a reserved username.

    Decision order:
      1. Empty / None → not reserved (registration regex rejects empties).
      2. Casefolded name in :data:`RESERVED_USERNAMES` → reserved.
      3. Casefolded name starts with any :data:`RESERVED_USERNAME_PREFIXES`
         entry → reserved.
      4. Otherwise → not reserved.

    Callers should reject reserved names with a 400 (registration) or
    raise ``ValueError`` (lower-level CRUD).
    """
    if not name:
        return False
    canon = name.casefold().strip()
    if not canon:
        return False
    if canon in RESERVED_USERNAMES:
        return True
    for prefix in RESERVED_USERNAME_PREFIXES:
        if canon.startswith(prefix.casefold()):
            return True
    return False


@dataclass
class User:
    id: str
    username: str
    display_name: str = ""
    role: str = "user"  # "admin" or "user"
    is_active: bool = True
    # Optional contact email — populated by invite onboarding (the invitee may
    # carry one) and future account recovery. Empty for accounts created via
    # the setup wizard / admin-create. See migration 280.
    email: str = ""
    quota_bytes: int = 0
    created_at: str = ""
    updated_at: str = ""
    # Content filtering level — set by admins for family/younger-user
    # accounts. 'unrestricted' = no filtering (default). 'family' = server
    # forces SFW on character-import search (chub.ai + risurealm)
    # regardless of client toggles. See migration 191.
    content_level: str = "unrestricted"
    # Which client the presenting auth session was created from
    # (auth_sessions.source: web / android / cast_receiver / ...).
    # Per-session, not per-user — the same account carries "web" on a
    # laptop and "cast_receiver" on the TV. Attention emit sites stamp
    # this onto bus payloads so the topical aggregator can filter by
    # `companion_attention_sources` (notes v2 provenance).
    session_source: str = "web"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_family_filtered(self) -> bool:
        """True when external content sources must enforce SFW for this user."""
        return self.content_level == "family"

    def to_public_dict(self) -> dict:
        """Return user info safe for API responses (no password hash)."""
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "is_active": self.is_active,
            "email": self.email,
            "created_at": self.created_at,
            "content_level": self.content_level,
        }


@dataclass
class AuthSession:
    token: str
    user_id: str
    created_at: str = ""
    expires_at: str = ""
    last_activity: str = ""
    ip_address: str = ""
    user_agent: str = ""


@dataclass
class WsTicket:
    ticket: str
    user_id: str
    expires_at: float = 0.0  # monotonic time
