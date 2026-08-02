"""Mobile pairing state and trusted-device persistence.

This intentionally mirrors the cast receiver QR ceremony without sharing the
cast receiver's final credential. Cast redeems a pair token into a receiver
cookie; Android redeems a one-time mobile grant into a scoped auth session
bound to a durable trusted_mobile_devices row.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


STATE_PENDING = "pending"
STATE_CLAIMED = "claimed"
STATE_APPROVED = "approved"
STATE_CONSUMED = "consumed"
STATE_EXPIRED = "expired"

_DEFAULT_TTL_S = 180.0
_MAX_ACTIVE_RECORDS = 128
_PAIR_CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _pair_code(length: int = 8) -> str:
    return "".join(secrets.choice(_PAIR_CODE_CHARS) for _ in range(length))


def _token(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(24)}"


def _trim(raw: str, max_len: int) -> str:
    return (raw or "").strip()[:max_len]


def _clean_string_list(values: list[str] | None, *, max_items: int = 64) -> list[str]:
    out: list[str] = []
    for value in values or []:
        item = _trim(str(value), 120)
        if item and item not in out:
            out.append(item)
        if len(out) >= max_items:
            break
    return out


@dataclass(slots=True)
class MobilePairRecord:
    pair_code: str
    user_id: str
    state: str
    expires_at: float
    claim_token: str = ""
    grant_token: str = ""
    device_id: str = ""
    label: str = ""
    platform: str = "android"
    app_version: str = ""
    public_key: str = ""
    key_alg: str = ""
    user_agent: str = ""
    scopes: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    claimed_at: float = 0.0
    approved_at: float = 0.0
    consumed_at: float = 0.0

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at

    def expires_in(self) -> int:
        return max(0, int(self.expires_at - time.time()))

    def claim_dict(self) -> dict[str, Any]:
        if not self.device_id:
            return {}
        return {
            "device_id": self.device_id,
            "label": self.label,
            "platform": self.platform,
            "app_version": self.app_version,
            "key_alg": self.key_alg,
            "scopes": list(self.scopes),
            "capabilities": list(self.capabilities),
            "claimed_at": self.claimed_at,
        }

    def public_status(self, *, include_grant: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "pair_code": self.pair_code,
            "state": self.state,
            "expires_in": self.expires_in(),
            "claim": self.claim_dict(),
        }
        if include_grant and self.state == STATE_APPROVED and self.grant_token:
            out["grant_token"] = self.grant_token
        return out


class MobilePairStore:
    """Process-local pending-pair store for Android/mobile setup."""

    def __init__(
        self,
        *,
        default_ttl_s: float = _DEFAULT_TTL_S,
        max_active: int = _MAX_ACTIVE_RECORDS,
    ) -> None:
        self._records: dict[str, MobilePairRecord] = {}
        self._claims: dict[str, str] = {}
        self._grants: dict[str, str] = {}
        self._default_ttl = max(30.0, float(default_ttl_s or _DEFAULT_TTL_S))
        self._max_active = max(1, int(max_active or _MAX_ACTIVE_RECORDS))

    def start(self, *, user_id: str, ttl_s: float | None = None) -> MobilePairRecord:
        if not user_id:
            raise ValueError("user_id required")
        self._prune()
        for _ in range(8):
            code = _pair_code()
            if code not in self._records:
                break
        else:
            code = _pair_code(12)
        ttl = float(ttl_s) if ttl_s and ttl_s > 0 else self._default_ttl
        record = MobilePairRecord(
            pair_code=code,
            user_id=user_id,
            state=STATE_PENDING,
            expires_at=time.time() + ttl,
        )
        self._records[code] = record
        log.info("mobile_pair_started", pair_code=code, user_id=user_id)
        return record

    def get_for_user(self, pair_code: str, *, user_id: str) -> MobilePairRecord | None:
        record = self.poll_code(pair_code)
        if record is None or record.user_id != user_id:
            return None
        return record

    def claim(
        self,
        pair_code: str,
        *,
        device_id: str,
        label: str = "",
        platform: str = "android",
        app_version: str = "",
        public_key: str = "",
        key_alg: str = "",
        scopes: list[str] | None = None,
        capabilities: list[str] | None = None,
        user_agent: str = "",
    ) -> MobilePairRecord | None:
        record = self.poll_code(pair_code)
        if record is None or record.state != STATE_PENDING:
            return None
        device_id = _trim(device_id, 160)
        if not device_id:
            return None
        record.state = STATE_CLAIMED
        record.claim_token = _token("mpc")
        record.device_id = device_id
        record.label = _trim(label, 120) or "Android phone"
        record.platform = _trim(platform, 40) or "android"
        record.app_version = _trim(app_version, 80)
        record.public_key = _trim(public_key, 4096)
        record.key_alg = _trim(key_alg, 80)
        record.user_agent = _trim(user_agent, 200)
        record.scopes = _clean_string_list(scopes)
        record.capabilities = _clean_string_list(capabilities)
        record.claimed_at = time.time()
        self._claims[record.claim_token] = record.pair_code
        log.info(
            "mobile_pair_claimed",
            pair_code=pair_code,
            user_id=record.user_id,
            device_id=device_id,
            platform=record.platform,
        )
        return record

    def approve(self, pair_code: str, *, user_id: str) -> MobilePairRecord | None:
        record = self.get_for_user(pair_code, user_id=user_id)
        if record is None or record.state != STATE_CLAIMED:
            return None
        record.state = STATE_APPROVED
        record.grant_token = _token("mpg")
        record.approved_at = time.time()
        self._grants[record.grant_token] = record.pair_code
        log.info(
            "mobile_pair_approved",
            pair_code=pair_code,
            user_id=user_id,
            device_id=record.device_id,
        )
        return record

    def poll_code(self, pair_code: str) -> MobilePairRecord | None:
        record = self._records.get(pair_code)
        if record is None:
            return None
        if record.is_expired() and record.state in (
            STATE_PENDING,
            STATE_CLAIMED,
            STATE_APPROVED,
        ):
            record.state = STATE_EXPIRED
        return record

    def poll_claim(self, claim_token: str) -> MobilePairRecord | None:
        pair_code = self._claims.get(claim_token)
        if not pair_code:
            return None
        return self.poll_code(pair_code)

    def consume_grant(self, grant_token: str) -> MobilePairRecord | None:
        pair_code = self._grants.pop(grant_token, None)
        if not pair_code:
            return None
        record = self.poll_code(pair_code)
        if record is None or record.state != STATE_APPROVED:
            return None
        record.state = STATE_CONSUMED
        record.consumed_at = time.time()
        if record.claim_token:
            self._claims.pop(record.claim_token, None)
        log.info(
            "mobile_pair_grant_consumed",
            pair_code=pair_code,
            user_id=record.user_id,
            device_id=record.device_id,
        )
        return record

    def _prune(self) -> None:
        now = time.time()
        expired: list[str] = []
        for code, record in self._records.items():
            if record.expires_at <= now or record.state == STATE_CONSUMED and record.consumed_at:
                expired.append(code)
        for code in expired:
            record = self._records.pop(code, None)
            if record is None:
                continue
            if record.claim_token:
                self._claims.pop(record.claim_token, None)
            if record.grant_token:
                self._grants.pop(record.grant_token, None)
        if len(self._records) < self._max_active:
            return
        ordered = sorted(self._records.items(), key=lambda item: item[1].expires_at)
        overflow = len(self._records) - self._max_active + 1
        for code, record in ordered[:overflow]:
            self._records.pop(code, None)
            if record.claim_token:
                self._claims.pop(record.claim_token, None)
            if record.grant_token:
                self._grants.pop(record.grant_token, None)


@dataclass(slots=True)
class TrustedMobileDevice:
    id: str
    user_id: str
    device_id: str
    label: str
    platform: str
    app_version: str
    public_key: str
    key_alg: str
    scopes: list[str]
    capabilities: list[str]
    created_at: str
    updated_at: str
    last_seen_at: str
    revoked_at: str

    @property
    def revoked(self) -> bool:
        return bool(self.revoked_at)

    def to_dict(self, *, include_user_id: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "device_id": self.device_id,
            "label": self.label or self.device_id,
            "platform": self.platform,
            "app_version": self.app_version,
            "key_alg": self.key_alg,
            "scopes": list(self.scopes),
            "capabilities": list(self.capabilities),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_seen_at": self.last_seen_at,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
        }
        if include_user_id:
            out["user_id"] = self.user_id
        return out


def _json_list(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return _clean_string_list([str(item) for item in parsed])


def _json_dumps_list(values: list[str]) -> str:
    return json.dumps(_clean_string_list(values), separators=(",", ":"), sort_keys=True)


def _row_to_mobile(row: Any) -> TrustedMobileDevice:
    return TrustedMobileDevice(
        id=row[0],
        user_id=row[1],
        device_id=row[2],
        label=row[3] or "",
        platform=row[4] or "android",
        app_version=row[5] or "",
        public_key=row[6] or "",
        key_alg=row[7] or "",
        scopes=_json_list(row[8]),
        capabilities=_json_list(row[9]),
        created_at=row[10] or "",
        updated_at=row[11] or "",
        last_seen_at=row[12] or "",
        revoked_at=row[13] or "",
    )


class TrustedMobileDeviceStore:
    """User-scoped CRUD over trusted_mobile_devices."""

    _SELECT = (
        "id, user_id, device_id, label, platform, app_version, public_key, "
        "key_alg, scopes_json, capabilities_json, created_at, updated_at, "
        "last_seen_at, revoked_at"
    )

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def upsert_from_pair(self, record: MobilePairRecord) -> TrustedMobileDevice:
        mobile_id = f"md_{secrets.token_hex(8)}"
        now = _now_iso()
        await self._conn.execute(
            """INSERT INTO trusted_mobile_devices
                  (id, user_id, device_id, label, platform, app_version,
                   public_key, key_alg, scopes_json, capabilities_json,
                   created_at, updated_at, last_seen_at, revoked_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')
               ON CONFLICT(user_id, device_id) WHERE device_id != ''
               DO UPDATE SET
                   label = excluded.label,
                   platform = excluded.platform,
                   app_version = excluded.app_version,
                   public_key = excluded.public_key,
                   key_alg = excluded.key_alg,
                   scopes_json = excluded.scopes_json,
                   capabilities_json = excluded.capabilities_json,
                   updated_at = excluded.updated_at,
                   last_seen_at = excluded.last_seen_at,
                   revoked_at = ''""",
            (
                mobile_id,
                record.user_id,
                record.device_id,
                record.label,
                record.platform,
                record.app_version,
                record.public_key,
                record.key_alg,
                _json_dumps_list(record.scopes),
                _json_dumps_list(record.capabilities),
                now,
                now,
                now,
            ),
        )
        await self._conn.commit()
        device = await self.get_by_device(record.device_id, user_id=record.user_id)
        if device is None:
            raise RuntimeError("trusted mobile device upsert failed")
        log.info(
            "trusted_mobile_device_upserted",
            user_id=record.user_id,
            device_id=record.device_id,
            mobile_id=device.id,
        )
        return device

    async def list_for_user(
        self,
        *,
        user_id: str,
        include_revoked: bool = False,
    ) -> list[TrustedMobileDevice]:
        if not user_id:
            return []
        query = f"SELECT {self._SELECT} FROM trusted_mobile_devices WHERE user_id = ?"
        if not include_revoked:
            query += " AND revoked_at = ''"
        query += " ORDER BY last_seen_at DESC, created_at DESC"
        cursor = await self._conn.execute(query, (user_id,))
        rows = await cursor.fetchall()
        return [_row_to_mobile(row) for row in rows]

    async def get(
        self,
        mobile_id: str,
        *,
        user_id: str,
        include_revoked: bool = False,
    ) -> TrustedMobileDevice | None:
        query = (
            f"SELECT {self._SELECT} FROM trusted_mobile_devices "
            "WHERE id = ? AND user_id = ?"
        )
        params: tuple[Any, ...] = (mobile_id, user_id)
        if not include_revoked:
            query += " AND revoked_at = ''"
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        return _row_to_mobile(row) if row else None

    async def get_by_device(
        self,
        device_id: str,
        *,
        user_id: str,
        include_revoked: bool = False,
    ) -> TrustedMobileDevice | None:
        query = (
            f"SELECT {self._SELECT} FROM trusted_mobile_devices "
            "WHERE user_id = ? AND device_id = ?"
        )
        if not include_revoked:
            query += " AND revoked_at = ''"
        cursor = await self._conn.execute(query, (user_id, device_id))
        row = await cursor.fetchone()
        return _row_to_mobile(row) if row else None

    async def touch_seen(self, *, user_id: str, device_id: str) -> None:
        if not user_id or not device_id:
            return
        await self._conn.execute(
            """UPDATE trusted_mobile_devices
               SET last_seen_at = ?, updated_at = ?
               WHERE user_id = ? AND device_id = ? AND revoked_at = ''""",
            (_now_iso(), _now_iso(), user_id, device_id),
        )
        await self._conn.commit()

    async def revoke(self, mobile_id: str, *, user_id: str) -> TrustedMobileDevice | None:
        existing = await self.get(mobile_id, user_id=user_id)
        if existing is None:
            return None
        now = _now_iso()
        await self._conn.execute(
            """UPDATE trusted_mobile_devices
               SET revoked_at = ?, updated_at = ?
               WHERE id = ? AND user_id = ?""",
            (now, now, mobile_id, user_id),
        )
        await self._conn.commit()
        log.info(
            "trusted_mobile_device_revoked",
            user_id=user_id,
            mobile_id=mobile_id,
            device_id=existing.device_id,
        )
        return await self.get(mobile_id, user_id=user_id, include_revoked=True)

