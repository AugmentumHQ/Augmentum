"""Jellyfin media-server provider."""

from __future__ import annotations

from augmentum.media.providers.emby_compat import EmbyCompatBase
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_SETUP_TIMEOUT = 15.0


class JellyfinProvider(EmbyCompatBase):
    name = "jellyfin"
    api_prefix = ""
    auth_scheme = "MediaBrowser"
    item_list_path_uses_user = False
    user_views_path_uses_user = False
    user_data_path_uses_user = False

    async def first_run_setup(
        self, base_url: str, username: str, password: str,
    ) -> None:
        """Idempotently complete Jellyfin's first-run wizard.

        A freshly provisioned Jellyfin boots into a setup wizard with NO
        users — ``/Users/AuthenticateByName`` (login) can't work until an
        admin exists. This drives the open ``/Startup/*`` endpoints to
        create the initial admin with the Augmentum-managed credential,
        then the dispatcher logs in normally.

        No-op if the wizard is already complete (re-install / restart):
        ``/System/Info/Public`` reports ``StartupWizardCompleted`` without
        auth, so we check that first. The ``/Startup`` endpoints need no
        token but do want the client-identity Authorization header.
        """
        base = base_url.rstrip("/")
        headers = self._headers("")

        # Already set up? Then there's nothing to do — login handles the rest.
        try:
            info = await self._http.get(
                f"{base}/System/Info/Public", headers=headers,
                timeout=_SETUP_TIMEOUT,
            )
            if info.status_code == 200 and info.json().get("StartupWizardCompleted"):
                return
        except Exception:  # noqa: BLE001 — not ready yet; attempt the wizard
            pass

        # 1) Initial server configuration (locale/metadata).
        await self._http.post(
            f"{base}/Startup/Configuration", headers=headers,
            json={
                "UICulture": "en-US",
                "MetadataCountryCode": "US",
                "PreferredMetadataLanguage": "en",
            },
            timeout=_SETUP_TIMEOUT,
        )
        # 2) Touch the wizard user endpoint (Jellyfin initializes its state).
        await self._http.get(
            f"{base}/Startup/User", headers=headers, timeout=_SETUP_TIMEOUT,
        )
        # 3) Create the initial admin account with the managed credential.
        await self._http.post(
            f"{base}/Startup/User", headers=headers,
            json={"Name": username, "Password": password},
            timeout=_SETUP_TIMEOUT,
        )
        # 4) Remote access on (LAN), no automatic UPnP port mapping.
        await self._http.post(
            f"{base}/Startup/RemoteAccess", headers=headers,
            json={"EnableRemoteAccess": True, "EnableAutomaticPortMapping": False},
            timeout=_SETUP_TIMEOUT,
        )
        # 5) Finish the wizard.
        await self._http.post(
            f"{base}/Startup/Complete", headers=headers, timeout=_SETUP_TIMEOUT,
        )
        log.info("jellyfin_first_run_setup_complete", base_url=base)
