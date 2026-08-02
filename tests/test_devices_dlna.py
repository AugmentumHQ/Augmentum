"""DLNA driver smoke tests.

The driver wraps existing dependency-free DLNA helpers in
``augmentum/media/receivers/dlna.py``. We don't talk to a real renderer
here — the underlying SOAP/SSDP helpers are stubbed and we verify that
the driver dispatches each action to the right wire call with the right
arguments.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from augmentum.devices.device import Device
from augmentum.devices.drivers.dlna import (
    DlnaDriver,
    _capabilities_for_receiver,
    _device_to_receiver,
)
from augmentum.devices.invocation import InvocationContext
from augmentum.media.receivers.dlna import DlnaReceiver


def _make_device(*, with_av_url: bool = True) -> Device:
    address: dict[str, Any] = {
        "host": "192.168.1.42",
        "port": 8200,
        "location": "http://192.168.1.42:8200/description.xml",
        "rendering_control_url": "http://192.168.1.42:8200/RenderingControl",
        "supported_commands": ["Play", "Pause", "Stop", "Seek", "SetVolume"],
    }
    if with_av_url:
        address["av_transport_url"] = "http://192.168.1.42:8200/AVTransport"
    return Device(
        id="dev_dlna_test",
        user_id="user_alice",
        driver="dlna",
        native_id="UDN:test",
        label="Living Room TV",
        capabilities=["media.video_play@1", "media.audio_play@1", "display.image_show@1"],
        address=address,
        metadata={"manufacturer": "Sony", "model_name": "Bravia X900"},
    )


class TestDlnaDriverProtocol:
    def test_declares_capabilities(self):
        d = DlnaDriver()
        assert "media.video_play@1" in d.capabilities
        assert "media.audio_play@1" in d.capabilities
        assert "display.image_show@1" in d.capabilities

    def test_metadata_fields(self):
        d = DlnaDriver()
        assert d.id == "dlna"
        assert d.requires_pairing is False
        assert "ssdp" in d.discovery_modes


class TestReceiverReconstruction:
    def test_device_to_receiver_full_address(self):
        device = _make_device(with_av_url=True)
        r = _device_to_receiver(device)
        assert r is not None
        assert r.av_transport_url == "http://192.168.1.42:8200/AVTransport"
        assert r.rendering_control_url == "http://192.168.1.42:8200/RenderingControl"
        assert r.label == "Living Room TV"
        assert r.manufacturer == "Sony"

    def test_device_to_receiver_returns_none_without_av_url(self):
        device = _make_device(with_av_url=False)
        assert _device_to_receiver(device) is None

    def test_capabilities_for_receiver_full_set(self):
        r = DlnaReceiver(
            receiver_id="x",
            label="Test",
            location="http://x",
            av_transport_url="http://x/AV",
        )
        caps = _capabilities_for_receiver(r)
        assert "media.video_play@1" in caps
        assert "media.audio_play@1" in caps
        assert "display.image_show@1" in caps


class TestDlnaInvocations:
    @pytest.mark.asyncio
    async def test_invoke_play_dispatches_launch(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device()
        ctx = InvocationContext(user_id="user_alice")

        # Driver builds DIDL + sends SetAVTransportURI then Play via SOAP.
        # Patch the underlying SOAP helper so we can assert both calls
        # happened with the expected arguments.
        with patch(
            "augmentum.media.receivers.dlna._soap_call",
            new_callable=AsyncMock, return_value=object(),  # any non-None
        ) as mock_soap, patch(
            "augmentum.devices.drivers.dlna.snapshot_dlna_receiver",
            new_callable=AsyncMock, return_value={"current_time_s": 0},
        ):
            result = await driver.invoke(
                device, "media.video_play@1", "play",
                {
                    "content_url": "http://x/y.mp4",
                    "title": "Movie",
                    "poster_url": "http://x/cover.jpg",
                    "author": "Director Name",
                },
                ctx,
            )

        assert result.ok is True, result
        # Two SOAP calls fire: SetAVTransportURI then Play.
        actions = [c.kwargs["action"] for c in mock_soap.call_args_list]
        assert "SetAVTransportURI" in actions
        assert "Play" in actions
        # The DIDL should embed the title, poster, and author.
        set_uri_call = next(c for c in mock_soap.call_args_list
                            if c.kwargs["action"] == "SetAVTransportURI")
        didl = set_uri_call.kwargs["arguments"]["CurrentURIMetaData"]
        assert "Movie" in didl
        assert "Director Name" in didl
        assert "cover.jpg" in didl

    @pytest.mark.asyncio
    async def test_invoke_pause_sends_playstate(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device()
        ctx = InvocationContext(user_id="user_alice")

        with patch(
            "augmentum.devices.drivers.dlna.send_dlna_playstate_command",
            new_callable=AsyncMock, return_value=True,
        ) as mock_send:
            result = await driver.invoke(
                device, "media.video_play@1", "pause", {}, ctx,
            )

        assert result.ok is True
        assert mock_send.call_args.kwargs["command"] == "Pause"

    @pytest.mark.asyncio
    async def test_invoke_resume_uses_unpause_command(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device()
        ctx = InvocationContext(user_id="user_alice")

        with patch(
            "augmentum.devices.drivers.dlna.send_dlna_playstate_command",
            new_callable=AsyncMock, return_value=True,
        ) as mock_send:
            await driver.invoke(device, "media.video_play@1", "resume", {}, ctx)

        assert mock_send.call_args.kwargs["command"] == "Unpause"

    @pytest.mark.asyncio
    async def test_invoke_seek_passes_position(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device()
        ctx = InvocationContext(user_id="user_alice")

        with patch(
            "augmentum.devices.drivers.dlna.send_dlna_playstate_command",
            new_callable=AsyncMock, return_value=True,
        ) as mock_send:
            await driver.invoke(
                device, "media.video_play@1", "seek",
                {"position_s": 137.5}, ctx,
            )

        assert mock_send.call_args.kwargs["command"] == "Seek"
        assert mock_send.call_args.kwargs["seek_position_s"] == 137.5

    @pytest.mark.asyncio
    async def test_invoke_set_volume(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device()
        ctx = InvocationContext(user_id="user_alice")

        with patch(
            "augmentum.devices.drivers.dlna.send_dlna_general_command",
            new_callable=AsyncMock, return_value=True,
        ) as mock_send:
            await driver.invoke(
                device, "media.video_play@1", "set_volume",
                {"level": 42}, ctx,
            )

        assert mock_send.call_args.kwargs["command"] == "SetVolume"
        assert mock_send.call_args.kwargs["arguments"] == {"Volume": 42}

    @pytest.mark.asyncio
    async def test_invoke_set_mute_routes_correctly(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device()
        ctx = InvocationContext(user_id="user_alice")

        with patch(
            "augmentum.devices.drivers.dlna.send_dlna_general_command",
            new_callable=AsyncMock, return_value=True,
        ) as mock_send:
            await driver.invoke(device, "media.video_play@1", "set_mute", {"muted": True}, ctx)
            assert mock_send.call_args.kwargs["command"] == "Mute"

            await driver.invoke(device, "media.video_play@1", "set_mute", {"muted": False}, ctx)
            assert mock_send.call_args.kwargs["command"] == "Unmute"

    @pytest.mark.asyncio
    async def test_invoke_returns_failure_on_no_av_url(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device(with_av_url=False)
        ctx = InvocationContext(user_id="user_alice")
        result = await driver.invoke(
            device, "media.video_play@1", "play",
            {"content_url": "http://x/y.mp4"}, ctx,
        )
        assert result.ok is False
        assert result.code == "device_address_invalid"

    @pytest.mark.asyncio
    async def test_invoke_play_requires_content_url(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device()
        ctx = InvocationContext(user_id="user_alice")
        result = await driver.invoke(
            device, "media.video_play@1", "play", {}, ctx,
        )
        assert result.ok is False
        assert result.code == "missing_arg"

    @pytest.mark.asyncio
    async def test_invoke_unsupported_action(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device()
        ctx = InvocationContext(user_id="user_alice")
        result = await driver.invoke(
            device, "media.video_play@1", "warp_drive", {}, ctx,
        )
        assert result.ok is False
        assert result.code == "unsupported_action"


class TestDlnaPairing:
    @pytest.mark.asyncio
    async def test_pair_start_returns_active(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device()
        ctx = InvocationContext(user_id="user_alice")
        result = await driver.pair_start(device, ctx)
        assert result.state == "active"
        assert result.requires_user_action is False

    @pytest.mark.asyncio
    async def test_pair_complete_returns_active(self):
        driver = DlnaDriver(http_client=object())
        device = _make_device()
        ctx = InvocationContext(user_id="user_alice")
        result = await driver.pair_complete(device, "anything", ctx)
        assert result.state == "active"
